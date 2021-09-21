import os
import warnings
from typing import Sequence, Union
from collections import defaultdict
from itertools import cycle, chain
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.modules.activation import PReLU

from ..utils import savefig
from ..optim import DummyLR, OneCycleLR, LinearLR
from ....utils.log_utl import print_progress_bar
from ....utils.os_utl import check_options, check_types, check_interval
from ....utils.stats_utl import moving_average
from ....utils.generic_utl import get_magnitude


# todo implement wrapper class to perform Parallel
# todo introduce ignite. implement early stopping, metrics
#  test _dynamic_dropout
class Engine(nn.Module):
    """
    Engine to run Neural Networks in PyTorch
    This class provides methods to fit, create_dataloader, save and load state

    :param string save_path: Path to folder to save checkpoints and other results to
    :param int n_jobs: Number of processes to use in the computations. GPU parallelization is not handled yet.
    """

    @check_types(save_path=str, n_jobs=int, clip_grad=bool, clip_value=(int, float))
    def __init__(self, device='auto', clip_grad=False, clip_value=10, save_path='.', n_jobs=-1, random_state=0,
                 **kwargs):
        super(Engine, self).__init__()

        self.train_loader = None
        self.validation_loader = None
        self.loss = Losses(('train', 'validation'))
        self.metrics = dict()

        self.n_jobs = min(n_jobs, os.cpu_count()) if n_jobs >= 1 else os.cpu_count()
        self.clip_grad = clip_grad
        self.clip_value = clip_value if clip_grad else None
        self.save_path = save_path

        self.device = None
        self.set_device(device)
        torch.set_num_threads(self.n_jobs)
        torch.manual_seed(random_state)

    @check_types(dataset=Dataset, val_size=float, batch_size=(int, Sequence))
    @check_interval('val_size', 0, 1)
    def create_dataloader(self, dataset, val_size=0.1, batch_size=10, shuffle=False, random_split=True, **kwargs):
        """
        Method to create training and validation dataloaders.

        :param dataset: instance of Dataset containing X's and y's
        :param float val_size: size to be used for validation dataloader
        :param int batch_size: Number of rows that compose a batch
        :param bool shuffle: Whether each dataloader should provide batches in random order
        :param bool random_split: Whether examples should be splitted randomly between the loaders or if they should
            preserve the order passed.
        :param kwargs: Additional keyword arguments for both dataloaders
        :return: None
        """

        trainset, valset = self._split_data(dataset, val_size, random_split)

        if not isinstance(batch_size, Sequence):
            batch_size = [batch_size, batch_size]

        self.train_loader = DataLoader(dataset=trainset, batch_size=batch_size[0], shuffle=shuffle, **kwargs)
        self.validation_loader = DataLoader(dataset=valset, batch_size=batch_size[1], shuffle=shuffle, **kwargs)

    @check_types(device=str)
    def set_device(self, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    @staticmethod
    def _split_data(data, val_size, random_split=True):
        val_length = int(len(data) * val_size)
        train_length = len(data) - val_length

        if random_split:
            train, val = torch.utils.data.random_split(data, [train_length, val_length])
        else:
            train = Subset(data, range(train_length))
            val = Subset(data, range(train_length, train_length + val_length))

        return train, val

    @staticmethod
    def _format_lrs(lrs, format_='.3g'):
        len_lrs = len(lrs)
        str_format = (len_lrs * f'{{:{format_}}},')[:-1]
        if len_lrs > 1:
            str_format = f'[{str_format}]'
        return str_format.format(*lrs)

    def _set_lr_scheduler(self, lr_scheduler, optimizer, epochs, super_convergence):
        if lr_scheduler is not None:
            if not hasattr(lr_scheduler, 'get_last_lr'):
                raise AttributeError('"lr_scheduler" passed does not implement "get_last_lr" method. Most likely it is '
                                     'an old "closed form" scheduler. "Chained form" must be used.')
            return lr_scheduler
        else:
            lrs = [group['lr'] for group in optimizer.param_groups]
            return OneCycleLR(optimizer, lrs, steps_per_epoch=len(self.train_loader), epochs=epochs, div_factor=25) \
                if super_convergence else DummyLR(optimizer, lrs)

    @staticmethod
    def _init_weights(mod, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        if hasattr(mod, 'weight') & (not isinstance(mod, PReLU)):
            nn.init.kaiming_uniform_(mod.weight, a=a, mode=mode, nonlinearity=nonlinearity)

    def _clip_params(self):
        if self.clip_grad:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.clip_value)

    def _eval_batch(self, inputs, outputs, criterion):
        if self.training:
            self.eval()

        inputs = send_to_device(inputs, self.device)
        targets = send_to_device(outputs, self.device)
        output = self.to(self.device)(*inputs)
        loss = criterion(output, targets)

        return loss.item()

    def _eval_epoch(self, criterion):
        self.eval()
        losses = list()

        with torch.no_grad():
            for idx, (*inputs, targets) in enumerate(self.validation_loader):
                loss = self._eval_batch(inputs, targets, criterion)
                losses.append(loss)

        loss = np.mean(losses)
        print(f'  Validation loss: {loss:.4g}')
        return loss

    @staticmethod
    def _train_epoch_initialization():
        ...

    def _train_batch(self, inputs, outputs, criterion, optimizer):
        if not self.training:
            self.train()

        # compute
        optimizer.zero_grad()   # zeros gradients to prepare for next iteration
        inputs = send_to_device(inputs, self.device)
        targets = send_to_device(outputs, self.device)
        output = self.to(self.device)(*inputs)
        loss = criterion(output, targets)

        # update criterion, optimizer
        loss.backward()         # computes gradients
        self._clip_params()     # if needed/requested clips the gradients
        optimizer.step()        # updates optimizer

        return loss.item()

    def _train_epoch(self, criterion, optimizer, prefix=''):
        self.train()
        self._train_epoch_initialization()

        losses = list()
        for idx, (*inputs, targets) in enumerate(self.train_loader):
            loss = self._train_batch(inputs, targets, criterion, optimizer)
            losses.append(loss)

            # ====================metrics====================
            # todo add estimated time for epoch and total

            # ===================log========================
            suffix = f'| Training loss: {np.mean(losses):.4g}'
            print_progress_bar(idx, len(self.train_loader), prefix=prefix, suffix=suffix, verbose=1)

        return np.mean(losses)

    # todo add tqdm (when fixed for ides)
    def fit(self, epochs, criterion, optimizer, lr_scheduler=None, super_convergence=False, continue_training=False,
            save_checkpoint=False):
        """
        Fit method of the engine

        :param int epochs: number of epochs to train for
        :param criterion: loss function to apply. Needs to implement "item" method to return the loss and a "backward"
            method to compute and apply the gradients
        :param optimizer: optimizer to use. Needs to implement "step" method to determine next step parameters and
            "zero_grad" to clear the gradients used on the step method
        :param lr_scheduler: scheduler to control learning rate evolution. Needs to implement a "step" method to
            determine next learning rate to be used. If None is passed, one of the standard schedulers will be used
        :param bool super_convergence: If no lr_scheduler is passed whether to use a constant learning rate or a
            OneCycleLR policy.
        :param bool continue_training: Whether it should initialize weights of the network (Xavier Uniform) or keep the
            current weights. To continue a previous training it is needed to load the weights previously. Same for
            different initializations.
        :param bool save_checkpoint: Whether to save the model state when the current validation loss is the best.
        # todo check if one can implement checkpoints at periodic times (changing name of file to avoid overwriting)
        :return:
        """

        # todo before starting to fit check if train_loader and validation_loader are set

        lr_scheduler = self._set_lr_scheduler(lr_scheduler, optimizer, epochs, super_convergence)

        # If flag continue_training, model will use the weights in place to start from
        if not continue_training:
            self.apply(self._init_weights)

        try:
            for epoch in range(epochs):
                prefix = f'Epoch {epoch + 1}/{epochs} (lr={self._format_lrs(lr_scheduler.get_last_lr())}):'
                loss = self._train_epoch(criterion, optimizer, prefix)
                self.loss.append('train', loss)

                loss = self._eval_epoch(criterion)
                self.loss.append('validation', loss)

                lr_scheduler.step()     # updates scheduler

                # ===================save=======================
                if save_checkpoint & (loss < min(self.loss.history['validation'], default=1e20)):
                    self.save_state(optimizer=optimizer, epoch=epoch, loss=loss)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    def save_state(self, optimizer=None, epoch=None, loss=None, extra_params=None, name='model'):
        """
        Save current model state
        # todo review and complete. Add scheduler state?

        :param optimizer:
        :param epoch:
        :param loss:
        :param extra_params:
        :param name:
        :return:
        """

        path = self.save_path + '/' + f'{name}.pth'
        params = dict()

        if epoch is not None:
            params['epoch'] = epoch
        if optimizer is not None:
            params['optimizer_state_dict'] = optimizer.state_dict()
        if loss is not None:
            params['best_loss'] = loss
        if params is not None:
            params['params'] = extra_params

        params['dataset_indices'] = {'train_indices': self.train_loader.dataset.indices,
                                     'test_indices':  self.validation_loader.dataset.indices}
        params['losses'] = self.loss
        params['model_state_dict'] = self.state_dict()
        params['model_structure'] = self.__repr__()

        torch.save(params, path)
        print(f'Saved to {path}.')

    @check_types(params=dict)
    def load_state(self, params):
        """
        Load model state from dict

        :param dict params: dict with data to load into model
        :return:
        """

        self.loss = params['losses']
        self.load_state_dict(params['model_state_dict'])


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    # todo adapt to model save. Save optim and scheduler states too
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class Losses:
    """
    Class to track and plot losses

    :param str|Sequence|Optional names: loss vectors to initialize. If None they will be initialized at runtime.
    """

    @check_types(names=(str, Sequence, type(None)))
    def __init__(self, names=None):
        names = {} if names is None else names if isinstance(names, Sequence) else [names]
        self.history = defaultdict(list, {name: list() for name in names})

    def append(self, name, value):
        """
        Appends a new loss to the vector defined by "name"

        :param str name: key name of loss being tracked
        :param int|float|np.ndarray value: loss value to be stored
        :return:
        """
        self.history[name].append(value)

    def get_smoothed_loss(self, name, smoothing_period=5):
        """
        Get a smoothed (moving) average over the "name" loss over the "smoothing_period"

        :param str name: loss in history to smooth
        :param smoothing_period:
        :return:
        """
        if name not in self.history:
            raise KeyError(f'No loss values registered in {name}.')
        return np.asarray(moving_average(self.history[name], smoothing_period))

    @property
    def _has_data(self):
        for name, loss in self.history.items():
            if len(loss) > 0:
                return True
        else:
            return False

    @check_types(smoothing_period=(type(None), int), lrs=(type(None), Sequence), ymax=(type(None), float, int),
                 ymin=(type(None), float, int), auto_zoom=(float, int))
    def plt_loss(self, smoothing_period=5, show=True, lrs=None, auto_zoom=0.99, ymax=None, ymin=None,
                 save_path='.', save=False, formats=('png',)):
        """
        Plot losses in history and if provided learning rates.

        :param int|Optional smoothing_period: smoothing period to use. If None passed it will not plot smoothed losses.
            Default: 5
        :param show: Whether to show the plot on console or not
        :param Sequence|Optional lrs: Iterable containing the learning rates used to generate the losses. Default: None
        :param int|float auto_zoom: value between 0 and 1 to control how to show the plot. This value represents a
            percentage of values smaller values to keep in the plot (thus removing higher peaks). If ymax or ymin are
            passed will not have any effect: Default: 0.99
        :param int|float|Optional ymax: maximum value to have on plot y scale. Default: None
        :param int|float|Optional ymin: minimum value to have on plot y scale. Default: None
        :param str save_path: Folder location where to save the plot for. Default: '.'
        :param bool save: Whether to save the plot or not. Default: False
        :param Iterable formats: Formats to save the plot in. Default: ('png',)
        :return:
        """

        if not self._has_data:
            raise ValueError('No data to plot yet. First you need to append some data.')

        # filter out losses without data
        losses = {name: loss for name, loss in self.history.items() if len(loss) > 0}

        # if smoothing
        if (smoothing_period is not None) and (smoothing_period < 0):
            raise ValueError('"smoothing_period" must either be None or a positive integer.')

        # create figure
        if (lrs is None) or (len(lrs) == 0):
            fig, ax = plt.subplots(figsize=(20, 10))
            ax2 = None
        else:
            # if to plot learning rate plot it right away
            # noinspection PyTypeChecker
            fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]}, sharex=True)

        # plot
        for name, color in zip(losses, cycle(mcolors.BASE_COLORS)):
            epochs = tuple(range(1, len(self.history[name]) + 1))
            ax.plot(epochs, np.asarray(losses[name]), c=color, ls='--', label=f'{name} loss', alpha=0.3)

            if smoothing_period:
                ax.plot(epochs, self.get_smoothed_loss(name, smoothing_period), c=color, ls='-', alpha=0.8,
                        label=f'Smoothed ({smoothing_period}) {name} loss')

        # define axes limits (if there is a big spike)
        self._set_ax_limits(ax, auto_zoom, ymin, ymax)

        # if learning_rate is passed
        lines_lr, labels_lr = self._add_plot_lr(ax2, lrs)

        # set legend and title
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines + lines_lr, labels + labels_lr, loc=0, frameon=False)
        fig.suptitle('Loss evolution over epochs')
        ax.set_ylabel('loss')
        try:
            ax2.set_xlabel('epochs')
        except AttributeError:
            ax.set_xlabel('epochs')

        # Show in console
        if show:
            fig.show()

        # save
        if save:
            for f in formats:
                savefig(fig, 'loss', save_path, fileformat=f)

    def _set_ax_limits(self, ax, auto_zoom=0.0, ymin=None, ymax=None):
        if (ymin is not None) or (ymax is not None):
            ax.set_ylim(ymin, ymax)
            return

        if auto_zoom != 0:
            if 0 < auto_zoom <= 1:
                values = np.asarray(list(chain.from_iterable(self.history.values())))
                ax.set_ylim(*self._get_zoom_limits(values, auto_zoom))
            else:
                warnings.warn('"auto_zoom" must be between 0 and 1. Will not apply')

    @staticmethod
    def _get_zoom_limits(arr, zoom):
        k = int(np.round(len(arr)*zoom))
        res = np.argpartition(np.abs(arr), k)
        include = np.asarray(arr)[res[:k]]

        ymin = include.min()
        ymax = include.max()
        ymin -= (10**(get_magnitude(ymin)-1)) * 5
        ymax += 10**(get_magnitude(ymax)-1) * 5
        return ymin, ymax

    @staticmethod
    def _add_plot_lr(ax, lrs):
        if lrs is None:
            return [], []

        elif len(lrs) == 0:
            warnings.warn('\nLength of learning rates list is 0. Learning rate will not be plotted', stacklevel=2)
            return [], []

        # add subplot with sharex
        epochs = range(1, len(lrs) + 1)
        ax.plot(epochs, lrs, 'y-', label='learning rates')
        ax.set_ylabel('learning rate')
        return ax.get_legend_handles_labels()


class LRFinder:
    """
    Class that implements a learning rate finder.
    """

    def __init__(self):
        self._diverge_flag = False
        self._history = None
        self._best_loss = None
        self._lr_scheduler = None

    @check_options(step_mode=('exp', 'linear'))
    @check_types(n_steps=(type(None), int), init_lr=float, final_lr=float)
    def run(self, net, optimizer, criterion, n_steps=None, step_mode='exp', init_lr=1e-8, end_lr=1e1,
            smooth_f=0.05, diverge_th=5.0):
        """

        :param net: network to be trained.
        :param optimizer: optimizer to use on the test. Should be the same intended to use on train
        :param criterion: criterion to use on the test. Should be the same intended to use on train
        :param n_steps: number of steps to run test for. If n_steps > number of train batches, batches will be cycled
            and reused. If n_steps = None, n_steps = number of train batches
        :param step_mode: "exp" or "linear", which way should the lr be increased from optimizer's initial lr to
            `end_lr`. Default, "exp".
        :param float init_lr: lower bound for the test.
        :param float end_lr: upper bound for the test
        :param float|Optional smooth_f: loss smoothing factor in range `[0, 1)`. Default, 0.05
        :param float|Optional diverge_th: Used for stopping the search when `current loss > diverge_th * best_loss`.
            Default, 5.0.
        :return: self instance for chaining
        """

        if not hasattr(net, '_train_batch'):
            raise NotImplementedError('Passed Network does not have a "_train_batch" method implemented. '
                                      'This method is required for the LRFinder to work. Method signature must be '
                                      '(inputs, outputs, criterion, optimizer, lr_scheduler) -> loss.item().')

        self._diverge_flag = False
        self._best_loss = None
        self._history = {'lr': [], 'loss': []}

        net = deepcopy(net)
        optimizer = optimizer(net.parameters())
        criterion = deepcopy(criterion)

        # define n_steps
        if n_steps is None:
            n_steps = len(net.train_loader)
        else:
            if n_steps > len(net.train_loader):
                warnings.warn('Number of steps requested is higher than the number of batches in train loader.'
                              'Train loader will be cycled up to the number of steps requested. Results might be '
                              'suboptimal.')

        # set_lr_scheduler
        self._set_lr_scheduler(optimizer, n_steps, step_mode, init_lr, end_lr)

        # Initialize net weights
        net.apply(net._init_weights)
        for i, (inputs, outputs) in enumerate(cycle(net.train_loader)):
            self._history['lr'].append(self._lr_scheduler.get_lr())
            loss = net._train_batch(inputs, outputs, criterion, optimizer, self._lr_scheduler)

            if i == 0:
                self._best_loss = loss
            else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * self._history['loss'][-1]
                if loss < self._best_loss:
                    self._best_loss = loss
            self._history['loss'].append(loss)

            # Check if the loss has diverged; if it has, stop the trainer
            if self._history["loss"][-1] > diverge_th * self._best_loss:
                self._diverge_flag = True
                print("Stopping early, the loss has diverged.")
                break

            # display progress
            print_progress_bar(i, n_steps, verbose=1)

            if i + 1 == n_steps:
                break

        return self

    def plot(self, skip_start=5, skip_end=3, log_lr=True, save_path='.', save=False):
        """Plots the learning rate range test.

        :param int|Optional skip_start: number of batches to trim from the start. Default: 5.
        :param int|Optional skip_end: number of batches to trim from the end. Default: 3.
        :param bool|Optional log_lr: True to plot the learning rate in a logarithmic scale; otherwise, plotted in
            a linear scale. Default: True.
        :param string save_path: Folder path to save to.
        :param bool save: Whether to save the plot or not
        """

        if self._history is None:
            raise RuntimeError("learning rate finder didn't run yet so results can't be plotted")

        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected

        lrs = self._history["lr"]
        losses = self._history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Plot loss as a function of the learning rate
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(lrs, losses)
        if log_lr:
            ax.set_xscale('log')
        ax.set_xlabel('Learning Rates')
        ax.set_ylabel('Losses')
        fig.suptitle('Loss evolution over learning rates')
        plt.show()

        if save:
            fig.savefig(save_path + '/' + 'learning_rate.jpg', format='png', dpi=600)

    def get_lr_suggestion(self):
        """
        Returns: learning rate at the minimum numerical gradient
        """

        if self._history is None:
            raise RuntimeError("learning rate finder didn't run yet so lr_suggestion can't be returned")
        loss = torch.tensor(self._history["loss"])
        grads = loss[1:] - loss[:-1]

        # restrict the search around minimum of loss
        lim = max(3, round(0.05 * len(loss)))
        lim_min, lim_max = loss.argmin() - lim, loss.argmin() + lim
        min_grad_idx = lim_min + grads[lim_min:lim_max].argmin() + 1
        return self._history["lr"][int(min_grad_idx)]

    def get_results(self):
        """
        Returns: dictionary with loss and lr logs fromm the previous run
        """
        return self._history

    def _set_lr_scheduler(self, optimizer, n_steps, step_mode, init_lr, end_lr):
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = init_lr

        if step_mode == 'exp':
            gamma = (end_lr/init_lr)**(1/(n_steps-1))
            self._lr_scheduler = ExponentialLR(optimizer, gamma)

        elif step_mode == 'linear':
            self._lr_scheduler = LinearLR(optimizer, end_lr, n_steps)

        else:
            raise NotImplementedError('"step_mode" passed does not have a LRScheduler implemented in LRFinder.')


def send_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)

    elif isinstance(x, (list, tuple)):
        return [send_to_device(d, device) for d in x]

    else:
        raise RuntimeError(f'Could not send data to {device}.\n {x}.')


def check_data_types(data) -> Union[list, torch.Tensor]:
    """
    Utility function to check if data passed is all on tensor format. Can handle iterables or single structures

    :param data: Data to be checked
    :return: list of tensors
    """

    if torch.is_tensor(data):
        return data

    else:
        try:
            return torch.tensor(data, dtype=torch.float)
        except ValueError:
            return [torch.tensor(x, dtype=torch.float) for x in data]
