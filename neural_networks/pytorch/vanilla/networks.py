import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import Dropout
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.nn.modules.activation import PReLU
from pandas.core.frame import DataFrame, Series
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

from _closer_packages.log_utils import print_progress_bar
from ....statistics import moving_average


# todo introduce ignite. implement early stopping, metrics, find_weight_decay, find_momentum
#  test _dynamic_dropout
class Net(nn.Module):
    def __init__(self, save_path='.'):
        super(Net, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = None
        self.test_loader = None
        self.holdout_loader = None
        self.val_loss = []
        self.train_loss = []
        self.lrs = None
        self.metrics = dict()

        self.save_path = save_path


    def create_dataloader(self, X, y=None, test_size=0.2, batch_size=1, shuffle=True, dtype=torch.float32,
                          num_workers=0):
        # todo reevaluate function. Can be made simpler
        assert isinstance(dtype, (torch.dtype, tuple))
        if type(dtype) == tuple:
            assert len(dtype) == 2
            assert all(isinstance(i, torch.dtype) for i in dtype)
        else:
            dtype = (dtype, dtype)

        if isinstance(X, DataFrame):
            X = X.values

        if y is None:
            y = X
        elif isinstance(y, (DataFrame, Series)):
            y = y.values

        assert X.shape[0] == y.shape[0]

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        data = torch.utils.data.TensorDataset(torch.tensor(X, dtype=dtype[0]), torch.tensor(y, dtype=dtype[1]))

        test_length = int(data.tensors[0].shape[0] * test_size)
        train_length = data.tensors[0].shape[0] - test_length
        data_train, data_test = torch.utils.data.random_split(data, [train_length, test_length])

        self.train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=shuffle,
                                                        num_workers=num_workers)
        self.test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=shuffle,
                                                       num_workers=num_workers)


    def fit(self, epochs, criterion, optimizer, super_convergence=True, save_checkpoint=False):
        self.apply(self._initialize_weights)

        # todo implement metrics module
        if super_convergence:
            lr_max = optimizer.param_groups[0]['lr']
            lr_min = lr_max / 10
            lr_final = lr_min / 100

            # 15% of epochs to save for final reduction of learning rate
            _epochs = int(0.85 * epochs)
            self.lrs = list(np.linspace(lr_min, lr_max, int(_epochs / 2))) + \
                       list(np.linspace(lr_max, lr_min, _epochs - int(_epochs / 2))) + \
                       list(np.linspace(lr_min, lr_final, epochs - _epochs))

        for epoch in range(epochs):
            self.train()
            prefix = f'Epoch {epoch + 1}/{epochs}:'
            loss_train = list()

            if super_convergence:
                optimizer.param_groups[0]['lr'] = self.lrs[epoch]
                prefix = prefix[:-1] + f' (lr={self.lrs[epoch]:#.3g}):'

            for idx, (inputs, targets) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # ===================for lstm====================
                if callable(getattr(self, 'init_hidden', None)):
                    self.init_hidden(inputs.size(0))

                # ===================forward=====================
                output = self.to(self.device)(inputs)
                loss = criterion(output, targets)
                # ===================backward====================
                loss.backward()

                if callable(getattr(self, 'init_hidden', False)) & (getattr(self, 'clip_grad', False)):
                    if self.clip_grad:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_value)

                optimizer.step()
                optimizer.zero_grad()

                # ====================metrics====================
                loss_train.append(loss.item())
                # todo add estimated time for epoch and total

                # ===================log========================
                suffix = f'| Training loss: {np.mean(loss_train):.4g}'
                print_progress_bar(idx + 1, len(self.train_loader), prefix=prefix, suffix=suffix, verbose=1)

            self.train_loss.append(np.mean(loss_train))

            # ===================Evaluation======================
            self.eval()
            loss_val = list()
            for idx, (inputs, targets) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # ===================for lstm====================
                if callable(getattr(self, 'init_hidden', None)):
                    self.init_hidden(inputs.size(0))

                output = self.to(self.device)(inputs)
                loss_val.append(criterion(output, targets).item())

            self.val_loss.append(np.mean(loss_val))
            # ===================log========================
            suffix = ''
            print(f'  Validation loss: {self.val_loss[-1]:.4g}' + suffix)

            # ===================save=======================
            if save_checkpoint & (self.val_loss[-1] < min(self.val_loss[:-1], default=1e20)):
                self.save_state(optimizer=optimizer, epoch=epoch, loss=self.val_loss[-1])


    def find_lr(self, optimizer, criterion, final_value=1e6, init_value=1e-12, beta=0.98, save=True):
        print('Evaluating best learning rates...')
        # todo implement prints and return lr and several betas
        # beta is the value for smooth losses
        self.apply(self._initialize_weights)
        self.train()

        num = len(self.train_loader) - 1  # total number of batches
        mult = (final_value / init_value) ** (1 / num)

        losses = []
        lrs = []
        best_loss = 0.
        avg_loss = 0.
        lr = init_value

        for batch_num, (inputs, targets) in enumerate(self.train_loader):
            # ===================for lstm====================
            if callable(getattr(self, 'init_hidden', None)):
                self.init_hidden(inputs.size(0))

            optimizer.param_groups[0]['lr'] = lr

            batch_num += 1
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = self.to(self.device)(inputs)
            loss = criterion(outputs, targets)

            # Compute the smoothed loss to create a clean graph
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** batch_num)

            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss

            # append loss and learning rates for plotting
            lrs.append(np.log10(lr))
            losses.append(smoothed_loss)

            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 10 * best_loss:
                break

            # backprop for next step
            loss.backward()
            optimizer.step()

            # update learning rate
            lr = mult * lr

            # display progress
            print_progress_bar(batch_num + 1, len(self.train_loader), verbose=1)

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.set_xlabel('Learning Rates (log10)')
        ax.set_ylabel('Losses')
        fig.suptitle('Loss evolution over learning rates')
        ax.plot(lrs, losses)
        plt.show()

        optimizer.zero_grad()
        self.zero_grad()

        if save:
            fig.savefig(self.save_path + '/' + 'learning_rates.eps', format='eps', dpi=600)
            fig.savefig(self.save_path + '/' + 'learning_rates.jpg', format='jpg', dpi=600)


    # def find_momentum(self, momentums, optimizer, criterion, beta=0.99, save=True):
    #
    #
    #     print('Evaluating best momentum...')
    #     # todo implement prints and return lr and several betas
    #     # beta is the value for smooth losses
    #     self.apply(self._initialize_weights)
    #     self.train()
    #
    #     num = len(self.train_loader) - 1  # total number of batches
    #     mult = (final_value / init_value) ** (1 / num)
    #
    #     losses = []
    #     lrs = []
    #     best_loss = 0.
    #     avg_loss = 0.
    #     lr = init_value
    #
    #     for batch_num, (inputs, targets) in enumerate(self.train_loader):
    #         optimizer.param_groups[0]['lr'] = lr
    #
    #         batch_num += 1
    #         inputs = inputs.to(self.device)
    #         targets = targets.to(self.device)
    #
    #         optimizer.zero_grad()
    #         outputs = self.to(self.device)(inputs)
    #         loss = criterion(outputs, targets)
    #
    #         # Compute the smoothed loss to create a clean graph
    #         avg_loss = beta * avg_loss + (1 - beta) * loss.item()
    #         smoothed_loss = avg_loss / (1 - beta ** batch_num)
    #
    #         # Record the best loss
    #         if smoothed_loss < best_loss or batch_num == 1:
    #             best_loss = smoothed_loss
    #
    #         # append loss and learning rates for plotting
    #         lrs.append(np.log10(lr))
    #         losses.append(smoothed_loss)
    #
    #         # Stop if the loss is exploding
    #         if batch_num > 1 and smoothed_loss > 10 * best_loss:
    #             break
    #
    #         # backprop for next step
    #         loss.backward()
    #         optimizer.step()
    #
    #         # update learning rate
    #         lr = mult * lr
    #
    #     fig, ax = plt.subplots(figsize=(20, 10))
    #     ax.set_xlabel('Learning Rates (log10)')
    #     ax.set_ylabel('Losses')
    #     fig.suptitle('Loss evolution over learning rates')
    #     ax.plot(lrs, losses)
    #     plt.show()
    #
    #     optimizer.zero_grad()
    #     self.zero_grad()
    #
    #     if save:
    #         fig.savefig(self.save_path + '/' + 'learning_rates.eps', format='eps', dpi=600)
    #         fig.savefig(self.save_path + '/' + 'learning_rates.jpg', format='jpg', dpi=600)

    def plot_loss(self, ma_n_periods=5, plot_lr=False, save=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        epochs = tuple(range(1, len(self.train_loss) + 1))

        # Calculate moving averages and plot limits
        ma_train_loss = moving_average(self.train_loss, n_periods=ma_n_periods)
        ma_val_loss = moving_average(self.val_loss, n_periods=ma_n_periods)
        _ylim_max = 10 ** (np.floor(np.log10(np.array(ma_val_loss).mean())) + 1)
        _ylim_min = 10 ** (np.floor(np.log10(np.array(ma_val_loss).min())))

        # plot losses
        ax.plot(epochs, np.array(self.train_loss), 'b-', label='train_loss', alpha=0.3)
        ax.plot(epochs, np.array(ma_train_loss), 'b-', label=f'train_loss (avg {ma_n_periods})', alpha=0.8)
        ax.plot(epochs, np.array(self.val_loss), 'r-', label='validation_loss', alpha=0.3)
        ax.plot(epochs, np.array(ma_val_loss), 'r-', label=f'validation_loss (avg {ma_n_periods})', alpha=0.8)
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')

        # set axes limits
        _cur_lims = ax.get_ylim()
        if _ylim_max < _cur_lims[1]:
            ax.set_ylim(_ylim_min, _ylim_max)

        # ask matplotlib for the plotted objects and their labels
        lines, labels = ax.get_legend_handles_labels()

        # plot learning rate
        if plot_lr:
            ax2 = ax.twinx()
            ax2.plot(epochs, np.array(self.lrs[:epochs.stop - epochs.start]), 'y-', label='learning rates')
            ax2.set_ylabel('learning rate')
            _cur_lims = ax2.get_ylim()
            ax2.set_ylim(_cur_lims[0], _cur_lims[1] * 5)

            _lines, _labels = ax2.get_legend_handles_labels()
            lines += _lines
            labels += _labels

        # add legend
        ax.legend(lines, labels, loc=0, frameon=False)
        fig.suptitle('Loss evolution over epochs')

        plt.show()

        if save:
            fig.savefig(self.save_path + '/' + 'loss.eps', format='eps', dpi=600)
            fig.savefig(self.save_path + '/' + 'loss.jpg', format='jpg', dpi=600)


    def save_state(self, optimizer=None, epoch=None, loss=None, extra_params=None, name='model'):
        print('Saving model...', end=' ')
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
                                     'test_indices':  self.test_loader.dataset.indices}
        params['losses'] = {'training_losses': self.train_loss, 'validation_losses': self.val_loss}
        params['model_state_dict'] = self.state_dict()
        params['model_structure'] = self.__repr__()

        torch.save(params, path)
        print(f'Saved to {path}.')


    def load_state(self, params):
        assert isinstance(params, dict)

        self.train_loss = params['losses']['training_losses']
        self.val_loss = params['losses']['validation_losses']
        self.load_state_dict(params['model_state_dict'])


    @staticmethod
    def _initialize_weights(mod):
        if hasattr(mod, 'weight') & (not isinstance(mod, (PReLU))):
            nn.init.xavier_uniform_(mod.weight)


    def _dynamic_dropout(self, upd=True):
        """
        Under development/testing - Still not included as an option

        :param upd:
        :type upd:
        :return:
        :rtype:
        """
        assert isinstance(upd, (list, bool))

        if upd:
            nr_dropouts = np.sum(True for mod in self.model._modules.values() if isinstance(mod, Dropout))
            _upd = np.random.choice(a=[False, True], size=nr_dropouts)

            if isinstance(upd, list):
                assert len(upd) == nr_dropouts
                _upd = [random if choice else False for random, choice in zip(_upd, upd)]

            n = 0
            for nr, module in self.model._modules.items():
                if isinstance(module, Dropout):
                    if _upd[n]:
                        module.p = module.p * 1.1

                    print(module)
                    n += 1


    def _eval_metrics(self, pred, target, metrics):
        """
                Under development/testing - Still not included as an option
        """

        pred = pred.detach().numpy().reshape(-1, )
        target = target.detach().numpy().reshape(-1, )
        # todo implement different metrics
        #  output should be a string to add to the suffix
        #  should also update an attribute with a dictionary of type "metric_name": array(metric_value)
        #  check which metrics are available according to the loss criterion
        #  create separate class and initialize it in __init__ instead of a method
        suffix = ""

        if 'mae' in metrics:
            err = (pred - target)

            mae = err.mean()
            mae_over = err[err > 0].mean()
            mae_under = err[err < 0].mean()

            suffix += f' | mae: {mae:.2f} ({mae_over:.2f}/{mae_under:.2f})'
            self.metrics['mae'].append({'mae_mean': mae, 'mae_over': mae_over, 'mae_under': mae_under})

        if 'mre' in metrics:
            err = (pred - target) * 100 / target

            mre = err.mean()
            mre_over = err[err > 0].mean()
            mre_under = err[err < 0].mean()

            suffix += f' | mre: {mre:.2f}% ({mre_over:.2f}%/{mre_under:.2f}%)'
            self.metrics['mre'].append({'mre_mean': mre, 'mre_over': mre_over, 'mre_under': mre_under})

        return suffix


class LSTM(Net):
    """

    """


    def __init__(self, n_features, seq_len, hidden_layer_size=100, output_size=1, n_layers=1, dropout=0.,
                 clip_grad=True, clip_value=10, save_path='.'):
        assert isinstance(n_features, int)
        assert n_features > 0
        super(LSTM, self).__init__(save_path=save_path)

        self.n_features = n_features
        self.n_hidden = hidden_layer_size
        self.n_layers = n_layers
        self.seq_len = int(seq_len)
        self.output_size = output_size
        self.clip_grad = clip_grad
        self.clip_value = clip_value
        dropout_lstm = dropout if self.n_layers > 1 else 0

        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size=self.n_features, hidden_size=self.n_hidden,
                            num_layers=self.n_layers, batch_first=True, dropout=dropout_lstm)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.n_hidden * self.seq_len, self.output_size)


    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        self.hidden = (hidden_state, cell_state)


    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        predictions = self.linear(self.dropout(lstm_out.contiguous().view(batch_size, -1)))
        return predictions.view(-1)


    def create_dataloader(self, X, seq_creator=None, val_size=0.1, batch_size=10, shuffle=False, **kwargs):
        train_seq = X if seq_creator is None else seq_creator(X, tw=self.seq_len)
        dataset = TensorDataset(*train_seq)

        val_length = int(len(dataset) * val_size)
        train_length = len(dataset) - val_length

        trainset = Subset(dataset, range(train_length))
        valset = Subset(dataset, range(train_length, train_length + val_length))

        self.train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle)
        self.test_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=shuffle)


    def save_state(self, optimizer=None, epoch=None, loss=None, extra_params=None, name='model'):
        extra_params = dict(n_features=self.n_features, n_hidden=self.n_hidden, seq_len=self.seq_len,
                            output_size=self.output_size, n_layers=self.n_layers, clip_grad=self.clip_grad,
                            clip_value=self.clip_value)
        super().save_state(optimizer, epoch, loss, extra_params, name)
