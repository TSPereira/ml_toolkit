from inspect import getfullargspec
from collections import namedtuple
from typing import Sequence, Callable
from numpy import ndarray

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from .engine import Engine, check_data_types
from .utils import check_non_linearity
from ....utils.os_utl import check_types, check_interval, NoneType


class MLP(Engine):
    """Class to implement a Multi Layer Perceptron.
    It is based on Engine and has flexibility to create any number of layers

    Arguments:
        input_size: number of features

        output_size: number of output neurons

        output_non_linearity: Non_linearity to be applied on the final output. It either must be a nn.Module or None

        output_bias: Whether to include a bias neuron for the final layer or not.

        hidden_layer_config: Configuration of all hidden layers to be created. It should be a sequence of layers, each
            containing: (size, bias, dropout, non_linearity).

            "size" is the number of neurons of the layer (not including bias). Should be a positive integer
            "bias" is a boolean to indicate whether a bias neuron should be considered for the layer
            "dropout" is a float between 0 and 1 to control the ratio of dropout to apply to the neurons of this layer.
                If 0, the dropout layer is not created.
            "non_linearity" is the non_linearity torch module to be applied to the layer outputs.
                Pass None if no non-linearity should be applied

        save_path: Path to save any outputs of the Engine module

        kwargs: Additional key word arguments to pass to Engine
    """

    def __init__(self, input_size, output_size=1, output_non_linearity=None, output_bias=False,
                 hidden_layer_config=None, save_path='.', **kwargs):

        # perform sanity checks
        hidden_layer_config = hidden_layer_config or []
        inputs = input_size, output_size, output_non_linearity, output_bias, hidden_layer_config
        self._sanity_checks_init_args(**dict(zip(getfullargspec(self.__class__).args[1:], inputs)))

        # Create model
        super(MLP, self).__init__(save_path=save_path, **kwargs)
        self.model = nn.Sequential(*self._construct_layers(*inputs))

    @staticmethod
    @check_types(input_size=int, output_size=int, output_non_linearity=(Callable, NoneType), output_bias=bool,
                 hidden_layer_config=(Sequence, NoneType))
    @check_interval(('input_size', 'output_size'), 0)
    def _sanity_checks_init_args(**kwargs):
        for name, value in kwargs.items():
            # check if non_linearity is a nn.Module
            if name == 'output_non_linearity':
                assert check_non_linearity(value), f'{name} must be a torch nn.Module or None.'

            # check each hidden layer content
            if (name == 'hidden_layer_config') and value:
                for i, layer in enumerate(value):
                    check_layer(layer, i)

        return

    def _construct_layers(self, input_size, output_size, output_non_linearity, output_bias, hidden_layers):
        # Generate full configuration of network
        LayerOptions = namedtuple('layer_options', ['size', 'bias', 'dropout', 'non_linearity'])
        config = [(input_size, False, False, False),
                  *hidden_layers,
                  (output_size, output_bias, False, output_non_linearity)]
        config = [LayerOptions(*layer) for layer in config]

        # Create each layer in config
        layers = []
        for i, layer_options in enumerate(config[1:], 1):
            # Add linear module
            setattr(self, f'fc_{i}', nn.Linear(int(config[i-1].size), int(layer_options.size), layer_options.bias))
            layers.append(getattr(self, f'fc_{i}'))

            # Add dropout
            if layer_options.dropout > 0:
                setattr(self, f'dropout_{i}', nn.Dropout(layer_options.dropout))
                layers.append(getattr(self, f'dropout_{i}'))

            # Add non_linearity
            if layer_options.non_linearity:
                setattr(self, f'actfn_{i}', layer_options.non_linearity())
                layers.append(getattr(self, f'actfn_{i}'))

        return layers

    # noinspection PyMethodOverriding
    @check_types(X=(torch.Tensor, Sequence, ndarray), y=(torch.Tensor, Sequence, ndarray))
    def create_dataloader(self, X, y, val_size=0.1, batch_size=10, shuffle=False, random_split=True, **kwargs):
        """
        Method to create training and validation dataloaders.

        :param X: Tensor or Sequence with examples
        :param y: Tensor or Sequence with labels
        :param float val_size: size to be used for validation dataloader
        :param int batch_size: Number of rows that compose a batch
        :param bool shuffle: Whether each dataloader should provide batches in random order
        :param bool random_split: Whether examples should be splitted randomly between the loaders or if they should
            preserve the order passed.
        :param kwargs: Additional keyword arguments for both dataloaders
        :return: None
        """
        dataset = TensorDataset(*map(check_data_types, [X, y]))
        super().create_dataloader(dataset, val_size, batch_size, shuffle, random_split, **kwargs)

    def forward(self, x):
        """Applies a forward pass on the Neural Network

        :param x: Tensor of n_examples x m_features
        :type x: torch.Tensor
        :return: Predictions for the examples passed
        :rtype: torch.Tensor
        """
        return self.model(x)


def check_layer(layer, idx):
    prefix = f'[Layer {idx}]'

    # Check number of arguments in layer
    assert len(layer) == 4, f'{prefix} Number of arguments should be 4. {len(layer)} were passed.'

    # Check size
    assert isinstance(layer[0], int), f'{prefix} First argument (size) must be integer.'
    assert layer[0] > 0, f'{prefix} First argument (size) must be > 0.'

    # Check bias
    assert isinstance(layer[1], bool), f'{prefix} Second argument (bias) must be boolean.'

    # Check dropout
    assert isinstance(layer[2], (int, float)), f'{prefix} Third argument (dropout) must be a float or int.'
    assert 0 <= layer[2] <= 1, f'{prefix} Third argument (dropout) must be between 0 and 1.'

    # Check non_linearity
    assert check_non_linearity(layer[3]), f'{prefix} Fourth argument (non_linearity) must be a torch nn.Module or None.'
    return
