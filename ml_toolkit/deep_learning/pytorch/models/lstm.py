from operator import itemgetter
from typing import Callable
import warnings

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, Dataset

from .engine import Engine, check_data_types
from ....utils.os_utl import check_types, check_interval
from ....utils.log_utl import wrap_text


# lstm args:
#     input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, non_linearity
#
# Additional arguments:
#     seq_len
#     output_size: if different from seq_len implies a linear layer after the lstm # same as return_sequences?


class LSTM(Engine):
    def __init__(self, input_size, seq_max_len, hidden_layer_size=100, output_size=1, n_layers=1, dropout=0.,
                 bias=True, bidirectional=False, batch_first=True, return_sequences=False, pack_sequences=True,
                 enforce_sorted=True, save_path='.', **kwargs):

        # self._sanity_checks_init_args(**dict(zip(getfullargspec(self.__class__).args[1:], inputs)))
        self._sanity_checks_init_args(input_size=input_size, seq_max_len=seq_max_len,
                                      hidden_layer_size=hidden_layer_size,
                                      output_size=output_size, n_layers=n_layers, dropout=dropout)

        super(LSTM, self).__init__(save_path=save_path, **kwargs)

        self.input_size = input_size
        self.seq_max_len = seq_max_len
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self._dropout = dropout
        self._bias = bias
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.pack_sequences = pack_sequences

        self.init_hidden_every = 'batch'  # epoch
        self.detach_history_every_batch = True
        self.enforce_sorted = enforce_sorted

        self.hidden = None
        dropout_lstm = self._dropout if self.n_layers > 1 else 0

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_layer_size, num_layers=self.n_layers,
                            bias=bias, batch_first=batch_first, dropout=dropout_lstm, bidirectional=self.bidirectional)

        # Create linear layer if self.return_sequences = False
        if not self.return_sequences:
            self.dropout = nn.Dropout(self._dropout)
            self.linear = nn.Linear(self.hidden_layer_size * self.seq_max_len, self.output_size, bias=self._bias)

    @staticmethod
    @check_types(input_size=int, seq_max_len=int, output_size=int, hidden_layer_size=int, n_layers=int, dropout=float)
    @check_interval(('input_size', 'seq_max_len', 'hidden_layer_size', 'output_size', 'n_layers'), 0)
    @check_interval('dropout', 0, 1, 'min')
    def _sanity_checks_init_args(**kwargs):
        # At the moment all of the checks are made in the decorators, but additional ones can be added in the function
        ...

    @staticmethod
    def _sanity_checks_dataloader_args(**kwargs):
        ...

    # todo redo
    @check_types(batch_size=int, shuffle=bool, random_split=bool)
    def create_dataloader(self, X, y, val_size=0.1, batch_size=10, shuffle=False, random_split=True, **kwargs):
        """
        Method to create training and validation dataloaders. When sequences need to be created it is possible to pass
        the callable to create them and it will be applied using attribute "seq_len" that has to be implemented in
        superseeding class.

        :param data: Tensor or Iterable of Tensors to create dataloader. If seq_creator is a callable, data can be of
        any type the callable accepts. If "collate_fn" is not user_defined, then it must be an Iterable of Tensors of
        format [X, y]
        :param float val_size: size to be used for validation dataloader
        :param int batch_size: Number of rows that compose a batch
        :param bool shuffle: Whether each dataloader should provide batches in random order
        :param bool random_split: Whether examples should be splitted randomly between the loaders or if they should
        preserve the order passed.
        :param kwargs: Additional keyword arguments for both dataloaders
        :return: None
        """


        # todo check inputs/outputs dimensions
        # self._sanity_checks_dataloader_args()

        # Create padder instance if not custom
        if not kwargs.get('collate_fn', False):
            kwargs['collate_fn'] = PadSequence(self.seq_max_len, self.return_sequences)

        # # If hidden unit should be initialized at every batch might be needed to drop last batch of the dataloaders
        # if (self.init_hidden_every == 'epoch') and (not kwargs.get('drop_last', False)):
        #     warnings.warn(wrap_text('With "init_hidden_every" set to "batch" it is necessary to drop the last batch '
        #                             'of the dataloaders. "drop_last" in the dataloaders will be set to true.', 160),
        #                   UserWarning, stacklevel=2)
        #     kwargs['drop_last'] = True

        # Create VariableLengthTensorDataset
        data = VariableLengthTensorDataset(*map(check_data_types, [X, y]))

        # noinspection PyCallingNonCallable
        super(LSTM, self).create_dataloader(data, val_size, batch_size, shuffle, random_split, **kwargs)

    def init_hidden(self, batch_size):
        # todo add another option (check initialization techniques)
        size = self.n_layers * (1 + self.bidirectional), batch_size, self.hidden_layer_size
        hidden_state = torch.zeros(*size, device=self.device)
        cell_state = torch.zeros(*size, device=self.device)
        self.hidden = (hidden_state, cell_state)

    def _train_epoch_initialization(self):
        if self.init_hidden_every == 'epoch':
            self.init_hidden(self.train_loader.batch_size)

    def _batch_initialization(self, batch):
        if self.training:
            if self.init_hidden_every == 'batch':
                self.init_hidden(self._get_batch_size(batch))

            elif self.detach_history_every_batch:
                self.hidden = repackage_hidden(self.hidden)

        else:
            self.init_hidden(self._get_batch_size(batch))

    def _get_batch_size(self, batch) -> int:
        idx = 0 if self.batch_first else 1
        return batch.size(idx) if torch.is_tensor(batch) else batch[0].size(idx)

    def forward(self, x):
        self._batch_initialization(x)

        # todo if not self.training, make sure input is of correct length and/or pad accordingly

        # Pad and pack sequences if needed.
        # Apply lstm and get output sequences
        # Unpack sequences if needed
        with PackUnpackPaddedSequence(x, self.pack_sequences, self.enforce_sorted) as pups:
            pups.lstm_out, self.hidden = self.lstm(pups.sequences, self.hidden)

        if self.return_sequences:
            return pups.lstm_out

        else:
            predictions = self.linear(self.dropout(pups.lstm_out.contiguous().view(self._get_batch_size(x), -1)))
            return predictions.view(-1)

    def save_state(self, optimizer=None, epoch=None, loss=None, extra_params=None, name='model'):
        extra_params = dict(input_size=self.input_size, hidden_layer_size=self.hidden_layer_size, seq_len=self.seq_len,
                            output_size=self.output_size, n_layers=self.n_layers, clip_grad=self.clip_grad,
                            clip_value=self.clip_value)
        super().save_state(optimizer, epoch, loss, extra_params, name)

    def load_state(self, params):
        super().load_state(params)
        self.lstm.flatten_parameters()


class PackUnpackPaddedSequence:
    """ only done for batch_first == True"""

    def __init__(self, sequences, pack_sequences=True, enforce_sorted=False):
        self._sequences = sequences
        self.sequences = sequences
        self.seq_max_len = sequences.size(1)
        self.lstm_output = None
        self.pack_sequences = pack_sequences
        self.enforce_sorted = enforce_sorted
        self.sorted_idx = None

    def sort_sequences(self, batch):
        _sorted = zip(*sorted(zip(batch, batch.seq_len, torch.arange(batch.size(0))), key=itemgetter(1), reverse=True))
        batch, batch.seq_len, self.sorted_idx = map(torch.stack, _sorted)
        return batch

    def unsort(self, batch):
        _unsorted, idx = zip(*sorted(zip(batch, self.sorted_idx), key=itemgetter(1)))
        batch = torch.stack(_unsorted)

        if isinstance(self._sequences, VariableLengthTensor):
            batch = VariableLengthTensor(batch, sequences_lengths=self._sequences.seq_len)

        return batch

    def __enter__(self):
        if self.enforce_sorted:
            # noinspection PyTypeChecker
            self.sequences = self.sort_sequences(self.sequences)

        if self.pack_sequences:
            self.sequences = pack_padded_sequence(self.sequences, self.sequences.seq_len, True, self.enforce_sorted)

        return self

    def __exit__(self, *exc):
        if self.lstm_output is None:
            raise AttributeError('"lstm_output" is not set. Assign this attribute within the context managed block.')

        if self.pack_sequences:
            # noinspection PyTypeChecker
            self.lstm_output, _ = pad_packed_sequence(self.lstm_output, True, 0, self.seq_max_len)

        if self.enforce_sorted:
            self.lstm_output = self.unsort(self.lstm_output)


class PadSequence:
    """
    Only works for batch_first
    """

    def __init__(self, seq_max_len, return_sequences=False):
        self.seq_max_len = seq_max_len
        self.return_sequences = return_sequences

    def __call__(self, batch):
        # ==================================== Prepare X =====================================================
        # Get each sequence X and trim it if bigger than seq_max_len
        x = list(map(lambda x_: itemgetter(0)(x_)[-self.seq_max_len:], batch))

        # Pad smaller sequences
        x_padded = pad_sequence(x, batch_first=True).type(torch.float)

        # adjust the sequence lengths to the seq_max_len passed.
        if self.seq_max_len > x_padded.size(1):
            add = torch.zeros((x_padded.size(0), self.seq_max_len - x_padded.size(1)))
            x_padded = torch.cat((x_padded, add), dim=1)

        # ==================================== Prepare y =====================================================
        # Get the labels of the batch
        y = list(map(itemgetter(1), batch))

        # if to return_sequences then y is also sequence and we need to ensure all have same length
        # else we return a tensor as column
        if self.return_sequences:
            try:
                y = pad_sequence(y, batch_first=True).type(torch.float)
            except IndexError:
                raise ValueError('Tried to pad target sequences but some targets were not passed as a sequence.')
        else:
            y = torch.tensor(y).unsqueeze(dim=1)

        # ==================================== Lengths =====================================================
        # This is later needed in order to pack the sequences
        lengths = torch.tensor(list(map(len, x)), dtype=torch.float)

        return VariableLengthTensor(x_padded, sequences_lengths=lengths), y


class VariableLengthTensorDataset(Dataset):
    def __init__(self, *tensors):
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return len(self.tensors[0])


class VariableLengthTensor(torch.Tensor):
    def __new__(cls, *args, sequences_lengths, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, sequences_lengths, **kwargs):
        self.seq_len = sequences_lengths


def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.

    :param h: Tensor(s) of the hidden cells to be detached
    :return: Tensor(s) repacked
    """

    if h is None:
        return None
    elif torch.is_tensor(h):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
