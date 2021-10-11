"""
Copyright (c) Facebook, Inc. and its affiliates, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
"""
from torch import nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


class LinearLayers(nn.Module):
    """
    One or multiple dense layers with skip connections from input to final output.
    """

    def __init__(self, hidden_size, num_layers=2, dropout_p=0.0, use_skip=False, use_batch_norm=True):
        super(LinearLayers, self).__init__()
        self.hidden_size = hidden_size

        layers = []
        for _ in range(num_layers):
            new_layers = [nn.Linear(hidden_size, hidden_size)]
            if use_batch_norm:
                bn = nn.BatchNorm1d(hidden_size)
                nn.init.uniform_(bn.weight)
                new_layers.append(bn)
            new_layers.append(nn.PReLU())
            new_layers.append(nn.Dropout(dropout_p))
            layers.extend(new_layers)

        self.layers = nn.Sequential(*layers)

        if use_skip:
            self.skip = lambda x, y: x + y
        else:
            self.skip = lambda x, y: y

    def forward(self, x):
        y = self.layers(x)
        out = self.skip(x, y)
        return out


class MLP(nn.Module):
    """
    An MLP mapping from input size to output size going through n hidden dense layers. Uses batch normalization,
    PReLU and can be configured to apply dropout.
    """

    def __init__(self, input_size, output_size, hidden_size, num_layers=2, dropout_p=0.0, skip_connection=False,
                 use_batch_norm=True):
        super(MLP, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_size)
            nn.init.uniform_(self.batch_norm.weight)
        else:
            self.batch_norm = nn.Identity()
        self.activation_fn = nn.PReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        hidden_layers = []
        for _ in range(num_layers):
            h = LinearLayers(hidden_size, dropout_p=dropout_p, use_batch_norm=use_batch_norm, use_skip=skip_connection)
            hidden_layers.append(h)
        self.hidden_layers = nn.Sequential(*hidden_layers)

    def forward(self, x):
        y = self.input_to_hidden(x)
        y = self.batch_norm(y)
        y = self.activation_fn(y)
        y = self.dropout(y)
        y = self.hidden_layers(y)
        y = self.hidden_to_output(y)
        return y


class RNNLayer(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers,
                 output_size=None, bidirectional=False, dropout=0.0, learn_init_state=False):
        """
        An LSTM-RNN.
        :param input_size: Input size.
        :param hidden_size: Hidden size of the RNN.
        :param num_layers: How many layers to use.
        :param output_size: If given, a dense layer is automatically added to map to this size.
        :param bidirectional: If the RNN should be bidirectional.
        :param dropout: Dropout applied directly to the inputs (off by default).
        :param learn_init_state: Whether to learn the initial hidden state.
        """
        super(RNNLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learn_init_state = learn_init_state
        self.is_bidirectional = bidirectional
        self.num_directions = 2 if self.is_bidirectional else 1

        if dropout > 0.0:
            self.input_drop = nn.Dropout(p=dropout)
        else:
            self.input_drop = nn.Identity()

        self.init_state = None
        self.final_state = None
        if self.learn_init_state:
            self.to_init_state_h = nn.Linear(self.input_size, self.hidden_size * self.num_layers * self.num_directions)
            self.to_init_state_c = nn.Linear(self.input_size, self.hidden_size * self.num_layers * self.num_directions)

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=self.is_bidirectional)

        if output_size is not None:
            self.to_out = nn.Linear(self.hidden_size * self.num_directions, output_size)
        else:
            self.to_out = nn.Identity()

    def cell_init(self, inputs_):
        """Return the initial state of the cell."""
        if self.learn_init_state:
            # Learn the initial state based on the first frame.
            c0 = self.to_init_state_c(inputs_[:, 0:1]).squeeze()
            c0 = c0.reshape(-1, self.num_layers, self.hidden_size).transpose(0, 1)
            h0 = self.to_init_state_h(inputs_[:, 0:1]).squeeze()
            h0 = h0.reshape(-1, self.num_layers, self.hidden_size).transpose(0, 1)
            return c0, h0
        else:
            return self.init_state

    def forward(self, x, seq_lengths):
        """
        Forward pass.
        :param x: A tensor of shape (N, F, input_size).
        :param seq_lengths: A tensor of shape (N, ) indicating the true sequence length for each batch entry.
        :return: The output of the RNN.
        """
        inputs_ = self.input_drop(x)

        # Get the initial state of the recurrent cells.
        self.init_state = self.cell_init(inputs_)

        # Make sure the padded frames are not shown to the LSTM.
        lstm_in = pack_padded_sequence(inputs_, seq_lengths, batch_first=True, enforce_sorted=False)

        # Feed it to the LSTM.
        lstm_out, final_state = self.lstm(lstm_in, self.init_state)
        self.final_state = final_state

        # Undo the packing.
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=inputs_.shape[1])  # (N, F, hidden)

        # May map to output size.
        outputs = self.to_out(lstm_out)  # (N, F, self.output_size)
        return outputs

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            if isinstance(module, nn.LSTM) and not mode:
                # Don't set the RNN into eval mode to avoid an exception thrown when using the RNN together with IEF.
                # This should not have an effect as train and eval mode in RNNs is exactly the same.
                continue
            module.train(mode)
        return self


class FeedForwardResidualBlock(nn.Module):
    """One residual block."""

    def __init__(self, input_size, output_size):
        super(FeedForwardResidualBlock, self).__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.activate = nn.ReLU()

    def forward(self, x):
        y = self.dense(x)
        y = y + x
        y = self.activate(y)
        return y
