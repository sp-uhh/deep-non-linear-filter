from typing import Literal
from torch import nn
import torch

"""
This file provides the torch modules for the paper

Kristina Tesch and Timo Gerkmann, "Insights into Deep Non-linear Filters for Improved Multi-channel Speech Enhancement", 
IEEE/ACM Transactions of Audio, Speech and Language Processing, vol 31, pp. 563-575, 2023.

and also

Kristina Tesch and Timo Gerkmann, "Multi-channel Speech Separation Using Spatially Selective Deeo Non-linear Filters", submitted to IEEE/ACM Transactions of Audio, Speech and Language Processing.

Included networks are:
JNF (implements T-JNF, F-JNF, T-NSF and F-NSF)
FTJNF (implements FT-JNF and FT-NSF)
JNF_SSF (implements FT-JNF with a conditioning on the DoA angle)

The network architecture T-JNF corresponds to the network proposed in 

X. Li und R. Horaud, „Multichannel Speech Enhancement Based On Time-Frequency Masking Using Subband Long Short-Term Memory“, 
in 2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA),  Okt. 2019, p. 298–302.
"""


class JNF(nn.Module):

    def __init__(self,
                 n_time_steps: int,
                 n_freqs: int,
                 n_channels: int,
                 n_lstm_hidden1: int = 256,
                 n_lstm_hidden2: int = 128,
                 bidirectional: bool = True,
                 output_type: Literal['IRM', 'CRM'] = 'CRM',
                 output_activation: Literal['sigmoid', 'tanh', 'linear'] = 'tanh', 
                 dropout: float = 0, 
                 append_freq_idx: bool = False,
                 permute_freqs: bool = False,
                 narrow_band: bool = False):
        """
        Initialize model.

        :param n_time_steps: number of STFT time frames in the input signal
        :param n_freqs: number of STFT frequency bins in the input signal
        :param n_channels: number of channel in the input signal
        :param n_lstm_hidden1: number of LSTM units in the first LSTM layer
        :param n_lstm_hidden2: number of LSTM units in the second LSTM layer
        :param bidirectional: set to True for a bidirectional LSTM
        :param output_type: set to 'IRM' for real-valued ideal ratio mask (IRM) and to 'CRM' for complex IRM
        :param output_activation: the activation function applied to the network output (options: 'sigmoid', 'tanh', 'linear')
        :param dropout: dropout percentage (default: no dropout)
        :param append_freq_idx: add the frequency-bin index to the input of the LSTM when using permuted sequences
        :param permute_freqs: permute the LSTM input sequence
        :param narrow_band: use narrow-band input if narrow_band else use wide-band input
        """
        super(JNF, self).__init__()

        self.n_time_steps = n_time_steps
        self.n_freqs = n_freqs
        self.n_channels = n_channels
        self.n_lstm_hidden1 = n_lstm_hidden1
        self.n_lstm_hidden2 = n_lstm_hidden2
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.output_type = output_type
        self.output_activation = output_activation
        self.append_freq_idx = append_freq_idx
        self.permute = permute_freqs
        self.narrow_band = narrow_band

        lstm_input = 2*n_channels
        if self.append_freq_idx and self.permute:
            lstm_input += 1

        self.lstm1 = nn.LSTM(input_size=lstm_input, hidden_size=self.n_lstm_hidden1, bidirectional=bidirectional, batch_first=False)
        self.lstm2 = nn.LSTM(input_size=2*self.n_lstm_hidden1, hidden_size=self.n_lstm_hidden2, bidirectional=bidirectional, batch_first=False)

        self.dropout = nn.Dropout(p=self.dropout)

        if self.output_type == 'IRM':
            self.linear_out_features = 1
        elif self.output_type == 'CRM':
            self.linear_out_features = 2
        else:
            raise ValueError(f'The output type {output_type} is not supported.')
        self.ff = nn.Linear(2*self.n_lstm_hidden2, out_features=self.linear_out_features)

        if self.output_activation == 'sigmoid':
            self.mask_activation = nn.Sigmoid()
        elif self.output_activation == 'tanh':
            self.mask_activation = nn.Tanh()
        elif self.output_activation == 'linear':
            self.mask_activation = nn.Identity()


    def forward(self, x: torch.Tensor):
        """
        Implements the forward pass of the model.

        :param x: input with shape [BATCH, CHANNEL, FREQ, TIME]
        :return: the output mask [BATCH, 1 (IRM) or 2 (CRM) , FREQ, TIME]
        """
        n_batch, n_channel, n_freq, n_times = x.shape

        if self.narrow_band:
            seq_len = n_times
            tmp_batch = n_batch*n_freq
            x = x.permute(3,0,2,1).reshape(n_times, n_batch*n_freq, n_channel)
        else: # wide_band
            seq_len = n_freq
            tmp_batch = n_batch * n_times
            x = x.permute(2,0,3,1).reshape(n_freq, n_batch*n_times, n_channel)

        if self.permute:
            perm = torch.randperm(seq_len)
            inv_perm = torch.zeros(seq_len, dtype=int)
            for i, val in enumerate(perm):
                inv_perm[val] = i
            x = x[perm]

            if self.append_freq_idx:
                if self.narrow_band:
                    freq_bins = torch.arange(n_freq).repeat(n_batch*n_times).reshape(seq_len, tmp_batch, 1).to(x.device)
                    x = torch.concat((x, freq_bins), dim=-1)
                else:
                    freq_bins = torch.arange(n_freq).repeat(int(seq_len/n_freq))[perm]
                    freq_bins = freq_bins.unsqueeze(1).unsqueeze(1).broadcast_to((seq_len, tmp_batch, 1)).to(x.device)
                    x = torch.concat((x, freq_bins), dim=-1)

        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = self.ff(x)

        if self.permute:
            x = x[inv_perm]

        if self.narrow_band:
            x = x.reshape(n_times, n_batch, n_freq, self.linear_out_features).permute(1,3,2,0)
        else: # wide_band
            x = x.reshape(n_freq, n_batch, n_times, self.linear_out_features).permute(1,3,0,2)
        x = self.mask_activation(x)
        return x


class FTJNF(nn.Module):
    """
    Mask estimation network composed of two LSTM layers. One LSTM layer uses the frequency-dimension as sequence input
    and the other LSTM uses the time-dimension as input.
    """
    def __init__(self,
                 n_channels: int,
                 n_lstm_hidden1: int = 512,
                 n_lstm_hidden2: int = 128,
                 bidirectional: bool = True,
                 freq_first: bool = True, 
                 output_type: Literal['IRM', 'CRM'] = 'CRM',
                 output_activation: Literal['sigmoid', 'tanh', 'linear'] = 'tanh', 
                 dropout: float = 0,
                 append_freq_idx: bool = False,
                 permute_freqs: bool = False):
        """
        Initialize model.

        :param n_channels: number of channel in the input signal
        :param n_lstm_hidden1: number of LSTM units in the first LSTM layer
        :param n_lstm_hidden2: number of LSTM units in the second LSTM layer
        :param bidirectional: set to True for a bidirectional LSTM
        :param freq_first: process frequency dimension first if freq_first else process time dimension first
        :param output_type: output_type: set to 'IRM' for real-valued ideal ratio mask (IRM) and to 'CRM' for complex IRM
        :param output_activation: the activation function applied to the network output (options: 'sigmoid', 'tanh', 'linear')
        :param dropout: dropout percentage (default: no dropout)
        """
        super(FTJNF, self).__init__()

        self.n_channels = n_channels
        self.n_lstm_hidden1 = n_lstm_hidden1
        self.n_lstm_hidden2 = n_lstm_hidden2
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.output_type = output_type
        self.output_activation = output_activation
        self.freq_first = freq_first
        self.append_freq_idx = append_freq_idx
        self.permute = permute_freqs

        lstm_input = 2*n_channels
        if self.append_freq_idx: 
            lstm_input += 1

        self.lstm1 = nn.LSTM(input_size=lstm_input, hidden_size=self.n_lstm_hidden1, bidirectional=bidirectional, batch_first=False)
        
        self.lstm1_out = 2*self.n_lstm_hidden1 if self.bidirectional else self.n_lstm_hidden1
        lstm2_input = self.lstm1_out
        if self.append_freq_idx: 
            lstm2_input+= 1

        self.lstm2 = nn.LSTM(input_size=lstm2_input, hidden_size=self.n_lstm_hidden2, bidirectional=bidirectional, batch_first=False)
        self.lstm2_out = 2*self.n_lstm_hidden2 if self.bidirectional else self.n_lstm_hidden2

        self.dropout = nn.Dropout(p=self.dropout)

        if self.output_type == 'IRM':
            self.linear_out_features = 1
        elif self.output_type == 'CRM':
            self.linear_out_features = 2
        else:
            raise ValueError(f'The output type {output_type} is not supported.')
        self.ff = nn.Linear(self.lstm2_out, out_features=self.linear_out_features)

        if self.output_activation == 'sigmoid':
            self.mask_activation = nn.Sigmoid()
        elif self.output_activation == 'tanh':
            self.mask_activation = nn.Tanh()
        elif self.output_activation == 'linear':
            self.mask_activation = nn.Identity()


    def forward(self, x: torch.Tensor):
        """
        Implements the forward pass of the model.

        :param x: input with shape [BATCH, CHANNEL, FREQ, TIME]
        :return: the output mask [BATCH, 1 (IRM) or 2 (CRM) , FREQ, TIME]
        """
        n_batch, n_channel, n_freq, n_times = x.shape

        if not self.freq_first: # narrow_band
            seq_len = n_times
            tmp_batch = n_batch*n_freq
            x = x.permute(3,0,2,1).reshape(n_times, n_batch*n_freq, n_channel)
        else: # wide_band
            seq_len = n_freq
            tmp_batch = n_batch*n_times
            x = x.permute(2,0,3,1).reshape(n_freq, n_batch*n_times, n_channel)

        if self.permute:
            perm = torch.randperm(seq_len)
            inv_perm = torch.zeros(seq_len, dtype=int)
            for i, val in enumerate(perm):
                inv_perm[val] = i
            x = x[perm]
        else:
            perm = torch.arange(seq_len)

        if self.append_freq_idx:
            if not self.freq_first: # narrow_band:
                freq_bins = torch.arange(n_freq).repeat(n_batch*n_times).reshape(seq_len, tmp_batch, 1).to(x.device)
                x = torch.concat((x, freq_bins), dim=-1)
            else: # wide_band
                freq_bins = torch.arange(n_freq).repeat(int(seq_len/n_freq))[perm]
                freq_bins = freq_bins.unsqueeze(1).unsqueeze(1).broadcast_to((seq_len, tmp_batch, 1)).to(x.device)
                x = torch.concat((x, freq_bins), dim=-1)

        x, _ = self.lstm1(x)
        x = self.dropout(x)

        if self.permute:
            x = x[inv_perm]

        if not self.freq_first: # narrow_band -> wide_band
            seq_len = n_freq
            tmp_batch = n_batch*n_times
            x = x.reshape(n_times, n_batch, n_freq, self.lstm1_out).permute(2,1,0,3).reshape(n_freq, n_batch*n_times, self.lstm1_out)
        else: # wide_band -> narrow_band
            seq_len = n_times
            tmp_batch = n_batch*n_freq
            x =  x.reshape(n_freq, n_batch, n_times, self.lstm1_out).permute(2,1,0,3).reshape(n_times, n_batch*n_freq, self.lstm1_out)

        if self.permute:
            perm = torch.randperm(seq_len)
            inv_perm = torch.zeros(seq_len, dtype=int)
            for i, val in enumerate(perm):
                inv_perm[val] = i
            x = x[perm]
        else:
            perm = torch.arange(seq_len)

        if self.append_freq_idx:
            if self.freq_first: # wide_band
                freq_bins = torch.arange(n_freq).repeat(n_batch*n_times).reshape(seq_len, tmp_batch, 1).to(x.device)
                x = torch.concat((x, freq_bins), dim=-1)
            else: # narrow_band
                freq_bins = torch.arange(n_freq).repeat(int(seq_len/n_freq))[perm]
                freq_bins = freq_bins.unsqueeze(1).unsqueeze(1).broadcast_to((seq_len, tmp_batch, 1)).to(x.device)
                x = torch.concat((x, freq_bins), dim=-1)

        x, _ = self.lstm2(x)
        x = self.dropout(x)

        if self.permute:
            x = x[inv_perm]

        x = self.ff(x)

        if not self.freq_first: # wide_band -> input shape
            x = x.reshape(n_freq, n_batch, n_times, self.linear_out_features).permute(1,3,0,2)
        else: # narrow_band -> input shape
            x = x.reshape(n_times, n_batch, n_freq, self.linear_out_features).permute(1,3,2,0)
        
        x = self.mask_activation(x)
        return x
    

class JNF_SSF(nn.Module):
    """
    Mask estimation network composed of two LSTM layers. One LSTM layer uses the frequency-dimension as sequence input
    and the other LSTM uses the time-dimension as input. 
    In addition to the noisy input, the network also gets a one-hot encoded DoA angle vector to indicate in which direction the target speaker is located. 
    """
    def __init__(self,
                 n_channels: int,
                 n_lstm_hidden1: int,
                 n_lstm_hidden2: int,
                 n_cond_emb_dim: int,
                 bidirectional: bool,
                 output_type: Literal['IRM', 'CRM'],
                 output_activation: Literal['sigmoid', 'tanh', 'linear'],
                 dropout: float = 0,
                 causal: bool = False, 
                 condition_nb_only: bool = False, 
                 condition_wb_only: bool = True):
        """
        Initialize model.

        :param n_channels: number of channel in the input signal
        :param n_lstm_hidden1: number of LSTM units in the first LSTM layer
        :param n_lstm_hidden2: number of LSTM units in the second LSTM layer
        :param bidirectional: set to True for a bidirectional LSTM
        :param output_type: output_type: set to 'IRM' for real-valued ideal ratio mask (IRM) and to 'CRM' for complex IRM
        :param output_activation: the activation function applied to the network output (options: 'sigmoid', 'tanh', 'linear')
        :param dropout: dropout percentage (default: no dropout)
        :param condition_wb_only: flag indicating if both LSTM layers or only the wide-band (first) should be conditioned on the target DoA
        :param condition_nb_only: flag indicating if both LSTM layers or only the narrowband (second) should be conditioned on the target DoA
        """
        super(JNF_SSF, self).__init__()

        self.n_channels = n_channels
        self.n_lstm_hidden1 = n_lstm_hidden1
        self.n_lstm_hidden2 = n_lstm_hidden2
        self.n_cond_emb_dim = n_cond_emb_dim
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.output_type = output_type
        self.output_activation = output_activation
        self.condition_nb_only = condition_nb_only 
        self.condition_wb_only = condition_wb_only

        assert not (condition_nb_only and condition_wb_only), "Config does not make sense."

        lstm_input = 2*n_channels

        if not self.condition_nb_only:
            self.cond_emb1 = nn.Linear(n_cond_emb_dim, self.n_lstm_hidden1)
        if not self.condition_wb_only:
            self.cond_emb2 = nn.Linear(n_cond_emb_dim, self.n_lstm_hidden2)

        self.lstm1 = nn.LSTM(input_size=lstm_input, hidden_size=self.n_lstm_hidden1,
                             bidirectional=self.bidirectional, batch_first=False)

        self.lstm1_out = 2*self.n_lstm_hidden1 if self.bidirectional else self.n_lstm_hidden1
        lstm2_input = self.lstm1_out

        self.bidirectional_second = (bidirectional and not causal)
        self.lstm2 = nn.LSTM(input_size=lstm2_input, hidden_size=self.n_lstm_hidden2,
                             bidirectional=self.bidirectional_second, batch_first=False)
        self.lstm2_out = 2*self.n_lstm_hidden2 if self.bidirectional_second else self.n_lstm_hidden2

        self.dropout = nn.Dropout(p=self.dropout)

        if self.output_type == 'IRM':
            self.linear_out_features = 1
        elif self.output_type == 'CRM':
            self.linear_out_features = 2
        else:
            raise ValueError(
                f'The output type {output_type} is not supported.')
        self.ff = nn.Linear(
            self.lstm2_out, out_features=self.linear_out_features)

        if self.output_activation == 'sigmoid':
            self.mask_activation = nn.Sigmoid()
        elif self.output_activation == 'tanh':
            self.mask_activation = nn.Tanh()
        elif self.output_activation == 'linear':
            self.mask_activation = nn.Identity()

    def forward(self, x: torch.Tensor, target_dirs: torch.Tensor, device: str):
        """
        Implements the forward pass of the model.

        :param x: input with shape [BATCH, CHANNEL, FREQ, TIME]
        :param target_dirs: the conditional input [BATCH, 1 (IDX)]
        :return: the output mask [BATCH, 1 (IRM) or 2 (CRM) , FREQ, TIME]
        """
        n_batch, n_channel, n_freq, n_times = x.shape


        # wide_band
        tmp_batch = n_batch*n_times
        bidirectional_dim = 2 if self.bidirectional else 1
        x = x.permute(2, 0, 3, 1).reshape(
            n_freq, n_batch*n_times, n_channel)

        if self.condition_nb_only:
            x, _ = self.lstm1(x)
        else:
            x_cond_emb1 = self.cond_emb1(target_dirs)
            x_cond_emb1_reshaped1 = x_cond_emb1.unsqueeze(0).unsqueeze(2).repeat(
            bidirectional_dim, 1, n_times, 1).reshape(bidirectional_dim, tmp_batch, self.n_lstm_hidden1)
            x, _ = self.lstm1(x,
                          (torch.zeros(bidirectional_dim, tmp_batch, self.n_lstm_hidden1, device=device), x_cond_emb1_reshaped1))
        x = self.dropout(x)

        # narrow_band
        tmp_batch = n_batch*n_freq
        x = x.reshape(n_freq, n_batch, n_times, self.lstm1_out).permute(
            2, 1, 0, 3).reshape(n_times, n_batch*n_freq, self.lstm1_out)
        if self.condition_wb_only:
            x, _ = self.lstm2(x)
        else:
            x_cond_emb2 = self.cond_emb2(target_dirs)
            x_cond_emb2_reshaped2 = x_cond_emb2.unsqueeze(0).unsqueeze(2).repeat(
                bidirectional_dim, 1, n_freq, 1).reshape(bidirectional_dim, tmp_batch, self.n_lstm_hidden2)

            x, _ = self.lstm2(x,
                            (torch.zeros(bidirectional_dim, tmp_batch, self.n_lstm_hidden2, device=device), x_cond_emb2_reshaped2))
        x = self.dropout(x)

        x = self.ff(x)

        # time_slice -> input shape
        x = x.reshape(n_times, n_batch, n_freq,
                          self.linear_out_features).permute(1, 3, 2, 0)

        x = self.mask_activation(x)
        return x