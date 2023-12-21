import pytorch_lightning as pl
import torch
from torch import nn
import numpy as np
from typing import List, Union, Literal
from utils.log_images import make_image_grid
from torch.optim import Adam


class EnhancementExp(pl.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 cirm_comp_K: float,
                 cirm_comp_C: float,
                 scheduler_type: str = None,
                 scheduler_params: dict = None
                 ):
        super(EnhancementExp, self).__init__()

        self.model = model

        self.cirm_K = cirm_comp_K
        self.cirm_C = cirm_comp_C

        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params

    def forward(self, input):
        pass

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage='val')


    def shared_step(self, batch, batch_idx, stage: Literal['train', 'val']):
        pass

    def loss(self, clean_td, est_clean_td, noise_td, est_noise_td,
             clean_stft, est_clean_stft, noise_stft, est_noise_stft):
        """
        Compute the loss based on L1-norms of time domain speech and noise signals and frequency magnitudes.

        :param clean_td: target clean signal in time domain
        :param est_clean_td: estimated clean signal in time domain
        :param noise_td: target noise signal in time domain
        :param est_noise_td: estimated noise signal in time domain
        :param clean_stft: target clean signal in STFT domain
        :param est_clean_stft: estimated clean signal in STFT domain
        :param noise_stft: target noise signal in STFT domain
        :param est_noise_stft: estimated noise signal in STFT domain
        :return: four loss terms based on L1-loss
        """
        clean_td_loss = torch.mean(torch.abs(clean_td - est_clean_td), dim=1)
        noise_td_loss = torch.mean(torch.abs(noise_td - est_noise_td), dim=1)
        clean_mag_loss = torch.mean(torch.abs(torch.abs(clean_stft) - torch.abs(est_clean_stft)))
        noise_mag_loss = torch.mean(torch.abs(torch.abs(noise_stft) - torch.abs(est_noise_stft)))

        return clean_td_loss, noise_td_loss, clean_mag_loss, noise_mag_loss
    
    def compute_global_si_sdr(self, est_clean_td, clean_td):
        """
        Compute the SI-SDR for a whole utterance. 

        :param enhanced_td: estimated clean signal in the time domain
        :param clean_td: clean signal in the time domain
        """

        def si_sdr(s, s_hat):
            alpha = torch.einsum('cs,cs->c', s_hat, s) / torch.einsum('cs,cs->c', s, s)
            scaled_ref = torch.unsqueeze(alpha, dim=1) * s
            sdr = 10 * torch.log10(torch.einsum('cs,cs->c', scaled_ref, scaled_ref) / (
                    torch.einsum('cs,cs->c', scaled_ref - s_hat, scaled_ref - s_hat) + 1e-14))
            return sdr

        enhanced_si_sdr = si_sdr(clean_td, est_clean_td)

        return enhanced_si_sdr

    def configure_optimizers(self):

        opt = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.scheduler_type == "ReduceLROnPLateau":
           lr_scheduler={
               'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, **self.scheduler_params),
               'name': 'lr_schedule',
               'monitor': 'monitor_loss'
           }
           return opt, lr_scheduler
        if self.scheduler_type == "MultiStepLR":
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer=opt, **self.scheduler_params),
                'name': 'lr_schedule'
            }
            return {"optimizer": opt, "lr_scheduler": lr_scheduler}

        return opt

    def get_complex_masks_from_stacked(self, real_mask):
        """
        Construct the complex clean speech and noise mask from the estimated stacked clean speech mask. Inverts the
        compression by tanh output activation to the range [-inf, inf] for real and imaginary components.

        :param real_mask: estimated mask with stacked real and imaginary components [BATCH, 2, F, T]
        :return: the complex masks [B, F, T]
        """
        compressed_complex_speech_mask = real_mask[:, 0, ...] + (1j) * real_mask[:, 1, ...]

        complex_speech_mask = (-1 / self.cirm_C) * torch.log(
            (self.cirm_K - self.cirm_K * compressed_complex_speech_mask) / (
                    self.cirm_K + self.cirm_K * compressed_complex_speech_mask))
        complex_noise_mask = (1 - torch.real(complex_speech_mask)) - (1j) * torch.imag(complex_speech_mask)

        return complex_speech_mask, complex_noise_mask

    def get_stft_rep(self, *td_signals, return_complex=True):
        """
        Compute the STFT for the given time-domain signals.

        :param *td_signals: the time-domain signals (list of arrays [CHANNEL, SAMPLES] or [SAMPLES])
        :return: list of stfts [BATCH, CHANNEL, FREQ, TIME]
        """
        result = []
        window = torch.sqrt(torch.hann_window(self.stft_length)).to(device=self.device)
        #window = torch.hann_window(self.stft_length).to(device=self.device)
        for td_signal in td_signals:
            if len(td_signal.shape) == 1:  # single-channel
                stft = torch.stft(td_signal, self.stft_length, self.stft_shift, window=window, center=True,
                                  onesided=True, return_complex=return_complex)
                result.append(stft)
            else:  # multi-channel and/or multiple speakers
                signal_shape = td_signal.shape
                reshaped_signal = td_signal.reshape((signal_shape[:-1].numel(), signal_shape[-1]))
                stfts = torch.stft(reshaped_signal, self.stft_length, self.stft_shift, window=window, center=True, onesided=True, return_complex=return_complex)
                if return_complex:
                    combined_dim, freq_dim, time_dim = stfts.shape
                    stfts = stfts.reshape(signal_shape[:-1]+(freq_dim, time_dim))
                else:
                    comined_dim, freq_dim, time_dim, complex_dim = stfts.shape
                    stfts = stfts.reshape(signal_shape[:-1]+(freq_dim, time_dim, complex_dim))
                result.append(stfts)

        return result

    def get_td_rep(self, *stfts):
        """
        Compute the time domain represetnation for the given STFTs.
        :param stfts: list of STFTs [BATCH, FREQ, TIME]
        :return: list of time domain signals [BATCH, SAMPLES]
        """
        result = []
        window = torch.sqrt(torch.hann_window(self.stft_length)).to(device=self.device)
        for stft in stfts:
            has_complex_dim = stft.shape[-1] == 2
            if (not has_complex_dim and len(stft.shape) <= 3) or (has_complex_dim and len(stft.shape) <= 4):  # single-channel
                td_signal = torch.istft(stft, self.stft_length, self.stft_shift, window=window, center=True,
                                        onesided=True,
                                        return_complex=False)
                result.append(td_signal)
            else:  # multi-channel
                signal_shape = stft.shape
                if not has_complex_dim:
                    reshaped_signal = stft.reshape((signal_shape[:-2].numel(), signal_shape[-2], signal_shape[-1]))
                    td_signals = torch.istft(reshaped_signal, self.stft_length, self.stft_shift, window=window, center=True, onesided=True, return_complex=False)
                    combined_dim, n_samples = td_signals.shape
                    td_signals = td_signals.reshape(signal_shape[:-2]+(n_samples,))
                else:
                    reshaped_signal = stft.reshape((signal_shape[:-3].numel(), signal_shape[-3], signal_shape[-2], signal_shape[-1]))
                    td_signals = torch.istft(reshaped_signal, self.stft_length, self.stft_shift, window=window, center=True, onesided=True, return_complex=False)
                    combined_dim, n_samples = td_signals.shape
                    td_signals = td_signals.reshape(signal_shape[:-3]+(n_samples,))
                result.append(td_signals)
        return result

    def log_batch_detailed_spectrograms(self,
                                        stfts: List[torch.Tensor],
                                        batch_idx: Union[int, None],
                                        tag: str = 'train',
                                        n_samples: int = -1):
        """
        Write spectrograms for a batch.

        The spectrograms are reordered so that the ith sample of all STFTs are displayed in the same row
        (e.g. noisy, clean, noise and enhanced side-by-side).

        :param stfts: a list of the STFTs [BATCH, FREQ, TIME]
        :param batch_idx: the batch index
        :param tag: the logging tag (e.g. val vs. train)
        :param n_samples: the number of samples to log (default is full batch)
        """

        tensorboard = self.logger.experiment

        log_name = f"{tag}/spectrogram{'_' + str(batch_idx) if not batch_idx is None else ''}"

        combined_stfts = torch.flatten(torch.stack(stfts, dim=1), start_dim=0, end_dim=1)
        if n_samples > 0:
            combined_stfts = combined_stfts[:n_samples * len(stfts)]
        spectrograms_db = 10 * torch.log10(torch.maximum(torch.square(torch.abs(combined_stfts)),
                                                         (10 ** (-15)) * torch.ones_like(combined_stfts,
                                                                                         dtype=torch.float32)))
        spectrograms_db = torch.flip(torch.unsqueeze(spectrograms_db, dim=1), dims=[-2])
        tensorboard.add_image(log_name, make_image_grid(spectrograms_db, vmin=-80, vmax=20, n_img_per_row=len(stfts)),
                              global_step=self.current_epoch)


    def log_batch_detailed_audio(self, noisy_td, enhanced_td, batch_idx: Union[int, None], tag: str,
                                 n_samples: int = 10):
        """
        Write audio logs for a batch.

        :param noisy_stft: the noisy stft [BATCH, FREQ, TIME]
        :param enhanced_stft: the enhanced stft
        :param batch_idx: the batch index
        :param tag: the logging tag (e.g. val vs. train)
        """
        tensorboard = self.logger.experiment

        cur_samples = len(noisy_td)
        for i in range(min(self.trainer.datamodule.batch_size, n_samples, cur_samples)):
            log_noisy_name = f"{tag}/{str(batch_idx) if not batch_idx is None else ''}_{i}_noisy"
            tensorboard.add_audio(log_noisy_name, noisy_td[i], global_step=self.current_epoch,
                                  sample_rate=self.trainer.datamodule.fs)

            log_enhanced_name = f"{tag}/{str(batch_idx) if not batch_idx is None else ''}_{i}_enhanced"
            tensorboard.add_audio(log_enhanced_name, enhanced_td[i], global_step=self.current_epoch,
                                  sample_rate=self.trainer.datamodule.fs)
