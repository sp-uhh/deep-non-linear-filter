from typing import Literal
import torch
from torch import nn
from models.exp_enhancement import EnhancementExp
from models.models import FTJNF

class JNFExp(EnhancementExp):

    def __init__(self,
                 model: nn.Module,
                 learning_rate: float,
                 weight_decay: float, 
                 loss_alpha: float,
                 stft_length: int,
                 stft_shift: int,
                 cirm_comp_K: float,
                 cirm_comp_C: float, 
                 reference_channel: int = 0):
        super(JNFExp, self).__init__(model=model, cirm_comp_K=cirm_comp_K, cirm_comp_C=cirm_comp_C)

        self.model = model

        self.stft_length = stft_length
        self.stft_shift = stft_shift

        self.cirm_K = cirm_comp_K
        self.cirm_C = cirm_comp_C

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_alpha = loss_alpha

        self.reference_channel = reference_channel

        #self.example_input_array = torch.from_numpy(np.ones((2, 6, 513, 75), dtype=np.float32))

    def forward(self, input):
        speech_mask = self.model(input)
        return speech_mask

    def shared_step(self, batch, batch_idx, stage: Literal['train', 'val']):

        noisy_td, clean_td, noise_td = batch['noisy_td'], batch['clean_td'], batch['noise_td']
        noisy_stft, clean_stft, noise_stft = self.get_stft_rep(noisy_td, clean_td, noise_td)

        # compute mask estimate
        stacked_noisy_stft = torch.concat((torch.real(noisy_stft), torch.imag(noisy_stft)), dim=1)

        if self.model.output_type == 'IRM':
            irm_speech_mask = self.model(stacked_noisy_stft)
            speech_mask, noise_mask = irm_speech_mask, 1-irm_speech_mask
        elif self.model.output_type == 'CRM':
            stacked_speech_mask = self.model(stacked_noisy_stft)
            speech_mask, noise_mask = self.get_complex_masks_from_stacked(stacked_speech_mask)
        else:
            raise ValueError(f'The output type {self.model.output_type} is not supported.')

        # compute estimates
        est_clean_stft = noisy_stft[:, self.reference_channel, ...] * speech_mask
        est_noise_stft = noisy_stft[:, self.reference_channel, ...] * noise_mask
        clean_td, noise_td, est_clean_td, est_noise_td = self.get_td_rep(clean_stft[:, self.reference_channel, ...], noise_stft[:, self.reference_channel, ...],
                                                                         est_clean_stft, est_noise_stft)

        # compute loss
        clean_td_loss, noise_td_loss, clean_mag_loss, noise_mag_loss = self.loss(clean_td, est_clean_td, noise_td,
                                                                                 est_noise_td,
                                                                                 clean_stft[:, self.reference_channel, ...], est_clean_stft,
                                                                                 noise_stft[:, self.reference_channel, ...],
                                                                                 est_noise_stft)

        loss = torch.mean(self.loss_alpha * (clean_td_loss + noise_td_loss) + (clean_mag_loss + noise_mag_loss))

        # logging
        on_step = False
        self.log(f'{stage}/loss', loss, on_step=on_step, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}/clean_td_loss', clean_td_loss.mean(), on_step=on_step, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}/noise_td_loss', noise_td_loss.mean(), on_step=on_step, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}/clean_mag_loss', clean_mag_loss.mean(), on_step=on_step, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}/noise_mag_loss', noise_mag_loss.mean(), on_step=on_step, on_epoch=True, logger=True, sync_dist=True)
        if batch_idx < 1:
            self.log_batch_detailed_audio(noisy_td[:, self.reference_channel, ...], est_clean_td, batch_idx, stage)
            self.log_batch_detailed_spectrograms(
                [noisy_stft[:, self.reference_channel, ...], clean_stft[:, self.reference_channel, ...], noise_stft[:, self.reference_channel, ...], est_clean_stft, est_noise_stft],
                batch_idx,
                stage, n_samples=10)
            # self.log_batch_detailed_maks([complex_speech_mask.abs(), complex_noise_mask.abs()], batch_idx, stage, n_samples=10)
        if stage == 'val':
            self.log(f'monitor_loss', loss, on_step=on_step, on_epoch=True, logger=True)
            global_si_sdr = self.compute_global_si_sdr(est_clean_td, clean_td)
            self.log('val/si_sdr', global_si_sdr.mean(), on_epoch=True, logger=True, sync_dist=True)

        return loss