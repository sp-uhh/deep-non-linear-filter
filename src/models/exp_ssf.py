import pytorch_lightning as pl
import torch
from torch import nn
import numpy as np
from models.exp_enhancement import EnhancementExp
from torchmetrics.aggregation import RunningMean
from typing import Literal


class SSFExp(EnhancementExp):

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        weight_decay: float,
        loss_alpha: float,
        stft_length: int,
        stft_shift: int,
        cirm_comp_K: float,
        cirm_comp_C: float,
        n_cond_emb_dim: int,
        condition_enc_type: Literal["index", "arange"],
        cond_arange_params: tuple = None,
        loss_type: str = "l1",
        scheduler_type: str = None,
        scheduler_params: dict = None,
        reference_channel: int = 0,
    ):
        super(SSFExp, self).__init__(
            model=model,
            cirm_comp_K=cirm_comp_K,
            cirm_comp_C=cirm_comp_C,
            scheduler_type=scheduler_type,
            scheduler_params=scheduler_params,
        )

        self.model = model

        self.stft_length = stft_length
        self.stft_shift = stft_shift

        self.cirm_K = cirm_comp_K
        self.cirm_C = cirm_comp_C

        self.reference_channel = reference_channel

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_alpha = loss_alpha

        self.n_cond_emb_dim = n_cond_emb_dim
        self.condition_enc_type = condition_enc_type
        self.cond_arange_params = cond_arange_params
        self.loss_type = loss_type

        self.running_loss = RunningMean(window=20)

        if self.condition_enc_type == "arange":
            assert cond_arange_params is not None, "Angle range parameters are missing"
            start, stop, step = cond_arange_params
            angles = range(start, stop, step)
            n_angles = len(angles)
            indices = range(n_angles)
            assert (
                n_angles == n_cond_emb_dim
            ), "The embedding dim does not match the angle range params"
            self.angle_index_map = dict(zip(angles, indices))

        # self.example_input_array = torch.from_numpy(np.ones((2, 6, 513, 75), dtype=np.float32))

    def forward(self, input, target_dir):
        target_dir_enc = self.encode_condition(target_dir)
        speech_mask = self.model(input, target_dir_enc, device=self.device)
        return speech_mask

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.shared_step(batch,
                                batch_idx,
                                stage="val",
                                dataloader_idx=dataloader_idx)

    def shared_step(self,
                    batch,
                    batch_idx,
                    stage: Literal["train", "val"],
                    dataloader_idx=0):
        noisy_td, clean_td, noise_td = (
            batch["noisy_td"],
            batch["clean_td"],
            batch["noise_td"],
        )
        noisy_stft, clean_stft, noise_stft = self.get_stft_rep(
            noisy_td, clean_td, noise_td)

        # compute mask estimate
        stacked_noisy_stft = torch.concat(
            (torch.real(noisy_stft), torch.imag(noisy_stft)), dim=1)

        target_dirs = batch["target_dir"]
        target_dirs_enc = self.encode_condition(target_dirs)

        if self.model.output_type == "IRM":
            irm_speech_mask = self.model(stacked_noisy_stft,
                                         target_dirs_enc,
                                         device=self.device)
            speech_mask, noise_mask = irm_speech_mask, 1 - irm_speech_mask
        elif self.model.output_type == "CRM":
            stacked_speech_mask = self.model(stacked_noisy_stft,
                                             target_dirs_enc,
                                             device=self.device)
            speech_mask, noise_mask = self.get_complex_masks_from_stacked(
                stacked_speech_mask)
        else:
            raise ValueError(
                f"The output type {self.model.output_type} is not supported.")

        # compute estimates
        est_clean_stft = noisy_stft[:, self.reference_channel,
                                    ...] * speech_mask
        est_noise_stft = noisy_stft[:, self.reference_channel,
                                    ...] * noise_mask
        clean_td, noise_td, est_clean_td, est_noise_td = self.get_td_rep(
            clean_stft[:, self.reference_channel, ...],
            noise_stft[:, self.reference_channel, ...],
            est_clean_stft,
            est_noise_stft,
        )

        # compute loss
        if self.loss_type == "l1":
            clean_td_loss, noise_td_loss, clean_mag_loss, noise_mag_loss = self.loss(
                clean_td,
                est_clean_td,
                noise_td,
                est_noise_td,
                clean_stft[:, 0, ...],
                est_clean_stft,
                noise_stft[:, 0, ...],
                est_noise_stft,
            )

            loss = torch.mean(self.loss_alpha * clean_td_loss + clean_mag_loss)
        elif self.loss_type == "sisdr":
            loss = -torch.mean(
                self.compute_global_si_sdr(est_clean_td, clean_td))

        # logging
        if stage == "train" or dataloader_idx == 0 or dataloader_idx is None:
            add_dataloader_idx = False
        else:
            add_dataloader_idx = True

        self.running_loss(loss)
        on_step = True if stage == 'train' else False
        self.log(
            f"{stage}/loss",
            self.running_loss.compute(),
            on_step=on_step,
            on_epoch=True,
            logger=True,
            add_dataloader_idx=add_dataloader_idx,
            sync_dist=True,
            prog_bar=True,
        )
        if self.loss_type == "l1":
            self.log(
                f"{stage}/clean_td_loss",
                clean_td_loss.mean(),
                on_step=on_step,
                on_epoch=True,
                logger=True,
                add_dataloader_idx=add_dataloader_idx,
                sync_dist=True,
            )

            self.log(
                f"{stage}/clean_mag_loss",
                clean_mag_loss.mean(),
                on_step=on_step,
                on_epoch=True,
                logger=True,
                add_dataloader_idx=add_dataloader_idx,
                sync_dist=True,
            )

        if batch_idx < 1:
            self.log_batch_detailed_audio(noisy_td[:, 0, ...], est_clean_td,
                                          batch_idx, stage)
            self.log_batch_detailed_spectrograms(
                [
                    noisy_stft[:, self.reference_channel, ...],
                    clean_stft[:, self.reference_channel, ...],
                    noise_stft[:, self.reference_channel, ...],
                    est_clean_stft,
                    est_noise_stft,
                ],
                batch_idx,
                stage,
                n_samples=10,
            )
            # self.log_batch_detailed_maks([complex_speech_mask.abs(), complex_noise_mask.abs()], batch_idx, stage, n_samples=10)
        if stage == "val":
            if dataloader_idx == 0:
                self.log(
                    "monitor_loss",
                    loss,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    add_dataloader_idx=add_dataloader_idx,
                    sync_dist=True,
                )
            global_si_sdr = torch.mean(
                self.compute_global_si_sdr(est_clean_td, clean_td))
            self.log(
                "val/si_sdr",
                global_si_sdr,
                on_epoch=True,
                logger=True,
                add_dataloader_idx=add_dataloader_idx,
                sync_dist=True,
            )

        return loss

    def encode_condition(self, target_dirs):
        """
        Provide an encoding of the target direction of length self.n_cond_emb_dim using the specified encoding strategy.

        Encoding strategys:
        - index: The target dir is already an index of the direction and only need a one_hot encoding.

        - arange: The target dirs have been generated using the arange function and will first be mapped to indices and then be one_hot encoded

        """

        if self.condition_enc_type == "index":
            return torch.nn.functional.one_hot(target_dirs,
                                               self.n_cond_emb_dim).float()

        elif self.condition_enc_type == "arange":
            index_mapped = (target_dirs.cpu().apply_(
                self.angle_index_map.get).to(self.device))
            return torch.nn.functional.one_hot(index_mapped,
                                               self.n_cond_emb_dim).float()
