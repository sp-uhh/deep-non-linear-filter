import json
from typing import List
import soundfile as sf
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import glob
import numpy as np
import h5py
import os
import random
random.seed(12345)
"""
Dataset generation for a speaker in a fixed location relative to the microphone array. This preprocessing script creates a HDF5 file with three datasets: 
- train
- val
- test

Each dataset has the shape [NUM_SAMPLES, 3, CHANNELS, MAX_SAMPLES_PER_FILE]. In the second axis, we store in this order the 
- spatialized target signal (includes reverb)
- the spatialized noise signal (sum of all interfering speakers)
- the dry target signal (including the time-shift caused by the direct path)


The code in this file was partially written by Nils Mohrmann. 
"""

# WSJ0 dataset path
WSJ0_PATH = "/path/to/wsj0/CSR-1-WSJ-0/WAV/wsj0"
# Path where to save the simulated data
SIM_DATA_PATH = "./prep/"


class RoomSimulation:

    def __init__(self, channels):
        self.channels = channels

    def set_room_properties(self, rt: float, room_dim: np.ndarray):
        """
        Recreate room with a new reverberation time, this deletes all sources and mics.
        :param rt: reverberation time
        :param room_dim: room dimension ([x,y,z])
        :return: None
        """
        self.rt60_tgt = rt
        if self.rt60_tgt > 0:
            e_absorption, max_order = pra.inverse_sabine(
                self.rt60_tgt, room_dim)

            self.room = pra.ShoeBox(room_dim,
                                    fs=16000,
                                    materials=pra.Material(e_absorption),
                                    max_order=max_order)
        else:
            e_absorption, max_order = pra.inverse_sabine(0.5, room_dim)
            self.room = pra.ShoeBox(room_dim,
                                    fs=16000,
                                    materials=pra.Material(e_absorption),
                                    max_order=0)

    def set_microphones(self, x: float, y: float, z: float, phi: float):
        """
        Add microphone array at position xyz with rotation phi
        Radius: 0.05 m
        :param x: x pos
        :param y: y pos
        :param z: z pos
        :param phi: The counterclockwise rotation of the first element in the array (from the x-axis)
        :return:
        """
        if self.channels == 2:
            # special orientation for 2 mics. -> speaker at broadside
            phi += np.pi / 2
        R = pra.beamforming.circular_2D_array(
            [x, y], self.channels, phi0=phi, radius=0.05)
        R = np.vstack((R, [[z] * self.channels]))
        self.room.add_microphone_array(pra.Beamformer(R, self.room.fs))

    def add_source(self, position: np.ndarray, signal: np.ndarray, delay: float):
        """
        Add signal source in room with a delay
        :param position: position [x, y, z]
        :param signal: The signal played by the source
        :param delay: A time delay until the source signal starts in the simulation
        :return: None
        """
        self.room.add_source(position, signal, delay)

    def measure_time(self):
        """
        Get measured RT60
        :return: rt60 in seconds
        """
        self.room.compute_rir()
        return self.room.measure_rt60()


class SPRoomSimulator:
    """
    Generate dataset for the training of the nonlinear training
    """

    def __init__(self, channels=3, seed=13, mode="train"):
        self.training = True
        if mode == "train":
            path = "si_tr_s"
        elif mode == "val":
            path = "si_dt_20"
        elif mode == "test":
            path = "si_et_05"
            self.training = False
        path = f'{WSJ0_PATH}/{path}'

        self.speaker = glob.glob(path + "/*")

        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.speaker)
        self.channels = channels
        self.exp_room = RoomSimulation(channels=self.channels)
        self.dry_room = RoomSimulation(channels=self.channels)
        self.fs = 16000

    def create_sample(self,
                      speaker_list: List[str],
                      seed2: int,
                      reverb: bool = True,
                      target_angle: float = 0,
                      rt60_min: float = 0.2,
                      rt60_max: float = 1,
                      snr_min: int = -10,
                      snr_max: int = 5,
                      side_room: int = 20):
        """
        Create for a list of speech signals (first one is the target signal) the spatial image using a randomly placed
        microphone array and distributing the interfering speakers (len(speaker_list)-1) uniformly around the array.

        :param n_interfering: number of interfering speaker
        :param seed2: Seed for the random audio files and positions
        :target_angle: place source at given fixed angle (given in degree)
        :reverb: turn off reverberation if set to False
        :rt60_min: minimum T60 reverb time
        :rt60_max: maximum T60 reverb time
        :side_room: angle of closest interfering source (default: 45 deg)
        :return: the audio signals as numpy array [N_SPEAKERS, N_CHANNELS, N_SAMPLES] and corresponding meta data
        """
        # set seed for this sample
        self.rng = np.random.default_rng(seed2)
        meta = {}

        signal = []
        for file in speaker_list:
            audio, fs = sf.read(file)
            signal.append(audio / np.max(np.abs(audio)) * 0.3)

        # ensure noise signal is long enough and does not start with zeros always 
        offset_indices = np.random.randint(
            low=-8000, high=8000, size=len(speaker_list)-1)
        target_signal_len = len(signal[0])
        for i in range(len(speaker_list)-1):
            new_signal = np.roll(
                np.resize(signal[1+i], target_signal_len), shift=offset_indices[i])
            signal[1+i] = new_signal

        # room properties
        RT = self.rng.uniform(rt60_min, rt60_max) if reverb else 0
        meta["rt"] = RT
        room_dim = np.squeeze(np.array([np.random.uniform(
            2.5, 5, 1), np.random.uniform(3, 9, 1), np.random.uniform(2.2, 3.5, 1)]))
        meta["room_dim"] = [room_dim[0], room_dim[1], room_dim[2]]
        self.exp_room.set_room_properties(RT, np.array(room_dim))
        self.dry_room.set_room_properties(0, np.array(room_dim))


        # random mic position in room (min 1 m to wall)
        mic_pos = self.rng.random(3) * (room_dim - 2.02) + 1.01
        mic_pos[2] = 1.5
        phi = self.rng.random() * 2 * np.pi  # microphone rotation
        self.exp_room.set_microphones(mic_pos[0], mic_pos[1], mic_pos[2], phi)
        self.dry_room.set_microphones(mic_pos[0], mic_pos[1], mic_pos[2], phi)

        meta["mic_pos"] = mic_pos.tolist()
        meta["mic_phi"] = phi

        # target speaker
        target_phi = phi + target_angle / 360 * 2 * np.pi
        main_source = mic_pos + \
                normal_vec(target_phi) * (self.rng.random() * 0.7 + 0.3)
        main_source[2] = self.rng.normal(1.60, 0.08) # height of speaker

        self.exp_room.add_source(main_source, signal[0], 0)
        self.dry_room.add_source(main_source, signal[0], 0)
        
        meta["target_file"] = speaker_list[0].split(
            "wsj0")[-1].replace("\\", "/")
        meta["n_samples"] = len(signal[0])
        meta["target_pos"] = main_source.tolist()
        meta["target_angle"] = target_angle

        # interering speakers
        n_interfering = len(speaker_list) - 1
        for interf_idx, interf_path in enumerate(speaker_list[1:]):
            for moveback in np.arange(0, 8, 0.25):
                # if pos outside from room, move back to the microphone
                # distance max 7 m, min 1 m
                side_room_rad = 2*np.pi/360*side_room
                speaker_range = (2*np.pi-2*side_room_rad)/n_interfering

                interf_source = mic_pos + normal_vec(
                    target_phi + side_room_rad + speaker_range * self.rng.random() + interf_idx * speaker_range) \
                    * max(1, self.rng.random() * 7 - moveback)

                # height of speaker is round about the height of standing people
                interf_source[2] = self.rng.normal(1.60, 0.08)
                if self.exp_room.room.is_inside(interf_source) and np.all(interf_source >= 0):
                    # if inside room, no need to move further to the mic
                    break

            self.exp_room.add_source(interf_source, signal[interf_idx + 1], 0)
            meta[f"interf{interf_idx}_file"] = interf_path.split(
                "wsj0")[-1].replace("\\", "/")
            meta[f"interf{interf_idx}_pos"] = interf_source.tolist()

        # return_premix allows separation of speaker signals
        mic_signals = self.exp_room.room.simulate(return_premix=True)
        dry_target_signal = self.dry_room.room.simulate(return_premix=True)

        reverb_target_signal = mic_signals[0, ...]
        noise_signal = np.sum(mic_signals[1:, ...], axis=0)
        dry_target_signal = dry_target_signal[0, ...]

        # scale to SNR 
        if not snr_min is np.nan:
            target_snr = self.rng.uniform(snr_min, snr_max)
            noise_factor = snr_scale_factor(
                reverb_target_signal, noise_signal, target_snr)
            noise_signal = noise_signal * noise_factor

            meta["snr"] = target_snr

        return reverb_target_signal, noise_signal, dry_target_signal, meta

    def get_room(self):
        return self.exp_room.room

    def plot(self):
        self.exp_room.plot()


def normal_vec(phi):
    return np.array([np.cos(phi), np.sin(phi), 0])


def snr_scale_factor(speech: np.ndarray, noise: np.ndarray, snr: int):
    """
    Compute the scale factor that has to be applied to a noise signal in order for the noisy (sum of noise and clean)
    to have the specified SNR.

    :param speech: the clean speech signal [..., SAMPLES]
    :param noise: the noise signal [..., SAMPLES]
    :param snr: the SNR of the mixture
    :return: the scaling factor
    """

    noise_var = np.mean(np.var(noise, axis=-1))
    speech_var = np.mean(np.var(speech, axis=-1))

    factor = np.sqrt(
        speech_var / np.maximum((noise_var * 10. ** (snr / 10.)), 10**(-6)))

    return factor


def prep_speaker_mix_data(store_dir: str,
                          post_fix: str = None,
                          wsj0_path: str = 'whatever',
                          n_channels: int = 3,
                          n_interfering_speakers: int = 3,
                          target_fs: int = 16000,
                          num_files: dict = {'train': -1,
                                             'val': -1,
                                             'test': -1},
                          reverb: bool = True,
                          target_angle: float = 0,
                          side_room: int = 20,
                          rt60_min=0.2,
                          rt60_max=0.8,
                          snr_min=-10,
                          snr_max=5
                          ):
    """
    Preparation of speaker mix dataset. The target speaker is placed in a fixed position relative to the microphone array. The interfering speakers are placed randomly with one speaker per angle segment.

    :param store_dir: path to directory in which to store the dataset
    :param post_fix: postfix to specify the characteristics of the dataset
    :param wsj0_path: path the the raw WSJ0 data
    :param n_channels: number of channels in the microphone array
    :param n_interfering_speakers: the number of interfering speakers
    :param target_fs: the target sampling rate for the dataset
    :param num_files: a dictionary specifying the number of examples per stage
    :param reverb: turn off reverberation if set to False
    :param rt60_min: min RT60 time (uniformly sampled if reverb)
    :param rt60_max: max RT60 time (uniformly sampled if reverb)
    :param snr_min: min SNR (uniformly sampled)
    :param snr_max: max SNR (uniformely sampled)
    :param side_room: angle of closest interfering source (default: 20 deg)
    :return:
    """
    prep_store_name = f"prep_mix{'_' + post_fix if post_fix else ''}.hdf5"

    train_samples = list(
        sorted(glob.glob(os.path.join(wsj0_path, 'si_tr_s/*/*.wav'))))
    val_samples = list(
        sorted(glob.glob(os.path.join(wsj0_path, 'si_dt_20/*/*.wav'))))
    test_samples = list(
        sorted(glob.glob(os.path.join(wsj0_path, 'si_et_05/*/*.wav'))))

    meta = {}
    with h5py.File(os.path.join(store_dir, prep_store_name), 'w') as prep_storage:
        for data_set, samples in (('train', train_samples),
                                  ('val', val_samples),
                                  ('test', test_samples)):
            if num_files[data_set] == 0:
                continue

            n_dataset_samples = num_files[data_set] if num_files[data_set] > 0 else len(samples)
            random.shuffle(samples)  # pick random speakers

            MAX_SAMPLES_PER_FILE = 12 * target_fs
            audio_dataset = prep_storage.create_dataset(data_set,
                                                        shape=(
                                                            n_dataset_samples, 3, n_channels, MAX_SAMPLES_PER_FILE),
                                                        chunks=(
                                                            1, 3, n_channels, MAX_SAMPLES_PER_FILE),
                                                        dtype=np.float32,
                                                        compression="gzip",
                                                        shuffle=True)

            set_meta = {}

            sproom = SPRoomSimulator(channels=n_channels, mode=data_set)

            for target_idx, target_path in enumerate(samples[:n_dataset_samples]):

                # select interfering speakers
                interfering_speakers = random.choices(
                    samples[:n_dataset_samples], k=n_interfering_speakers)
                

                reverb_target_signal, noise_signal, dry_target_signal, sample_meta = sproom.create_sample(
                    speaker_list=[target_path] + interfering_speakers,
                    seed2=target_idx,
                    reverb=reverb,
                    target_angle=target_angle,
                    side_room=side_room,
                    rt60_min=rt60_min,
                    rt60_max=rt60_max,
                    snr_min=snr_min,
                    snr_max=snr_max)
                n_audio_samples = min(
                    sample_meta['n_samples'], MAX_SAMPLES_PER_FILE)
                sample_meta['n_samples'] = n_audio_samples

                # store reverb clean
                audio_dataset[target_idx, 0, :,
                              :n_audio_samples] = reverb_target_signal[:, :n_audio_samples]
                audio_dataset[target_idx, 0,
                              :, n_audio_samples:MAX_SAMPLES_PER_FILE] = 0

                # store noise
                audio_dataset[target_idx, 1, :,
                              :n_audio_samples] = noise_signal[:, :n_audio_samples]
                audio_dataset[target_idx, 1,
                              :, n_audio_samples:MAX_SAMPLES_PER_FILE] = 0

                set_meta[target_idx] = sample_meta

                # store dry clean
                audio_dataset[target_idx, 2, :,
                              :n_audio_samples] = dry_target_signal[:, :n_audio_samples]
                audio_dataset[target_idx, 2,
                              :, n_audio_samples:MAX_SAMPLES_PER_FILE] = 0

                if target_idx % 10 == 0:
                    print(
                        f'{data_set}: {target_idx} of {n_dataset_samples}')

            meta[data_set] = set_meta

    with open(os.path.join(store_dir, f"prep_mix_meta{'_' + post_fix if post_fix else ''}.json"),
              'w') as prep_meta_storage:
        json.dump(meta, prep_meta_storage, indent=4)


if __name__ == '__main__':
    prep_speaker_mix_data(SIM_DATA_PATH,
                          'ch3_sp5_small',
                          WSJ0_PATH,
                          n_interfering_speakers=5,
                          n_channels=3,
                          num_files={'train': 6000, 'val': 1000, 'test': 600},
                          reverb=True,
                          target_angle=0,
                          rt60_min=0.2,
                          rt60_max=0.5,
                          snr_min=np.nan,
                          snr_max=5,
                          side_room=15)