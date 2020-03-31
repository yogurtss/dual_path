import musdb
import soundfile
import os
import librosa as lib
from tqdm import tqdm
import numpy as np
import torch
from sortedcontainers import SortedList
import h5py
import torch.nn as nn
from dataset.util import *


def getMUSDB(database_path):
    # 导入数据
    mus = musdb.DB(root=database_path, is_wav=False)

    subsets = list()
    for subset in ["train", "test"]:
        tracks = mus.load_mus_tracks(subset)
        samples = list()

        # Go through tracks
        for track in tracks:
            # Skip track if mixture is already written, assuming this track is done already
            # track_path = track.path[:-4]
            track_path = SAVE_PATH + subset + '/' + track.name
            if not os.path.exists(track_path):
                os.mkdir(track_path)
            mix_path = track_path + "/mix.wav"
            acc_path = track_path + "/accompaniment.wav"
            if os.path.exists(mix_path):
                print("WARNING: Skipping track " + mix_path + " since it exists already")

                # Add paths and then skip
                paths = {"mix": mix_path, "accompaniment": acc_path}
                paths.update({key: track_path + "_" + key + ".wav" for key in ["bass", "drums", "other", "vocals"]})

                samples.append(paths)

                continue

            rate = track.rate

            # Go through each instrument
            paths = dict()
            stem_audio = dict()
            for stem in ["bass", "drums", "other", "vocals"]:
                path = track_path + '/' + stem + ".wav"
                audio = track.targets[stem].audio.T
                soundfile.write(path, audio, rate, "PCM_16")
                stem_audio[stem] = audio
                paths[stem] = path

            # Add other instruments to form accompaniment
            acc_audio = np.clip(sum([stem_audio[key] for key in list(stem_audio.keys()) if key != "vocals"]), -1.0, 1.0)
            soundfile.write(acc_path, acc_audio, rate, "PCM_16")
            paths["accompaniment"] = acc_path

            # Create mixture
            mix_audio = track.audio.T
            soundfile.write(mix_path, mix_audio, rate, "PCM_16")
            paths["mix"] = mix_path

            diff_signal = np.abs(mix_audio - acc_audio - stem_audio["vocals"])
            print("Maximum absolute deviation from source additivity constraint: " + str(
                np.max(diff_signal)))  # Check if acc+vocals=mix
            print("Mean absolute deviation from source additivity constraint:    " + str(np.mean(diff_signal)))

            samples.append(paths)

        subsets.append(samples)

    train_val_list = subsets[0]
    test_list = subsets[1]

    np.random.seed(42)
    train_list = np.random.choice(train_val_list, 75, replace=False)
    val_list = [elem for elem in train_val_list if elem not in train_list]
    dataset = {'train': train_list,
               'val': val_list,
               'test': test_list}
    return dataset


class AudioDataset(nn.Module):
    def __init__(self, partition, instruments, sr, channels, out_channels, random_hops, hdf_dir, shapes, audio_transform=None, in_memory=False):
        super(AudioDataset, self).__init__()
        self.hdf_dir = os.path.join(hdf_dir, partition + ".hdf5")
        self.random_hops = random_hops
        self.sr = sr
        self.channels = channels
        self.audio_transform = audio_transform
        self.in_memory = in_memory
        self.instruments = instruments
        self.shapes = shapes
        self.out_channels = out_channels

        print('Preparing {} dataset...'.format(partition))

        # Go through HDF and collect lengths of all audio files
        with h5py.File(self.hdf_dir, "r") as f:
            lengths = [f[str(song_idx)].attrs["target_length"] for song_idx in range(len(f))]

            # Subtract input_size from lengths and divide by hop size to determine number of starting positions
            lengths = [(l // self.shapes['length']) + 1 for l in lengths]

        self.start_pos = SortedList(np.cumsum(lengths))
        self.length = self.start_pos[-1]
        self.dataset = h5py.File(self.hdf_dir, 'r', driver="core")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Find out which slice of targets we want to read
        audio_idx = self.start_pos.bisect_right(idx)

        if audio_idx > 0:
            idx = idx - self.start_pos[audio_idx - 1]

        # Check length of audio signal
        audio_length = self.dataset[str(audio_idx)].attrs["length"]
        target_length = self.dataset[str(audio_idx)].attrs["target_length"]

        # Determine position where to start targets
        if self.random_hops:
            start_target_pos = np.random.randint(0, max(target_length - self.shapes['length'] + 1, 1))
        else:
            # Map item index to sample position within song
            start_target_pos = idx * self.shapes['length']
        start_pos = start_target_pos
        if start_pos < 0:
            # Pad manually since audio signal was too short
            pad_front = abs(start_pos)
            start_pos = 0
        else:
            pad_front = 0
        end_pos = start_target_pos + self.shapes['length']
        if end_pos > audio_length:
            # Pad manually since audio signal was too short
            pad_back = end_pos - audio_length
            end_pos = audio_length
        else:
            pad_back = 0

        # Read and return
        audio = self.dataset[str(audio_idx)]["inputs"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)
        targets = self.dataset[str(audio_idx)]["targets"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            targets = np.pad(targets, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        sources = {}
        for id, inst in enumerate(self.instruments.keys()):
            sources[inst] = targets[id * self.channels:(id + 1) * self.channels]
        del targets

        if hasattr(self, "audio_transform") and self.audio_transform is not None:
            audio, sources = self.audio_transform(audio, sources)
        idx_temp = 0
        targets = np.zeros([self.out_channels, self.shapes['length']], dtype=np.float32)
        if self.out_channels == 1:
            targets = sources['accompaniment']
        else:
            for k in sources.keys():
                if k == 'other':
                    continue
                targets[idx_temp] = sources[k]
                idx_temp += 1
        return torch.tensor(audio).squeeze(), torch.tensor(targets)


if __name__ == '__main__':
    partition = 'train'
    INSTRUMENTS = {"bass": True,
                   "drums": True,
                   "other": True,
                   "vocals": True,
                   "accompaniment": True}
    shapes = {'start_frame': 6140,
              'end_frame': 51201,
              'output_len': 45061,
              'input_len': 57341}
    sr = 22050
    channels = 1

    augment_func = lambda mix, targets: random_amplify(mix, targets, 0.7, 1.0, shapes)
    crop_func = lambda mix, targets: crop(mix, targets, shapes)
    dataset = AudioDataset(partition, INSTRUMENTS, sr, channels, 2, True, hdf_dir='../H5/', shapes=shapes, audio_transform=augment_func)
    dataset[0]
    dataset[1]
    print('test')












