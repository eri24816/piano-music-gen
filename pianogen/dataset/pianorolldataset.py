from math import ceil
from pathlib import Path
from typing import Any
from music_data_analysis import Song
import torch
from torch.utils.data import Dataset as TorchDataset
from music_data_analysis.data_access import Dataset as MusicDataset
from music_data_analysis.data.pianoroll import Pianoroll
import multiprocessing as mp
import numpy as np

class Sample:
    def __init__(self, song: Song, start: int, end: int):
        self.song = song
        self.start = start
        self.end = end
        self.duration = end - start

    def get_feature_slice(self, file_name: str, granularity: int, pad_to: int = 0, pad_value: Any = 0):
        j = self.song.read_json(file_name)
        if isinstance(j, list):
            unpadded = j[self.start // granularity : self.end // granularity]
            if pad_to:
                return unpadded + [pad_value] * (pad_to - len(unpadded))
            else:
                return unpadded
        else:
            assert isinstance(j, dict)
            result = {}
            for k, v in j.items():
                result[k] = v[self.start // granularity : self.end // granularity]
                if pad_to:
                    result[k] += [pad_value] * (pad_to - len(result[k]))
            return result

class PianoRollDataset(TorchDataset):
    def __init__(
        self,
        data_dir,
        segment_len=0,
        hop_len=32,
        max_duration=32 * 180,
    ):
        self.mds = MusicDataset(Path(data_dir))
        self.segment_length = segment_len

        # flattened data structure for the samples for faster pickling
        self.samples_song = []
        self.samples_start = []
        self.samples_end = []

        if segment_len:
            for song in self.mds.songs():
                duration = song.read_json("duration")
                num_seg = ceil(duration / hop_len)
                self.samples_song += [song] * num_seg
                self.samples_start += [hop_len * i for i in range(num_seg)]
                self.samples_end += [hop_len * i + segment_len for i in range(num_seg)]

        else:
            self.max_duration = max_duration
            # do not include songs longer than max_duration
            for song in self.mds.songs():
                duration = song.read_json("duration")
                if duration <= max_duration:
                    self.samples_song.append(song)
                    self.samples_start.append(0)
                    self.samples_end.append(duration)

        self.length = len(self.samples_song)

        # use shared memory for the samples. Otherwise, the pickling of these arrays would be very slow
        self.samples_start = mp.Array('i', self.samples_start)
        self.samples_start = np.ctypeslib.as_array(self.samples_start.get_obj())
        self.samples_start = torch.from_numpy(self.samples_start)
        self.samples_end = mp.Array('i', self.samples_end)
        self.samples_end = np.ctypeslib.as_array(self.samples_end.get_obj())
        self.samples_end = torch.from_numpy(self.samples_end)

        print(f"Loaded {self.length} samples from {len(self.mds)} songs")

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.get_sample(idx)
        if self.segment_length:
            return sample.song.read_pianoroll("pianoroll").to_tensor(
                sample.start, sample.end, padding=True, normalized=False
            )
        else:
            return sample.song.read_pianoroll("pianoroll").to_tensor(
                0, self.max_duration, padding=True, normalized=False
            )

    def get_piano_roll(self, idx: int) -> Pianoroll:
        sample = self.get_sample(idx)
        if self.segment_length:
            return sample.song.read_pianoroll("pianoroll").slice(
                sample.start, sample.end
            )
        else:
            return sample.song.read_pianoroll("pianoroll").slice(0, self.max_duration)

    def get_sample(self, idx: int) -> Sample:
        return Sample(
            self.samples_song[idx], self.samples_start[idx], self.samples_end[idx]
        )

    def get_all_piano_rolls(self) -> list[Pianoroll]:
        return [self.get_piano_roll(i) for i in range(len(self))]