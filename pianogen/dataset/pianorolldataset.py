from math import ceil
from pathlib import Path
from music_data_analysis import Song
import torch
from torch.utils.data import Dataset as TorchDataset
from music_data_analysis.data_access import Dataset as MusicDataset
from music_data_analysis.data.pianoroll import PianoRoll


class Sample:
    def __init__(self, song: Song, start: int, end: int):
        self.song = song
        self.start = start
        self.end = end
        self.duration = end - start


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
        self.samples: list[Sample] = []
        if segment_len:
            for song in self.mds.songs():
                duration = song.read_json("duration")
                num_seg = ceil(duration / hop_len)
                self.samples += [
                    Sample(song, hop_len * i, hop_len * i + segment_len)
                    for i in range(num_seg)
                ]

        else:
            self.max_duration = max_duration
            # do not include songs longer than max_duration
            for song in self.mds.songs():
                duration = song.read_json("duration")
                if duration <= max_duration:
                    self.samples.append(Sample(song, 0, duration))

        self.length = len(self.samples)

        print(f"Loaded {self.length} samples from {len(self.mds)} songs")

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.samples[idx]
        if self.segment_length:
            return sample.song.read_pianoroll("pianoroll").to_tensor(
                sample.start, sample.end, padding=True, normalized=False
            )
        else:
            return sample.song.read_pianoroll("pianoroll").to_tensor(
                0, self.max_duration, padding=True, normalized=False
            )

    def get_piano_roll(self, idx: int) -> PianoRoll:
        sample = self.samples[idx]
        if self.segment_length:
            return sample.song.read_pianoroll("pianoroll").slice(
                sample.start, sample.end
            )
        else:
            return sample.song.read_pianoroll("pianoroll").slice(0, self.max_duration)

    def get_sample(self, idx: int) -> Sample:
        return self.samples[idx]

    def get_all_piano_rolls(self) -> list[PianoRoll]:
        return [self.get_piano_roll(i) for i in range(len(self))]