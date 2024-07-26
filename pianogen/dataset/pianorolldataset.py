import json
from math import ceil
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset

from pianogen.data.pianoroll import PianoRoll


class AttributeManager:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.loaded_attributes = {}

    def get_attribute_file(self, attribute_name):
        if attribute_name in self.loaded_attributes:
            return self.loaded_attributes[attribute_name]
        else:
            file_path = os.path.join(self.data_dir, f"{attribute_name}.json")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Attribute file {file_path} not found")
            with open(file_path, "r") as f:
                self.loaded_attributes[attribute_name] = json.load(f)
            return self.loaded_attributes[attribute_name]

    def get_attribute(self, attribute_name, song_idx):
        attribute_file = self.get_attribute_file(attribute_name)
        return attribute_file[song_idx]


class PianoRollDataset(Dataset):
    def __init__(
        self,
        data_dir,
        segment_len=0,
        hop_len=32,
        max_duration=32 * 180,
        max_pieces=None,
    ):
        self.segment_length = segment_len
        print(f"Creating dataset segment_len = {segment_len}")

        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        self.data_dir = data_dir

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} not found")

        self.attr = AttributeManager(data_dir / "attr")

        file_list = list((data_dir / "pianoroll").glob("*.json"))
        file_list = file_list[:max_pieces]

        self.sample_to_song: list[tuple[int, int, int]] = []  # song_idx, start, end
        if segment_len:
            num_segments = [
                ceil(duration / hop_len)
                for duration in self.attr.get_attribute_file("duration")
            ]

            for pr_idx, num_seg in enumerate(num_segments):
                self.sample_to_song += [
                    (pr_idx, hop_len * i, hop_len * i + segment_len)
                    for i in range(num_seg)
                ]
        else:
            self.max_duration = max_duration
            # do not include songs longer than max_duration
            for i in range(len(file_list)):
                duration = self.attr.get_attribute("duration", i)
                if duration <= max_duration:
                    self.sample_to_song.append((i, 0, duration))

        self.length = len(self.sample_to_song)

        print(f"Created dataset with {self.length} samples from {len(file_list)} songs")

    def __len__(self):
        return self.length

    def __getitem__(self, idx) -> torch.Tensor:
        if self.segment_length:
            song_idx, start, end = self.sample_to_song[idx]
            return PianoRoll.load(
                self.data_dir / "pianoroll" / f"{song_idx}.json"
            ).to_tensor(start, end, padding=True, normalized=False)
        else:
            return PianoRoll.load(
                self.data_dir / "pianoroll" / f"{idx}.json"
            ).to_tensor(0, self.max_duration, padding=True, normalized=False)

    def get_piano_roll(self, idx) -> PianoRoll:
        if self.segment_length:
            piece, start, end = self.sample_to_song[idx]
            return PianoRoll.load(self.data_dir / "pianoroll" / f"{piece}.json").slice(
                start, end
            )
        else:
            return PianoRoll.load(self.data_dir / "pianoroll" / f"{idx}.json").slice(
                0, self.max_duration
            )

    def get_all_piano_rolls(self) -> list[PianoRoll]:
        return [self.get_piano_roll(i) for i in range(len(self))]
