from pathlib import Path
import shutil
from tempfile import tempdir
import tempfile
from typing import Dict
import music_data_analysis
from torch.utils.data import Dataset
from pianogen.dataset.pianorolldataset import PianoRollDataset, Sample
from pianogen.model.with_feature import FeatureLoader

class FeatureDataset(Dataset):
    """
    indices: [L]
    pos: [L]
    features: [L, D]
    """

    @staticmethod
    def from_midi(midi_path: str|Path, loaders: Dict[str, FeatureLoader], sync: bool = False, segment_len: int = 0, hop_len: int = 32, max_duration: int = 32*180):
        '''From a midi file or a directory of midi files, create a FeatureDataset.
        This function will run music_data_analysis on the midi files to create a dataset of pianorolls.
        The dataset will be stored in a temporary directory.

        Args:
            midi_path (str | Path): A path to a midi file or a directory containing midi files
            loaders (Dict[str, FeatureLoader]): A dictionary of FeatureLoader objects
        '''
        dataset_path = music_data_analysis.run(midi_path, verbose=False).dataset_path
        return FeatureDataset(dataset_path, loaders, segment_len, hop_len, max_duration)

    @staticmethod
    def from_piano_roll(piano_roll: music_data_analysis.Pianoroll, loaders: Dict[str, FeatureLoader], segment_len: int = 0, hop_len: int = 32, max_duration: int|None = None):
        '''From a Pianoroll object, create a FeatureDataset.

        Args:
            piano_roll (Pianoroll): A Pianoroll object
            loaders (Dict[str, FeatureLoader]): A dictionary of FeatureLoader objects
        '''
        midi_path = Path(tempfile.mktemp(suffix='.mid'))
        piano_roll.to_midi(midi_path)
        dataset_path = music_data_analysis.run(midi_path, verbose=False).dataset_path
        midi_path.unlink()
        if max_duration is None:
            max_duration = piano_roll.duration
        return FeatureDataset(dataset_path, loaders, segment_len, hop_len, max_duration)

    def __init__(
        self,
        path: str|Path,
        loaders: Dict[str, FeatureLoader],
        segment_len=0,
        hop_len=32,
        max_duration=32 * 180,
    ):
        self.ds = PianoRollDataset(path, segment_len, hop_len, max_duration)
        self.path = Path(path)
        self.loaders = loaders
        self.segment_len = segment_len

    def __len__(self):
        return len(self.ds)

    def collect_features(self, sample: Sample, pad_to: int):
        features = {}
        for name, loader in self.loaders.items():
            features[name] = loader.load(sample, pad_to)
        return features
        
    def __getitem__(self, idx):
        sample = self.ds.get_sample(idx)
        pad_to = self.ds.max_duration if self.segment_len == 0 else self.segment_len
        features = self.collect_features(sample, pad_to = pad_to)
        return features
    
    def get_human_representation(self, idx, feature_names: None|list[str] = None):
        pad_to = self.ds.max_duration if self.segment_len == 0 else self.segment_len
        if feature_names is None:
            feature_names = list(self.loaders.keys())
        sample = self.ds.get_sample(idx)
        return {name: loader.to_human_representation(loader.load(sample,pad_to)) for name, loader in self.loaders.items() if name in feature_names}

    def delete(self):
        '''Delete the dataset from disk. Only works if the dataset is in a temporary directory.'''
        
        if Path(tempdir) not in self.path.parents:
            raise ValueError(f"Cannot delete dataset at {self.path}. It is not in a temporary directory. {tempdir} not in {self.path.parents}")

        shutil.rmtree(self.path)