from pathlib import Path
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
        dataset_path = music_data_analysis.run(midi_path).dataset_path
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
        self.loaders = loaders

    def __len__(self):
        return len(self.ds)

    def collect_features(self, sample: Sample, pad_to: int):
        features = {}
        for name, loader in self.loaders.items():
            features[name] = loader.load(sample, pad_to)
        return features
        
    def __getitem__(self, idx):
        sample = self.ds.get_sample(idx)
        features = self.collect_features(sample, pad_to = self.ds.max_duration)
        return features

