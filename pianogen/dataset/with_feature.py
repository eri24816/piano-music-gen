from typing import Dict
from torch.utils.data import Dataset
from pianogen.dataset.pianorolldataset import PianoRollDataset, Sample
from pianogen.model.with_feature import Feature

class FeatureDataset(Dataset):
    """
    indices: [L]
    pos: [L]
    features: [L, D]
    """

    def __init__(
        self,
        piano_roll_dataset: PianoRollDataset,
        features: Dict[str, Feature],
    ):
        self.ds = piano_roll_dataset
        self.features = features

    def __len__(self):
        return len(self.ds)

    def collect_features(self, sample: Sample, pad_to: int):
        features = {}
        for name, feature in self.features.items():
            features[name] = feature.load_for_dataset(sample, pad_to)
        return features
        
    def __getitem__(self, idx):
        sample = self.ds.get_sample(idx)
        features = self.collect_features(sample, pad_to = self.ds.max_duration)
        return features

