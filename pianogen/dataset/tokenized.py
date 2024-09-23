from torch.utils.data import Dataset
from pianogen.dataset.pianorolldataset import PianoRollDataset, Sample
from pianogen.tokenizer import PianoRollTokenizer

class TokenizedPianoRollDataset(Dataset):
    """
    indices: [L]
    pos: [L]
    features: [L, D]
    """

    def __init__(
        self,
        piano_roll_dataset: PianoRollDataset,
        tokenizer: PianoRollTokenizer,
    ):
        self.ds = piano_roll_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ds)

    def collect_features(self, sample: Sample):
        song = sample.song
        features = [
            song.read_json("chords")[sample.start // 16 : sample.end // 16],
            song.read_json("note_density")[sample.start // 32 : sample.end // 32],
            song.read_json("polyphony")[sample.start // 32 : sample.end // 32],
            song.read_json("highest_pitch")[sample.start // 32 : sample.end // 32],
            song.read_json("lowest_pitch")[sample.start // 32 : sample.end // 32],
        ]

    def __getitem__(self, idx):
        pr = self.ds.get_piano_roll(idx)
        tokens = self.tokenizer.tokenize(pr)

        indices = self.tokenizer.vocab.tokens_to_indices(tokens)

        pos = self.tokenizer.get_frame_indices(tokens)

        output_mask = self.tokenizer.get_output_mask(tokens[:-1])

        sample = self.ds.get_sample(idx)
        features = self.collect_features(sample)

        return {
            "indices": indices,
            "pos": pos,
            "features": features,
            "output_mask": output_mask,
        }
