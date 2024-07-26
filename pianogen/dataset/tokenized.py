from torch.utils.data import Dataset
from pianogen.dataset.pianorolldataset import PianoRollDataset
from pianogen.tokenizer import PianoRollTokenizer


class TokenizedPianoRollDataset(Dataset):
    """
    Input: [pitch(n_pitch), velocity(n_velocity), next_frame(1), start(1), pos, target_pos]
    Output: [pitch(n_pitch), velocity(n_velocity), next_frame(1)]
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

    def __getitem__(self, idx):
        pr = self.ds.get_piano_roll(idx)
        tokens = self.tokenizer.tokenize(pr)

        indices = self.tokenizer.vocab.tokens_to_indices(tokens)

        pos = self.tokenizer.get_frame_indices(tokens)

        output_mask = self.tokenizer.get_output_mask(tokens[:-1])

        return {"indices": indices, "pos": pos, "output_mask": output_mask}
