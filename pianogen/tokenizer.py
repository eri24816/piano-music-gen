import torch
from pianogen.data.pianoroll import Note, PianoRoll
from pianogen.data.vocab import Vocabulary, WordArray


class PianoRollTokenizer:
    def __init__(
        self,
        n_pitch,
        n_velocity,
        duration=None,
        token_seq_len=None,
        token_seq_len_per_bar=None,
    ):
        self.n_pitch = n_pitch
        self.n_velocity = n_velocity
        self.duration = duration
        self.token_seq_len = token_seq_len
        self.token_seq_len_per_bar = token_seq_len_per_bar
        self.vocab = Vocabulary(
            [
                "pad",
                "start",
                "end",
                "next_frame",
                WordArray("pitch", {"value": range(n_pitch)}),
                WordArray("velocity", {"value": range(n_velocity)}),
            ]
        )

    def tokenize(self, pr: PianoRoll) -> list[dict]:
        """
        Convert a PianoRoll to a list of tokens.
        """
        return tokenize(
            pr,
            n_velocity=self.n_velocity,
            duration=self.duration,
            seq_len=self.token_seq_len,
            seq_len_per_bar=self.token_seq_len_per_bar,
        )

    def token_to_idx(self, token: dict) -> int:
        return self.vocab.get_idx(token)

    def idx_to_token(self, idx: int) -> dict:
        return self.vocab.get_token(idx)

    def __getitem__(self, token: int | dict) -> dict | int:
        return self.vocab[token]

    def get_frame_indices(self, tokens: list[dict], shift=0):
        frame = shift
        result = []
        for token in tokens:
            result.append(frame)
            if token["type"] == "next_frame":
                frame += 1
        return torch.tensor(result, dtype=torch.long)

    def get_output_mask(self, tokens: list[dict]) -> torch.Tensor:
        return get_output_mask(self.vocab, tokens)

    def sample_from_logits(self, logits: torch.Tensor, last_token: dict) -> dict:
        # apply output mask
        mask = self.get_output_mask([last_token])
        mask = mask.unsqueeze(0).expand_as(logits)
        logits = logits.masked_fill(mask == 0, -1e9)
        idx = sample_from_top_k(logits, 15).item()
        assert isinstance(idx, int)
        return self.idx_to_token(idx)


def sample_from_top_k(logits: torch.Tensor, k):
    values, indices = logits.topk(k)
    probs = torch.softmax(values, dim=0)
    selected = torch.multinomial(probs, 1)
    return indices[selected]


def tokenize(
    pr: PianoRoll,
    n_velocity,
    duration: int | None = None,
    seq_len: int | None = None,
    seq_len_per_bar: int | None = None,
):
    bar_len = 32

    tokens = []

    tokens.append({"type": "start"})

    if seq_len_per_bar is None:
        if seq_len is not None:
            seq_len = seq_len - 2  # start and end tokens
        tokens += tokenize_raw(pr.notes, n_velocity, pr.duration, duration, seq_len)
        tokens.append({"type": "end"})

    else:
        for bar in pr.iter_over_bars(bar_len):
            tokens += tokenize_raw(
                bar,
                n_velocity,
                pr_duration=bar_len,
                duration=bar_len,
                seq_len=seq_len_per_bar,
            )

        if seq_len is not None:
            if len(tokens) < seq_len:
                tokens.append({"type": "end"})
                tokens += [{"type": "pad"}] * (seq_len + 1 - len(tokens))

    return tokens


def tokenize_raw(
    notes: list[Note],
    n_velocity,
    pr_duration: int,
    duration: int | None = None,
    seq_len: int | None = None,
):
    tokens = []
    frame = 0
    if duration is None:
        duration = pr_duration

    for note in notes:
        while note.onset > frame:
            tokens.append({"type": "next_frame"})
            frame += 1

        tokens.append({"type": "pitch", "value": note.pitch - 21})
        tokens.append(
            {
                "type": "velocity",
                "value": int(note.velocity * (n_velocity / 128)),
            }
        )

    while duration > frame:
        tokens.append({"type": "next_frame"})
        frame += 1

    tokens.pop()  # remove the last next_frame 128

    if seq_len is not None:
        tokens = tokens[:seq_len]

        if len(tokens) < seq_len:
            tokens += [{"type": "pad"}] * (seq_len - len(tokens))

    return tokens


def get_output_mask(vocab: Vocabulary, tokens: list[dict]) -> torch.Tensor:
    """
    An additive mask for the model's output (logits) to prevent the model from predicting invalid tokens.

    The first token must be pitch or next_frame.
    The next token of pitch must be velocity.
    The next token of next_frame can be pitch or next_frame.
    The next token of velocity must be pitch or next_frame.

    Accroding to the above rule, we can construct a mask as a prior on the model's prediction.
    """

    mask_token_types = []

    for i in range(len(tokens)):
        # output shape: Output: [pitch(n_pitch), velocity(n_velocity), next_frame(1)]
        token = tokens[i]

        if token["type"] in ["start", "velocity", "next_frame", "pad"]:
            # enable pitch or next_frame
            mask_token_types.append(["pitch", "next_frame"])
        elif token["type"] == "pitch":
            # enable velocity
            mask_token_types.append("velocity")
        elif token["type"] == "end":
            # end token
            mask_token_types.append([])
        else:
            raise ValueError(f"Invalid token type: {token['type']}")

    mask = vocab.get_mask(mask_token_types)

    return mask