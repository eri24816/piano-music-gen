import torch
from torch.utils.data import Dataset
from pianogen.data.pianoroll import PianoRoll
from pianogen.dataset.pianorolldataset import PianoRollDataset


def tokenize(
    pr: PianoRoll,
    n_velocity=128,
    duration: int | None = None,
    seq_len: int | None = None,
):
    tokens = []
    frame = 0
    if duration is None:
        duration = pr.duration

    tokens.append({"type": "start", "frame": frame})
    for note in pr.notes:
        while note.onset > frame:
            tokens.append({"type": "next_frame", "frame": frame})
            frame += 1

        tokens.append({"type": "pitch", "frame": frame, "value": note.pitch - 21})
        tokens.append(
            {
                "type": "velocity",
                "frame": frame,
                "value": int(note.velocity * (n_velocity / 128)),
            }
        )

    while duration > frame:
        tokens.append({"type": "next_frame", "frame": frame})
        frame += 1

    # fill in the next_frame
    for i in range(len(tokens) - 1):
        tokens[i]["next_frame"] = tokens[i + 1]["frame"]

    tokens.pop()  # remove the last next_frame 128

    # we're using seq_len+1 because start token doesn't count
    if seq_len is not None:
        tokens = tokens[: seq_len + 1]

        if len(tokens) < seq_len + 1:
            tokens += [{"type": "pad"}] * (seq_len + 1 - len(tokens))

    return tokens


def construct_input_frame(token: dict, pos_encoding: torch.Tensor, n_pitch, n_velocity):
    if token["type"] == "pad":
        return torch.zeros(n_pitch + n_velocity + 2 + pos_encoding.shape[1] * 2)

    # pitch
    pitch = torch.zeros(n_pitch)
    if token["type"] == "pitch":
        pitch[token["value"]] = 1

    # velocity
    velocity = torch.zeros(n_velocity)
    if token["type"] == "velocity":
        velocity[token["value"]] = 1

    # next_frame
    next_frame = torch.zeros(1)
    if token["type"] == "next_frame":
        next_frame[0] = 1

    # start
    start = torch.zeros(1)
    if token["type"] == "start":
        start[0] = 1

    # pos
    pos = pos_encoding[token["frame"]]

    # target pos
    target_pos = pos_encoding[token["next_frame"]]

    return torch.cat([pitch, velocity, next_frame, start, pos, target_pos], dim=0)


def construct_input_tensor(tokens, pos_encoding: torch.Tensor, n_pitch, n_velocity):
    frame_axis = []

    for token in tokens:
        frame_axis.append(
            construct_input_frame(token, pos_encoding, n_pitch, n_velocity)
        )

    return torch.stack(frame_axis, dim=0)


def construct_output_mask(tokens, n_pitch, n_velocity):
    """
    An additive mask for the model's output (logits) to prevent the model from predicting invalid tokens.

    The first token must be pitch or next_frame.
    The next token of pitch must be velocity.
    The next token of next_frame can be pitch or next_frame.
    The next token of velocity must be pitch or next_frame.

    Accroding to the above rule, we can construct a mask as a prior on the model's prediction.
    """

    mask = torch.zeros(len(tokens), n_pitch + n_velocity + 1)
    # fill with -inf
    mask = mask - 1e7

    mask[0, :n_pitch] = 0
    mask[0, n_pitch + n_velocity] = 0

    for i in range(len(tokens) - 1):
        # output shape: Output: [pitch(n_pitch), velocity(n_velocity), next_frame(1)]
        token = tokens[i]

        if token["type"] == "pitch":
            # enable velocity
            mask[i + 1, n_pitch : n_pitch + n_velocity] = 0
        if token["type"] == "velocity":
            # enable pitch or next_frame
            mask[i + 1, :n_pitch] = 0
            mask[i + 1, n_pitch + n_velocity] = 0
        if token["type"] == "next_frame":
            # enable pitch or next_frame
            mask[i + 1, :n_pitch] = 0
            mask[i + 1, n_pitch + n_velocity] = 0

    return mask


def construct_target(tokens, n_pitch, n_velocity):
    res = []
    for i, token in enumerate(tokens):
        if token["type"] == "pitch":
            res.append(token["value"])
        elif token["type"] == "velocity":
            res.append(n_pitch + token["value"])
        elif token["type"] == "next_frame":
            res.append(n_pitch + n_velocity)
        elif token["type"] == "pad":
            res.append(-100)  # -100 is the ignore index
        else:
            raise ValueError(f"Unknown token type: {token['type']}")

    return torch.tensor(res, dtype=torch.long)


class TokenizedPianoRollDataset(Dataset):
    """
    Input: [pitch(n_pitch), velocity(n_velocity), next_frame(1), start(1), pos, target_pos]
    Output: [pitch(n_pitch), velocity(n_velocity), next_frame(1)]
    """

    def __init__(
        self,
        path: str,
        pos_encoding: torch.Tensor,
        segment_length: int,
        hop_len: int,
        seq_len: int,
        n_pitch: int,
        n_velocity: int,
    ):
        self.ds = PianoRollDataset(path, segment_len=segment_length, hop_len=hop_len)
        self.pos_encoding = pos_encoding
        self.seq_len = seq_len
        self.n_pitch = n_pitch
        self.n_velocity = n_velocity
        self.segment_length = segment_length

        # self.tokens = []
        # for idx in range(len(self.ds)):
        #    self.tokens.append(tokenize(self.ds.get_piano_roll(idx), n_velocity=self.n_velocity, duration=self.segment_length, seq_len=self.seq_len))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # tokens = self.tokens[idx]

        pr = self.ds.get_piano_roll(idx)
        tokens = tokenize(
            pr,
            n_velocity=self.n_velocity,
            duration=self.segment_length,
            seq_len=self.seq_len,
        )

        tokens_without_start = tokens[1:]

        # the last token is not needed to be an input
        input = construct_input_tensor(
            tokens[:-1],
            pos_encoding=self.pos_encoding,
            n_pitch=self.n_pitch,
            n_velocity=self.n_velocity,
        )
        target = construct_target(
            tokens_without_start, n_pitch=self.n_pitch, n_velocity=self.n_velocity
        )
        output_mask = construct_output_mask(
            tokens_without_start, n_pitch=self.n_pitch, n_velocity=self.n_velocity
        )
        return {"input": input, "target": target, "output_mask": output_mask}

    def get_loss_weight(self):
        """
        The loss weight for each token.
        """
        res = torch.ones(self.n_pitch + self.n_velocity + 1)
        res[self.n_pitch + self.n_velocity] = (
            0.05  # next_frame is too common so we need to reduce its weight
        )
