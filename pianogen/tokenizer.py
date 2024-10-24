from typing import Literal
import torch
from music_data_analysis.data import Note, Pianoroll
from pianogen.data.vocab import Vocabulary, WordArray


class PianoRollTokenizer:
    def __init__(
        self,
        n_pitch,
        n_velocity,
        token_seq_len=None,
    ):
        self.n_pitch = n_pitch
        self.n_velocity = n_velocity
        self.token_seq_len = token_seq_len
        self.vocab = Vocabulary(
            [
                "pad",
                #"start",
                #"end",
                "next_frame",
                WordArray("pitch", {"type":["pitch"],"value": range(n_pitch)}),
                WordArray("velocity", {"type":["velocity"],"value": range(n_velocity)}),
            ]
        )

    def tokenize(self, pr: Pianoroll, pad: bool = True, token_seq_len: int | None = None, token_per_bar: int | None = None, need_end_token: bool = False
                 ) -> list[dict]:
        """
        Convert a Pianoroll to a list of tokens.
        """
        if token_seq_len is not None:
            if pad:
                token_seq_len = self.token_seq_len
        return tokenize(
            pr,
            n_velocity=self.n_velocity,
            seq_len=self.token_seq_len,
            token_per_bar=token_per_bar,
            need_end_token=need_end_token,
        ) 

    def detokenize(self, tokens: list[dict]) -> Pianoroll:
        """
        Convert a list of tokens to a Pianoroll.
        """
        return detokenize(tokens, self.n_velocity)

    def token_to_idx(self, token: dict) -> int:
        return self.vocab.get_idx(token)

    def idx_to_token(self, idx: int) -> dict:
        return self.vocab.get_token(idx)
    
    def token_to_idx_seq(self, tokens: list[dict]) -> list[int]:
        return [self.token_to_idx(token) for token in tokens]
    
    def idx_to_token_seq(self, indices: list[int]) -> list[dict]:
        return [self.idx_to_token(idx) for idx in indices]
    
    def pr_to_idx(self, pr: Pianoroll, return_pos: bool = False):
        tokens = self.tokenize(pr, pad=False)
        if return_pos:
            return self.token_to_idx_seq(tokens), self.get_frame_indices(tokens)
        else:
            return self.token_to_idx_seq(tokens)
    
    def idx_to_pr(self, indices: list[int], pos: list[int]) -> Pianoroll:
        tokens = self.idx_to_token_seq(indices)
        return detokenize(tokens, self.n_velocity)

    def __getitem__(self, token: int | dict) -> dict | int:
        return self.vocab[token]

    def get_frame_indices(self, tokens: list[dict], shift=0, infer_next_frame=False):
        """
        input: L
        output:
            if infer_next_frame: L + 1
            else: L
        """
        frame = shift
        result = []
        for token in tokens:
            result.append(frame)
            if token == "next_frame":
                frame += 1
        if infer_next_frame:
            result.append(frame)
        return torch.tensor(result, dtype=torch.long)
    
    def get_pitch_sequence(self, tokens: list[dict]) -> torch.Tensor:
        result = []
        current_pitch = 0
        for token in tokens:
            if isinstance(token, dict) and token["type"] == "pitch":
                current_pitch = token["value"]
            result.append(current_pitch)
        return torch.tensor(result, dtype=torch.long)

    def get_output_mask(self, tokens: list[str|dict|None]) -> torch.Tensor:
        return get_output_mask(self.vocab, tokens)

    def sample_from_logits(self, logits: torch.Tensor, last_token: str|dict|None, top_k: int = 15, p: float = 0.9, method: Literal["top_k", "nucleus"] = "nucleus") -> dict:
        # apply output mask
        mask = self.get_output_mask([last_token]).squeeze(0)
        mask = mask.expand_as(logits)
        logits += mask
        if method == "top_k":
            idx = top_k_sampling(logits, top_k).item()
        elif method == "nucleus":
            idx = nucleus_sampling(logits, p).item()
        else:
            raise ValueError(f"Invalid sampling method: {method}")
        assert isinstance(idx, int)
        return self.idx_to_token(idx)


def top_k_sampling(logits: torch.Tensor, k):
    values, indices = logits.topk(k)
    probs = torch.softmax(values, dim=0)
    selected = torch.multinomial(probs, 1)
    return indices[selected]

def nucleus_sampling(logits: torch.Tensor, p: float):
    probs = torch.softmax(logits, dim=0)
    sorted_probs, sorted_indices = torch.sort(probs, dim=0, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    selected_indices = []
    selected_probs = []
    for i in range(len(sorted_probs)):
        selected_indices.append(sorted_indices[i])
        selected_probs.append(sorted_probs[i])
        if cumulative_probs[i] > p:
            break
    # sample from selected_indices
    # normalize selected_probs
    selected_probs = torch.tensor(selected_probs)
    selected_probs = selected_probs / torch.sum(selected_probs)
    selected = torch.multinomial(selected_probs, 1)
    return selected_indices[selected]

def tokenize(
    pr: Pianoroll,
    n_velocity,
    seq_len: int | None = None,
    token_per_bar: int | None = None,
    need_end_token: bool = False,
):
    
    if token_per_bar is not None:
        tokens = []
        for bar in pr.iter_over_bars_pr():
            tokens += tokenize(bar, n_velocity, seq_len = token_per_bar, need_end_token=False)

        if need_end_token:
            for i in range(len(tokens) - 1, -1, -1):
                if tokens[i]!= "pad":
                    tokens[i] = "end"
                    break
            else:
                if len(tokens) > 0:
                    tokens[0] = "end"

        if seq_len is not None:
            tokens += ["pad" for _ in range(seq_len - len(tokens))]

        return tokens

    tokens = []

    tokens.append("start")

    # fill tokens with notes
    frame = 0
    
    for note in pr.notes:
        while note.onset > frame:
            tokens.append("next_frame")
            frame += 1
    
        tokens.append({"type": "pitch", "value": note.pitch - 21})
        tokens.append(
            {
                "type": "velocity",
                "value": int(note.velocity * (n_velocity / 128)),
            }
        )

    for _ in range(pr.duration - frame):
        tokens.append("next_frame")


    if need_end_token:
        tokens.append("end")
    
    if seq_len is not None:
        tokens = tokens[:seq_len]
    
        if len(tokens) < seq_len:
            tokens += ["pad"] * (seq_len - len(tokens))

    return tokens


def get_output_mask(vocab: Vocabulary, tokens: list[str|dict|None]) -> torch.Tensor:
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

        if token in ["start", "next_frame", None] or (isinstance(token, dict) and token["type"] == "velocity"):
            # enable pitch or next_frame
            mask_token_types.append(["pitch", "next_frame"])
        elif isinstance(token, dict) and token["type"] == "pitch":
            # enable velocity
            mask_token_types.append("velocity")
        elif token in ['pad', 'end']:
            # end token
            mask_token_types.append([])  # tokens after end are ignored
        else:
            raise ValueError(f"Invalid token type: {token}")

    mask = vocab.get_mask(mask_token_types)

    return mask


def detokenize(tokens, n_velocity):
    notes = []
    frame = 0
    last_pitch = None
    for token in tokens:
        if token == "start":
            continue
        if isinstance(token, dict) and token["type"] == "pitch":
            last_pitch = token["value"]
        if isinstance(token, dict) and token["type"] == "velocity":
            assert last_pitch is not None
            notes.append(
                Note(
                    onset=frame,
                    pitch=last_pitch + 21,
                    velocity=int(token["value"] * (128 / n_velocity)),
                )
            )
        if token == "next_frame":
            frame += 1
        if token == "end":
            break
    return Pianoroll(notes)
