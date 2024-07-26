from math import prod
import json
from typing import Sequence

import torch


class WordArray:
    def __init__(self, type, dimensions: dict[str, Sequence]):
        self.type = type
        self.dimensions = dimensions
        self.length = prod([len(values) for values in dimensions.values()])
        self.token_to_idx = {}
        self.idx_to_token = []
        i = 0
        import itertools

        for token in itertools.product(*dimensions.values()):
            token = dict(zip(dimensions.keys(), token))
            token["type"] = type
            self.token_to_idx[json.dumps(token, sort_keys=True)] = i
            self.idx_to_token.append(token)
            i += 1

    def __len__(self):
        return self.length

    def tokens(self):
        return self.idx_to_token


class Word:
    def __init__(self, type: str):
        self.type = type
        self.idx_to_token = [type]

    def __len__(self):
        return 1

    def tokens(self):
        return [{"type": self.type}]


class Vocabulary:
    """
    maps tokens to indices and vice versa.
    """

    def __init__(self, vocab: list[str | WordArray | Word]):
        self.vocab: list[WordArray | Word] = []
        for w in vocab:
            if isinstance(w, str):
                w = Word(w)
            self.vocab.append(w)

        self._token_to_idx = {}
        self._idx_to_token = {}
        i = 0

        for word in self.vocab:
            for token in word.tokens():
                self._token_to_idx[json.dumps(token, sort_keys=True)] = i
                self._idx_to_token[i] = token
                i += 1

        self.token_type_to_range: dict[str, range] = {}
        i = 0
        for word in self.vocab:
            self.token_type_to_range[word.type] = range(i, i + len(word))
            i += len(word)

    def __len__(self):
        return sum([len(token) for token in self.vocab])

    def get_idx(self, token):
        if isinstance(token, dict):
            token = json.dumps(token, sort_keys=True)
        return self._token_to_idx[token]

    def get_token(self, idx):
        return self._idx_to_token[idx]

    def __getitem__(self, token):
        if isinstance(token, int):
            return self.get_token(token)
        return self.get_idx(token)

    def get_range(self, token_or_type: str | dict) -> range:
        if isinstance(token_or_type, str):
            # it's a token type
            return self.token_type_to_range[token_or_type]
        else:
            # it's a token
            idx = self.get_idx(token_or_type)
            return range(idx, idx + 1)

    def tokens_to_one_hot(self, tokens: list[dict]):
        """
        Returns [len(tokens), len(token_map)] tensor.
        Uses sparse tensor to efficiently create a tensor from tokens.
        """

        indices = []
        values = []
        for i, token in enumerate(tokens):
            idx = self.get_idx(token)
            indices.append([i, idx])
            values.append(1)

        return torch.sparse_coo_tensor(
            torch.tensor(indices).T, torch.tensor(values), [len(tokens), len(self)]
        )

    def tokens_to_indices(self, tokens: list[dict]):
        """
        Returns [len(tokens)] tensor.
        """

        return torch.tensor([self.get_idx(token) for token in tokens], dtype=torch.long)

    def get_mask(
        self,
        tokens: list[str | dict | list[str | dict]],
        positive_value=0,
        negative_value=-1e7,
    ):
        mask = torch.zeros(len(tokens), len(self))
        mask = mask + negative_value

        for i, token in enumerate(tokens):
            if isinstance(token, list):
                for t in token:
                    mask[i, self.get_range(t)] = positive_value
            else:
                mask[i, self.get_range(token)] = positive_value

        return mask
