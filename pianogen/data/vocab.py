from math import prod
import json
from typing import Sequence

import torch


class WordArray:
    def __init__(self, name:str, dimensions: dict[str, Sequence]):
        self.name = name
        self.dimensions = dimensions
        self.length = prod([len(values) for values in dimensions.values()])
        self.token_to_idx = {}
        self.idx_to_token = []
        i = 0
        import itertools

        for token in itertools.product(*dimensions.values()):
            token = dict(zip(dimensions.keys(), token))
            self.token_to_idx[json.dumps(token, sort_keys=True)] = i
            self.idx_to_token.append(token)
            i += 1

    def __len__(self):
        return self.length

    def tokens(self):
        return self.idx_to_token


class Word:
    def __init__(self, name: str):
        self.name = name
        self.idx_to_token = [name]

    def __len__(self):
        return 1

    def tokens(self):
        return [self.name]


class Vocabulary:
    """
    maps tokens to indices and vice versa.
    """

    def __init__(self, vocab: Sequence[str |dict| WordArray | Word]):
        self.vocab: list[WordArray | Word] = []
        for w in vocab:
            if isinstance(w, dict):
                w = Word(json.dumps(w, sort_keys=True))
            if isinstance(w, str):
                w = Word(w)
            self.vocab.append(w)

        self._token_to_idx = {}
        self._idx_to_token = {}
        i = 0

        for word in self.vocab:
            for token in word.tokens():
                if not isinstance(token, str):
                    token = json.dumps(token, sort_keys=True)
                self._token_to_idx[token] = i
                self._idx_to_token[i] = token
                i += 1

        self.token_type_to_range: dict[str, range] = {}
        i = 0
        for word in self.vocab:
            self.token_type_to_range[word.name] = range(i, i + len(word))
            i += len(word)

    def __len__(self):
        return sum([len(token) for token in self.vocab])

    def get_idx(self, token):
        if isinstance(token, dict):
            token = json.dumps(token, sort_keys=True)
        return self._token_to_idx[token]

    def get_token(self, idx):
        result = self._idx_to_token[idx]
        if result[0] == "{":
            return json.loads(result)
        return result

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
        '''
        Returns a mask tensor of shape [len(tokens), len(vocab)] with positive_value for tokens in the list and negative_value for the rest.
        '''
        mask = torch.zeros(len(tokens), len(self))
        mask = mask + negative_value

        for i, token in enumerate(tokens):
            if isinstance(token, list):
                for t in token:
                    mask[i, self.get_range(t)] = positive_value
            else:
                mask[i, self.get_range(token)] = positive_value

        return mask
