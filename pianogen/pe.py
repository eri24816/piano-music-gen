import torch


def binary_positional_encoding(length: int, dim: int):
    res = []
    for i in range(length):
        res.append([int(x) for x in f"{i:0{dim}b}"][-dim:])
        # pad
        res[-1] += [0] * (dim - len(res[-1]))

    return torch.tensor(res, dtype=torch.float32)


def sinusoidal_positional_encoding(length: int, dim: int):
    res = []
    for d in range(dim // 2):
        res.append(torch.sin(torch.arange(length) / 10000 ** (2 * d / dim)))
    for d in range(dim // 2):
        res.append(torch.cos(torch.arange(length) / 10000 ** (2 * d / dim)))
    if dim % 2:
        res.append(torch.zeros(length))
    return torch.stack(res, dim=1)
