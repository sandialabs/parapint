import numpy as np
from scipy.sparse import eye
from typing import Sequence, Dict


def distribute_blocks(num_blocks: int, rank: int, size: int) -> Sequence[int]:
    local_blocks = list()
    for ndx in range(num_blocks):
        if ndx % size == rank:
            local_blocks.append(ndx)
    return local_blocks


def get_ownership_map(num_blocks: int, size: int) -> Dict[int, int]:
    ownership_map = dict()
    for ndx in range(num_blocks):
        for rank in range(size):
            if ndx % size == rank:
                ownership_map[ndx] = rank
                break
    return ownership_map


def get_random_n_diagonal_matrix(n, nnz_per_row):
    assert nnz_per_row % 2 == 1
    m = eye(m=n, n=n, k=0, format="coo")
    for ndx in range(1, int((nnz_per_row - 1) / 2) + 1):
        m += eye(m=n, n=n, k=ndx, format="coo")
        m += eye(m=n, n=n, k=-ndx, format="coo")
    m.data *= np.random.normal(loc=0, scale=5, size=m.data.size)
    return m
