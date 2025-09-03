from parapint.linalg.results import LinearSolverStatus, LinearSolverResults
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from mpi4py import MPI
import itertools
from typing import Dict, Optional, List


comm: MPI.Comm = MPI.COMM_WORLD
rank: int = comm.Get_rank()
size: int = comm.Get_size()

def _process_sub_results(res, sub_res):
    if sub_res.status == LinearSolverStatus.successful:
        pass
    else:
        res.status = sub_res.status


def _gather_results(res: LinearSolverResults) -> LinearSolverResults:
    stat = res.status.value
    stats = comm.allgather(stat)
    sub_res = LinearSolverResults()
    res = LinearSolverResults()
    res.status = LinearSolverStatus.successful
    for stat in stats:
        sub_res.status = LinearSolverStatus(stat)
        _process_sub_results(res, sub_res)
        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            break
    return res


class _BorderMatrix(object):
    def __init__(self, matrix):
        self.csr: csr_matrix = matrix.tocsr()
        self.nonzero_rows: np.ndarray = self._get_nonzero_rows()
        self.selection_matrix: csr_matrix = self._get_selection_matrix()

        # maps row index to index in self.nonzero_rows
        self.nonzero_row_to_ndx_map: dict = self._get_nonzero_row_to_ndx_map()

        self.sc_data_offset: Optional[int] = None

    def _get_nonzero_rows(self):
        _tmp = np.empty(self.csr.indptr.size, dtype=np.int64)
        _tmp[0:-1] = self.csr.indptr[1:]
        _tmp[-1] = self.csr.indptr[-1]
        nonzero_rows = (_tmp - self.csr.indptr).nonzero()[0]
        return nonzero_rows

    def _get_nonzero_row_to_ndx_map(self):
        res = dict()
        for i, _row in enumerate(self.nonzero_rows):
            res[_row] = i
        return res
    
    def _get_reduced_matrix(self):
        return self.csr[self.nonzero_rows, :]
    
    def _get_selection_matrix(self, format='csr'):
        data = np.ones(self.nonzero_rows.size, dtype=np.int64)
        row_idx = self.nonzero_rows
        col_idx = np.arange(self.nonzero_rows.size) # Note: assumes linear ordering
        coo_n = coo_matrix((data, (row_idx, col_idx)), shape=(self.csr.shape[0], self.nonzero_rows.size))
        if format == 'coo':
            return coo_n
        return coo_n.tocsr()

    @property
    def num_nonzero_rows(self):
        return self.nonzero_rows.size
    
    @property
    def reduced_matrix(self):
        return self.csr[self.nonzero_rows, :]


def _get_nested_comms() -> List[MPI.Comm]:
    nested_comms = list()  # root first, leaf last
    nested_comms.append(comm)

    last_comm = comm
    while last_comm.Get_size() > 3:
        if last_comm.Get_rank() < last_comm.Get_size()/2:
            color = 0
        else:
            color = 1
        last_comm = last_comm.Split(color, last_comm.Get_rank())
        nested_comms.append(last_comm)

    return nested_comms


def _combine_nonzero_elements(rows, cols):
    nonzero_elements = list(zip(rows, cols))
    nonzero_elements = {i: None for i in nonzero_elements}
    nonzero_elements = list(nonzero_elements.keys())
    nonzero_elements.sort()
    nonzero_rows, nonzero_cols = tuple(zip(*nonzero_elements))
    nonzero_rows = np.asarray(nonzero_rows, dtype=np.int64)
    nonzero_cols = np.asarray(nonzero_cols, dtype=np.int64)
    return nonzero_rows, nonzero_cols


def _get_all_nonzero_elements_in_sc(border_matrices: Dict[int, _BorderMatrix]):
    nested_comms = _get_nested_comms()

    nonzero_rows = np.zeros(0, dtype=np.int64)
    nonzero_cols = np.zeros(0, dtype=np.int64)

    for ndx, mat in border_matrices.items():
        mat_nz_elements = list(itertools.product(mat.nonzero_rows, mat.nonzero_rows))
        mat_nz_rows, mat_nz_cols = tuple(zip(*mat_nz_elements))
        nonzero_rows = np.concatenate([nonzero_rows, mat_nz_rows])
        nonzero_cols = np.concatenate([nonzero_cols, mat_nz_cols])
        nonzero_rows, nonzero_cols = _combine_nonzero_elements(nonzero_rows, nonzero_cols)

    for _comm in reversed(nested_comms):
        tmp_nz_rows_size = np.zeros(_comm.Get_size(), dtype=np.int64)
        tmp_nz_cols_size = np.zeros(_comm.Get_size(), dtype=np.int64)

        tmp_nz_rows_size[_comm.Get_rank()] = nonzero_rows.size
        tmp_nz_cols_size[_comm.Get_rank()] = nonzero_cols.size

        nz_rows_size = np.zeros(_comm.Get_size(), dtype=np.int64)
        nz_cols_size = np.zeros(_comm.Get_size(), dtype=np.int64)

        _comm.Allreduce(tmp_nz_rows_size, nz_rows_size)
        _comm.Allreduce(tmp_nz_cols_size, nz_cols_size)

        all_nonzero_rows = np.zeros(nz_rows_size.sum(), dtype=np.int64)
        all_nonzero_cols = np.zeros(nz_cols_size.sum(), dtype=np.int64)

        _comm.Allgatherv(nonzero_rows, [all_nonzero_rows, nz_rows_size])
        _comm.Allgatherv(nonzero_cols, [all_nonzero_cols, nz_cols_size])

        nonzero_rows = all_nonzero_rows
        nonzero_cols = all_nonzero_cols

        nonzero_rows, nonzero_cols = _combine_nonzero_elements(nonzero_rows, nonzero_cols)

    return nonzero_rows, nonzero_cols

def _get_all_nonzero_elements_in_sc_using_ix(border_matrices: Dict[int, _BorderMatrix],
                                             local_block_indices: List[int],
                                             num_blocks: int):

    num_nnz_per_local_blocks = np.array([border_matrices[ndx].num_nonzero_rows for ndx in local_block_indices], dtype=np.int64)
    if rank == 0:
        all_num_nnz_per_block = np.zeros(num_blocks, dtype=np.int64)
    else:
        all_num_nnz_per_block = None

    # Collect num. of nonzeros per block in correct order
    sendcounts = np.array(comm.gather(len(num_nnz_per_local_blocks), 0))
    comm.Gatherv(sendbuf=num_nnz_per_local_blocks,
                 recvbuf=(all_num_nnz_per_block, sendcounts),
                 root=0)

    local_nnzs = np.concatenate([border_matrices[ndx].nonzero_rows for ndx in local_block_indices])
    sendcounts = np.array(comm.gather(len(local_nnzs), 0))
    if rank == 0:
        nnz_indices_vector = np.zeros(sum(sendcounts), dtype=np.int64)
    else:
        nnz_indices_vector = None

    # Collect nonzero indices for each block in 1-D array, ordered as above
    comm.Gatherv(sendbuf=local_nnzs, 
                 recvbuf=(nnz_indices_vector, sendcounts),
                 root=0)

    if rank == 0:
        # Retrieve indices for each block from 1-D array
        local_indices_vectors = np.split(nnz_indices_vector, np.cumsum(all_num_nnz_per_block))
        sc_dim = border_matrices[local_block_indices[0]].csr.shape[0]
        sc = np.zeros((sc_dim, sc_dim), dtype=np.bool_)
        # Define incidence of SC by sum of cross product for all local block indices
        for local_indices in local_indices_vectors:
            sc[np.ix_(local_indices, local_indices)] = True
        sc = coo_matrix(sc)
        nonzero_rows = sc.row
        nonzero_cols = sc.col
    else:
        nonzero_rows = np.zeros(0, dtype=np.int64)
        nonzero_cols = np.zeros(0, dtype=np.int64)

    nonzero_rows = comm.bcast(nonzero_rows, root=0)
    nonzero_cols = comm.bcast(nonzero_cols, root=0)

    return nonzero_rows, nonzero_cols

