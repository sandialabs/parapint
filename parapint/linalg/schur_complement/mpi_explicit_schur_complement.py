from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
from parapint.linalg.base_linear_solver_interface import LinearSolverInterface
from parapint.linalg.results import LinearSolverStatus, LinearSolverResults
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from mpi4py import MPI
import itertools
from .explicit_schur_complement import _process_sub_results
from typing import Dict, Optional
from pyomo.common.timing import HierarchicalTimer


comm: MPI.Comm = MPI.COMM_WORLD
rank: int = comm.Get_rank()
size: int = comm.Get_size()


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

        # maps row index to index in self.nonzero_rows
        self.nonzero_row_to_ndx_map: dict = self._get_nonzero_row_to_ndx_map()

        self.sc_data_offset: Optional[int] = None

    def _get_nonzero_rows(self):
        _tmp = np.empty(self.csr.indptr.size, dtype=np.int)
        _tmp[0:-1] = self.csr.indptr[1:]
        _tmp[-1] = self.csr.indptr[-1]
        nonzero_rows = (_tmp - self.csr.indptr).nonzero()[0]
        return nonzero_rows

    def _get_nonzero_row_to_ndx_map(self):
        res = dict()
        for i, _row in enumerate(self.nonzero_rows):
            res[_row] = i
        return res

    @property
    def num_nonzero_rows(self):
        return self.nonzero_rows.size


class MPISchurComplementLinearSolver(LinearSolverInterface):
    """

    Solve the system Ax = b.

    A must be block-bordered-diagonal and symmetric::

      K1          transpose(A1)
          K2      transpose(A2)
              K3  transpose(A3)
      A1  A2  A3  Q

    Only the lower diagonal needs supplied.

    Some assumptions are made on the block matrices provided to do_symbolic_factorization and do_numeric_factorization:
      * Q must be owned by all processes
      * K :sub:`i` and A :sub:`i` must be owned by the same process

    Parameters
    ----------
    subproblem_solvers: dict
        Dictionary mapping block index to linear solver
    schur_complement_solver: LinearSolverInterface
        Linear solver to use for factorizing the schur complement

    """
    def __init__(self, subproblem_solvers: Dict[int, LinearSolverInterface],
                 schur_complement_solver: LinearSolverInterface):
        self.subproblem_solvers = subproblem_solvers
        self.schur_complement_solver = schur_complement_solver
        self.block_dim = 0
        self.block_matrix = None
        self.local_block_indices = list()
        self.schur_complement = coo_matrix((0, 0))
        self.border_matrices = dict()

    def do_symbolic_factorization(self,
                                  matrix: MPIBlockMatrix,
                                  raise_on_error: bool = True,
                                  timer: Optional[HierarchicalTimer] = None) -> LinearSolverResults:
        """
        Perform symbolic factorization. This performs symbolic factorization for each diagonal block and
        collects some information on the structure of the schur complement for sparse communication in
        the numeric factorization phase.

        Parameters
        ----------
        matrix: MPIBlockMatrix
            A Pynumero MPIBlockMatrix. This is the A matrix in Ax=b
        raise_on_error: bool
            If False, an error will not be raised if an error occurs during symbolic factorization. Instead the
            status attribute of the results object will indicate an error ocurred.
        timer: pyutilib.misc.timing.HierarchicalTimer
            A timer for profiling.

        Returns
        -------
        res: LinearSolverResults
            The results object
        """
        if timer is None:
            timer = HierarchicalTimer()

        block_matrix = matrix
        nbrows, nbcols = block_matrix.bshape
        if nbrows != nbcols:
            raise ValueError('The block matrix provided is not square.')
        self.block_dim = nbrows

        # split up the blocks between ranks
        self.local_block_indices = list()
        for ndx in range(self.block_dim - 1):
            if ((block_matrix.rank_ownership[ndx, ndx] == rank) or
                    (block_matrix.rank_ownership[ndx, ndx] == -1 and rank == 0)):
                self.local_block_indices.append(ndx)

        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        timer.start('factorize')
        for ndx in self.local_block_indices:
            sub_res = self.subproblem_solvers[ndx].do_symbolic_factorization(matrix=block_matrix.get_block(ndx, ndx),
                                                                             raise_on_error=False)
            _process_sub_results(res, sub_res)
            if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
                break
        timer.stop('factorize')
        res = _gather_results(res)
        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            if raise_on_error:
                raise RuntimeError('Symbolic factorization unsuccessful; status: ' + str(res.status))
            else:
                return res

        timer.start('sc_structure')
        self._get_sc_structure(block_matrix=block_matrix, timer=timer)
        timer.stop('sc_structure')

        return res

    def _get_sc_structure(self, block_matrix, timer):
        """
        Parameters
        ----------
        block_matrix: pyomo.contrib.pynumero.sparse.mpi_block_matrix.MPIBlockMatrix
        """
        self.border_matrices = dict()
        tmp_num_nonzero_rows = np.zeros(self.block_dim - 1, dtype=np.int)
        for ndx in self.local_block_indices:
            self.border_matrices[ndx] = border_matrix = _BorderMatrix(block_matrix.get_block(self.block_dim - 1, ndx))
            tmp_num_nonzero_rows[ndx] = border_matrix.num_nonzero_rows
        num_nonzero_rows = np.zeros(self.block_dim - 1, dtype=np.int)
        comm.Allreduce(tmp_num_nonzero_rows, num_nonzero_rows)

        sc_nnz = 0
        local_block_indices_set = set(self.local_block_indices)
        for block_ndx in range(self.block_dim - 1):
            if block_ndx in local_block_indices_set:
                border_matrix = self.border_matrices[block_ndx]
                border_matrix.sc_data_offset = sc_nnz
            sc_nnz += num_nonzero_rows[block_ndx]**2

        sc_row = np.zeros(sc_nnz, dtype=np.int)
        sc_col = np.zeros(sc_nnz, dtype=np.int)
        for block_ndx in self.local_block_indices:
            border_matrix = self.border_matrices[block_ndx]
            for i, _row in enumerate(border_matrix.nonzero_rows):
                start = border_matrix.sc_data_offset + i*border_matrix.num_nonzero_rows
                end = start + border_matrix.num_nonzero_rows
                sc_row[start:end] = _row
                _slice = np.arange(border_matrix.sc_data_offset + i,
                                   border_matrix.sc_data_offset + border_matrix.num_nonzero_rows**2,
                                   border_matrix.num_nonzero_rows)
                sc_col[_slice] = _row
        global_row = np.zeros(sc_nnz, dtype=np.int)
        global_col = np.zeros(sc_nnz, dtype=np.int)
        timer.start('Allreduce')
        comm.Allreduce(sc_row, global_row)
        comm.Allreduce(sc_col, global_col)
        timer.stop('Allreduce')
        sc_values = np.zeros(sc_nnz, dtype=np.double)
        sc_dim = block_matrix.get_row_size(self.block_dim - 1)
        self.schur_complement = coo_matrix((sc_values, (global_row, global_col)), shape=(sc_dim, sc_dim))

    def do_numeric_factorization(self,
                                 matrix: MPIBlockMatrix,
                                 raise_on_error: bool = True,
                                 timer: Optional[HierarchicalTimer] = None) -> LinearSolverResults:
        """
        Perform numeric factorization:
          * perform numeric factorization on each diagonal block
          * form and communicate the Schur-Complement
          * factorize the schur-complement

        This method should only be called after do_symbolic_factorization.

        Parameters
        ----------
        matrix: MPIBlockMatrix
            A Pynumero MPIBlockMatrix. This is the A matrix in Ax=b
        raise_on_error: bool
            If False, an error will not be raised if an error occurs during symbolic factorization. Instead the
            status attribute of the results object will indicate an error ocurred.
        timer: pyutilib.misc.timing.HierarchicalTimer
            A timer for profiling.

        Returns
        -------
        res: LinearSolverResults
            The results object
        """
        if timer is None:
            timer = HierarchicalTimer()

        self.block_matrix = block_matrix = matrix

        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        timer.start('form SC')
        for ndx in self.local_block_indices:
            timer.start('factorize')
            sub_res = self.subproblem_solvers[ndx].do_numeric_factorization(matrix=block_matrix.get_block(ndx, ndx),
                                                                            raise_on_error=False)
            timer.stop('factorize')
            _process_sub_results(res, sub_res)
            if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
                break
        res = _gather_results(res)
        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            if raise_on_error:
                raise RuntimeError('Numeric factorization unsuccessful; status: ' + str(res.status))
            else:
                timer.stop('form SC')
                return res

        # in a scipy csr_matrix,
        #     data contains the values
        #     indices contains the column indices
        #     indptr contains the number of nonzeros in the row
        self.schur_complement.data = np.zeros(self.schur_complement.data.size, dtype=np.double)
        for ndx in self.local_block_indices:
            border_matrix: _BorderMatrix = self.border_matrices[ndx]
            _slice = (border_matrix.sc_data_offset +
                      np.arange(border_matrix.num_nonzero_rows, dtype=np.int) * border_matrix.num_nonzero_rows)
            A = border_matrix.csr
            _rhs = np.zeros(A.shape[1], dtype=np.double)
            solver = self.subproblem_solvers[ndx]
            for row_ndx in border_matrix.nonzero_rows:
                for indptr in range(A.indptr[row_ndx], A.indptr[row_ndx + 1]):
                    col = A.indices[indptr]
                    val = A.data[indptr]
                    _rhs[col] += val
                timer.start('back solve')
                contribution = solver.do_back_solve(_rhs)
                timer.stop('back solve')
                timer.start('dot product')
                contribution = A.dot(contribution)
                timer.stop('dot product')
                self.schur_complement.data[_slice + border_matrix.nonzero_row_to_ndx_map[row_ndx]] -= contribution[border_matrix.nonzero_rows]
                for indptr in range(A.indptr[row_ndx], A.indptr[row_ndx + 1]):
                    col = A.indices[indptr]
                    val = A.data[indptr]
                    _rhs[col] -= val

        timer.start('communicate')
        timer.start('zeros')
        sc = np.zeros(self.schur_complement.data.size, dtype=np.double)
        timer.stop('zeros')
        timer.start('Barrier')
        comm.Barrier()
        timer.stop('Barrier')
        timer.start('Allreduce')
        comm.Allreduce(self.schur_complement.data, sc)
        timer.stop('Allreduce')
        self.schur_complement.data = sc
        timer.start('add')
        sc = self.schur_complement + block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()
        timer.stop('add')
        timer.stop('communicate')
        timer.stop('form SC')

        timer.start('factor SC')
        sub_res = self.schur_complement_solver.do_symbolic_factorization(sc, raise_on_error=raise_on_error)
        _process_sub_results(res, sub_res)
        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            timer.stop('factor SC')
            return res
        sub_res = self.schur_complement_solver.do_numeric_factorization(sc)
        _process_sub_results(res, sub_res)
        timer.stop('factor SC')
        return res

    def do_back_solve(self, rhs, timer=None):
        """
        Performs a back solve with the factorized matrix. Should only be called after
        do_numeric_factorixation.

        Parameters
        ----------
        rhs: MPIBlockVector
        timer: pyutilib.misc.timing.HierarchicalTimer

        Returns
        -------
        result: MPIBlockVector
        """
        if timer is None:
            timer = HierarchicalTimer()
        timer.start('back_solve')

        schur_complement_rhs = np.zeros(rhs.get_block(self.block_dim - 1).size, dtype='d')
        for ndx in self.local_block_indices:
            A = self.block_matrix.get_block(self.block_dim-1, ndx)
            contribution = self.subproblem_solvers[ndx].do_back_solve(rhs.get_block(ndx))
            schur_complement_rhs -= A.tocsr().dot(contribution.flatten())
        res = np.zeros(rhs.get_block(self.block_dim - 1).shape[0], dtype='d')
        comm.Allreduce(schur_complement_rhs, res)
        schur_complement_rhs = rhs.get_block(self.block_dim - 1) + res

        result = rhs.copy_structure()
        coupling = self.schur_complement_solver.do_back_solve(schur_complement_rhs)

        for ndx in self.local_block_indices:
            A = self.block_matrix.get_block(self.block_dim-1, ndx)
            result.set_block(ndx, self.subproblem_solvers[ndx].do_back_solve(rhs.get_block(ndx) -
                                                                             A.tocsr().transpose().dot(coupling.flatten())))

        result.set_block(self.block_dim-1, coupling)

        timer.stop('back_solve')

        return result

    def get_inertia(self):
        """
        Get the inertia. Should only be called after do_numeric_factorization.

        Returns
        -------
        num_pos: int
            The number of positive eigenvalues of A
        num_neg: int
            The number of negative eigenvalues of A
        num_zero: int
            The number of zero eigenvalues of A
        """
        num_pos = 0
        num_neg = 0
        num_zero = 0

        for ndx in self.local_block_indices:
            _pos, _neg, _zero = self.subproblem_solvers[ndx].get_inertia()
            num_pos += _pos
            num_neg += _neg
            num_zero += _zero

        num_pos = comm.allreduce(num_pos)
        num_neg = comm.allreduce(num_neg)
        num_zero = comm.allreduce(num_zero)

        _pos, _neg, _zero = self.schur_complement_solver.get_inertia()
        num_pos += _pos
        num_neg += _neg
        num_zero += _zero

        return num_pos, num_neg, num_zero

    def increase_memory_allocation(self, factor):
        """
        Increases the memory allocation of each sub-solver. This method should only be called
        if the results status from do_symbolic_factorization or do_numeric_factorization is
        LinearSolverStatus.not_enough_memory.

        Parameters
        ----------
        factor: float
            The factor by which to increase memory allocation. Should be greater than 1.
        """
        for ndx in self.local_block_indices:
            sub_solver = self.subproblem_solvers[ndx]
            sub_solver.increase_memory_allocation(factor=factor)
        self.schur_complement_solver.increase_memory_allocation(factor=factor)
