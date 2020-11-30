from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
from parapint.linalg.base_linear_solver_interface import LinearSolverInterface
from parapint.linalg.results import LinearSolverStatus, LinearSolverResults
from scipy.sparse import coo_matrix
from typing import Dict
from pyomo.common.timing import HierarchicalTimer


def _process_sub_results(res, sub_res):
    if sub_res.status == LinearSolverStatus.successful:
        pass
    else:
        res.status = sub_res.status


class SchurComplementLinearSolver(LinearSolverInterface):
    """
    Solve the system Ax = b
    A must be block-bordered diagonal and symmetric:

    K1          transpose(A1)
        k2      transpose(A2)
            k3  transpose(A3)
    A1  A2  A3  Q

    Only the lower diagonal needs supplied
    """
    def __init__(self, subproblem_solvers: Dict[int, LinearSolverInterface],
                 schur_complement_solver: LinearSolverInterface):
        """
        Parameters
        ----------
        subproblem_solvers: dict
            Dictionary mapping block index to linear solver
        schur_complement_solver: LinearSolverInterface
            Linear solver to use for factorizing the schur complement
        """
        self.subproblem_solvers: Dict[int, LinearSolverInterface] = subproblem_solvers
        self.schur_complement_solver: LinearSolverInterface = schur_complement_solver
        self.dim = 0
        self.block_dim = 0
        self.block_matrix = None

    def do_symbolic_factorization(self,
                                  matrix: BlockMatrix,
                                  raise_on_error: bool = True,
                                  timer=None) -> LinearSolverResults:
        """
        Parameters
        ----------
        matrix: BlockMatrix
        raise_on_error: bool
        timer: HierarchicalTimer

        Returns
        -------
        res: LinearSolverResults
        """
        block_matrix = matrix
        nbrows, nbcols = block_matrix.bshape
        if nbrows != nbcols:
            raise ValueError('The block matrix provided is not square.')
        self.block_dim = nbrows

        nrows, ncols = block_matrix.shape
        if nrows != ncols:
            raise ValueError('The block matrix provided is not square.')
        self.dim = nrows

        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        for ndx in range(self.block_dim - 1):
            sub_res = self.subproblem_solvers[ndx].do_symbolic_factorization(matrix=block_matrix.get_block(ndx, ndx),
                                                                             raise_on_error=raise_on_error)
            _process_sub_results(res, sub_res)
            if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
                break
        return res

    def do_numeric_factorization(self,
                                 matrix: BlockMatrix,
                                 raise_on_error: bool = True,
                                 timer=None) -> LinearSolverResults:
        """
        Parameters
        ----------
        matrix: BlockMatrix
        raise_on_error: bool
        timer: HierarchicalTimer

        Returns
        -------
        res: LinearSolverResults
        """
        self.block_matrix = block_matrix = matrix

        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        for ndx in range(self.block_dim - 1):
            sub_res = self.subproblem_solvers[ndx].do_numeric_factorization(matrix=block_matrix.get_block(ndx, ndx),
                                                                            raise_on_error=raise_on_error)
            _process_sub_results(res, sub_res)
            if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
                break
        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            return res

        schur_complement = block_matrix.get_block(self.block_dim - 1, self.block_dim - 1).toarray()

        # in a scipy csr_matrix,
        #     data contains the values
        #     indices contains the column indices
        #     indptr contains the number of nonzeros in the row
        for ndx in range(self.block_dim - 1):
            A = block_matrix.get_block(self.block_dim-1, ndx).tocsr()
            for row_ndx in range(A.shape[0]):
                row_nnz = A.indptr[row_ndx + 1] - A.indptr[row_ndx]
                if row_nnz != 0:
                    _rhs = A[row_ndx, :].toarray()[0]
                    contribution = self.subproblem_solvers[ndx].do_back_solve(_rhs)
                    schur_complement[:, row_ndx] -= A.dot(contribution)
        schur_complement = coo_matrix(schur_complement)
        sub_res = self.schur_complement_solver.do_symbolic_factorization(schur_complement, raise_on_error=raise_on_error)
        _process_sub_results(res, sub_res)
        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            return res
        sub_res = self.schur_complement_solver.do_numeric_factorization(schur_complement, raise_on_error=raise_on_error)
        _process_sub_results(res, sub_res)
        return res

    def do_back_solve(self, rhs):
        """
        Parameters
        ----------
        rhs: BlockVector

        Returns
        -------
        result: BlockVector
        """
        schur_complement_rhs = rhs.get_block(self.block_dim - 1)
        for ndx in range(self.block_dim - 1):
            A = self.block_matrix.get_block(self.block_dim-1, ndx)
            contribution = self.subproblem_solvers[ndx].do_back_solve(rhs.get_block(ndx))
            schur_complement_rhs -= A.tocsr().dot(contribution)

        result = BlockVector(self.block_dim)
        coupling = self.schur_complement_solver.do_back_solve(schur_complement_rhs)
        result.set_block(self.block_dim-1, coupling)

        for ndx in range(self.block_dim - 1):
            A = self.block_matrix.get_block(self.block_dim-1, ndx)
            result.set_block(ndx, self.subproblem_solvers[ndx].do_back_solve(rhs.get_block(ndx) - A.tocsr().transpose().dot(coupling.flatten())))

        return result

    def get_inertia(self):
        num_pos = 0
        num_neg = 0
        num_zero = 0

        for ndx in range(self.block_dim - 1):
            _pos, _neg, _zero = self.subproblem_solvers[ndx].get_inertia()
            num_pos += _pos
            num_neg += _neg
            num_zero += _zero
        _pos, _neg, _zero = self.schur_complement_solver.get_inertia()
        num_pos += _pos
        num_neg += _neg
        num_zero += _zero

        return num_pos, num_neg, num_zero

    def increase_memory_allocation(self, factor):
        for ndx, sub_solver in self.subproblem_solvers.items():
            sub_solver.increase_memory_allocation(factor=factor)
        self.schur_complement_solver.increase_memory_allocation(factor=factor)
