import enum
import importlib
import inspect
from abc import abstractmethod

import numpy as np
import scipy.sparse.linalg
from mpi4py import MPI
from mpi4py.util import dtlib
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
from scipy.sparse import coo_matrix

from parapint.linalg.base_linear_solver_interface import LinearSolverInterface
from parapint.linalg.iterative import (
    LbfgsSamplingOptions,
    PcgOptions,
    PcgSolution,
    PcgSolutionStatus,
    pcg_solve,
)
from parapint.linalg.results import LinearSolverResults, LinearSolverStatus
from parapint.linalg.schur_complement.utils import (
    _BorderMatrix,
    _gather_results,
    _get_all_nonzero_elements_in_sc_using_ix,
    _process_sub_results,
)

comm: MPI.Comm = MPI.COMM_WORLD
rank: int = comm.Get_rank()
size: int = comm.Get_size()


class ImplicitSchurComplementPreconditionerType(enum.Enum):
    """Options for implicit Schur complement preconditioners."""

    incomplete_cholesky = 1
    incomplete_lu = 2
    lbfgs = 3
    adaptive_refactorization = 4
    block_jacobi = 5
    additive_schwarz = 6
    diagonal_additive_schwarz = 7


class MPIBaseImplicitSchurComplementLinearSolver(LinearSolverInterface):
    """Base class for implicit Schur complement linear solvers."""

    def __init__(
        self,
        subproblem_solvers: dict[int, LinearSolverInterface],
        pcg_options: PcgOptions,
    ) -> None:
        self.subproblem_solvers = subproblem_solvers
        self.border_matrices: dict[int, _BorderMatrix] = dict()
        self.pcg_options = pcg_options
        self.local_block_indices = list()
        self.block_matrix: MPIBlockMatrix | None = None
        self.local_var_indices = list()
        self.sc_dim: int = 0
        self.current_pcg_solution: PcgSolution | None = None

    @abstractmethod
    def _get_sc_structure(
        self, block_matrix: MPIBlockMatrix, timer: HierarchicalTimer | None = None
    ) -> None:
        pass

    @abstractmethod
    def _form_sc_components(self, timer: HierarchicalTimer | None = None) -> None:
        pass

    @abstractmethod
    def _factorize_sc_components(
        self, timer: HierarchicalTimer | None = None
    ) -> LinearSolverResults:
        pass

    @abstractmethod
    def _apply_preconditioner(self, r: np.ndarray) -> np.ndarray:
        pass

    def _update_preconditioner(self, pcg_sol: PcgSolution) -> None:
        pass

    def _symbolic_factorize_diag_blocks(
        self,
        matrix: MPIBlockMatrix,
        timer: HierarchicalTimer,
    ) -> LinearSolverResults:
        self.block_matrix = block_matrix = matrix
        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        timer.start("factorize_diag_blocks")
        for ndx in self.local_block_indices:
            sub_res = self.subproblem_solvers[ndx].do_symbolic_factorization(
                matrix=block_matrix.get_block(ndx, ndx), raise_on_error=False
            )
            _process_sub_results(res, sub_res)
            if res.status not in {
                LinearSolverStatus.successful,
                LinearSolverStatus.warning,
            }:
                break
        timer.stop("factorize_diag_blocks")
        res = _gather_results(res)
        return res

    def _build_border_matrices(self, block_matrix: MPIBlockMatrix):
        self.border_matrices = dict()
        for ndx in self.local_block_indices:
            self.border_matrices[ndx] = _BorderMatrix(
                block_matrix.get_block(self.block_dim - 1, ndx)
            )
            self.local_var_indices.extend(
                self.border_matrices[ndx].nonzero_rows.tolist()
            )
        self.local_var_indices = sorted(list(set(self.local_var_indices)))

    def do_symbolic_factorization(
        self,
        matrix: MPIBlockMatrix,
        raise_on_error: bool = True,
        timer: HierarchicalTimer | None = None,
    ) -> LinearSolverResults:
        """Perform symbolic factorization of all relevant blocks."""
        if timer is None:
            timer = HierarchicalTimer()

        self.block_matrix = block_matrix = matrix
        nbrows, nbcols = block_matrix.bshape
        if nbrows != nbcols:
            raise ValueError("The block matrix provided is not square.")
        self.block_dim = nbrows
        self.sc_dim = block_matrix.get_row_size(self.block_dim - 1)

        # split up the blocks between ranks
        self.local_block_indices = list()
        for ndx in range(self.block_dim - 1):
            if (block_matrix.rank_ownership[ndx, ndx] == rank) or (
                block_matrix.rank_ownership[ndx, ndx] == -1 and rank == 0
            ):
                self.local_block_indices.append(ndx)

        res = self._symbolic_factorize_diag_blocks(block_matrix, timer)

        if res.status not in {
            LinearSolverStatus.successful,
            LinearSolverStatus.warning,
        }:
            if raise_on_error:
                raise RuntimeError(
                    "Symbolic factorization unsuccessful; status: " + str(res.status)
                )
            else:
                return res

        self._build_border_matrices(block_matrix)
        self._get_sc_structure(block_matrix, timer)

        return res

    def _numeric_factorize_diag_blocks(
        self, block_matrix: MPIBlockMatrix, timer: HierarchicalTimer
    ) -> LinearSolverResults:
        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful

        for ndx in self._local_block_indices_for_numeric_factorization:
            timer.start("factorize_diag_blocks")
            sub_res = self.subproblem_solvers[ndx].do_numeric_factorization(
                matrix=block_matrix.get_block(ndx, ndx), raise_on_error=False
            )
            timer.stop("factorize_diag_blocks")
            _process_sub_results(res, sub_res)
            if res.status not in {
                LinearSolverStatus.successful,
                LinearSolverStatus.warning,
            }:
                break
        res = _gather_results(res)
        return res

    def do_numeric_factorization(
        self,
        matrix: MPIBlockMatrix,
        raise_on_error: bool = True,
        timer: HierarchicalTimer | None = None,
    ) -> LinearSolverResults:
        """Perform numeric factorization of all relevant blocks."""
        if timer is None:
            timer = HierarchicalTimer()

        self.block_matrix = block_matrix = matrix

        # factorize all local blocks
        self._local_block_indices_for_numeric_factorization = self.local_block_indices
        res = self._numeric_factorize_diag_blocks(block_matrix, timer)

        if res.status not in {
            LinearSolverStatus.successful,
            LinearSolverStatus.warning,
        }:
            if raise_on_error:
                raise RuntimeError(
                    "Numeric factorization unsuccessful; status: " + str(res.status)
                )
            else:
                return res

        self._form_sc_components(timer)

        sub_res = self._factorize_sc_components(timer)

        _process_sub_results(res, sub_res)

        if res.status not in {
            LinearSolverStatus.successful,
            LinearSolverStatus.warning,
        }:
            if raise_on_error:
                raise RuntimeError(
                    "Symbolic factorization unsuccessful; status: " + str(res.status)
                )

        return res

    def _sc_matvec(self, u: np.ndarray) -> np.ndarray:
        res = np.zeros_like(u)
        for ndx in self.local_block_indices:
            border_matrix: _BorderMatrix = self.border_matrices[ndx]
            A = border_matrix.csr.transpose()
            v = A.dot(u)
            x = self.subproblem_solvers[ndx].do_back_solve(v)
            y = A.transpose().dot(x)
            res -= y
        comm.Allreduce(MPI.IN_PLACE, res, op=MPI.SUM)
        res += self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).dot(u)
        return res

    def do_back_solve(
        self, rhs: MPIBlockVector, timer: HierarchicalTimer | None = None
    ) -> MPIBlockVector:
        """Solve KKT system for given rhs."""
        if timer is None:
            timer = HierarchicalTimer()
        timer.start("form_rhs_SC")
        sc_dim = rhs.get_block(self.block_dim - 1).size
        schur_complement_rhs = np.zeros(sc_dim, dtype="d")
        for ndx in self.local_block_indices:
            A = self.block_matrix.get_block(self.block_dim - 1, ndx)
            contribution = self.subproblem_solvers[ndx].do_back_solve(
                rhs.get_block(ndx)
            )
            schur_complement_rhs -= A.tocsr().dot(contribution.flatten())

        comm.Allreduce(MPI.IN_PLACE, schur_complement_rhs, op=MPI.SUM)
        schur_complement_rhs += rhs.get_block(self.block_dim - 1)

        timer.stop("form_rhs_SC")

        def _S(u: np.ndarray) -> np.ndarray:
            timer.start("S_matvec")
            r = self._sc_matvec(u)
            timer.stop("S_matvec")
            return r

        def _M(r: np.ndarray) -> np.ndarray:
            timer.start("M_matvec")
            u = self._apply_preconditioner(r)
            timer.stop("M_matvec")
            return u

        SC_linop = scipy.sparse.linalg.LinearOperator(
            shape=(sc_dim, sc_dim), matvec=_S, dtype="d"
        )
        M_linop = scipy.sparse.linalg.LinearOperator(
            shape=(sc_dim, sc_dim), matvec=_M, dtype="d"
        )

        if self.pcg_options.lbfgs_approx_options.distributed:
            local_var_indices = self.local_var_indices
        else:
            local_var_indices = None
        timer.start("pcg")
        pcg_sol: PcgSolution = pcg_solve(
            A=SC_linop,
            b=schur_complement_rhs,
            M=M_linop,
            pcg_options=self.pcg_options,
            local_var_indices=local_var_indices,
        )
        timer.stop("pcg")
        coupling = pcg_sol.x
        self.current_pcg_solution = pcg_sol

        self._update_preconditioner(pcg_sol)

        timer.start("local_back_solve")
        result = rhs.copy_structure()
        for ndx in self.local_block_indices:
            A = self.block_matrix.get_block(self.block_dim - 1, ndx)
            result.set_block(
                ndx,
                self.subproblem_solvers[ndx].do_back_solve(
                    rhs.get_block(ndx) - A.tocsr().transpose().dot(coupling.flatten())
                ),
            )
        timer.stop("local_back_solve")
        result.set_block(self.block_dim - 1, coupling)

        return result

    def get_inertia(self) -> tuple[int, int, int]:
        """Compute the inertia of the KKT system, assuming SC is positive definite."""
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

        # Assumes SC is pos. definitite here
        # If inertia of diagonal blocks in incorrect, it will be detected
        # If SC is not pos. def. this will be caught during PCG
        _pos, _neg, _zero = self.sc_dim, 0, 0
        num_pos += _pos
        num_neg += _neg
        num_zero += _zero

        return num_pos, num_neg, num_zero


class ImplicitUsingFullSchurComplement(MPIBaseImplicitSchurComplementLinearSolver):
    """Abstract class defining functionalities to form full Schur complement."""

    def __init__(
        self,
        subproblem_solvers: dict[int, LinearSolverInterface],
        pcg_options: PcgOptions,
    ) -> None:
        super().__init__(subproblem_solvers, pcg_options)
        self.schur_complement: coo_matrix = coo_matrix((0, 0))

    def _get_sc_structure(self, block_matrix: MPIBlockMatrix, timer: HierarchicalTimer):
        timer.start("build_border_matrices")
        self.border_matrices = dict()
        for ndx in self.local_block_indices:
            self.border_matrices[ndx] = _BorderMatrix(
                block_matrix.get_block(self.block_dim - 1, ndx)
            )
        timer.stop("build_border_matrices")
        timer.start("gather_all_nonzero_elements")
        nonzero_rows, nonzero_cols = _get_all_nonzero_elements_in_sc_using_ix(
            self.border_matrices, self.local_block_indices, self.block_dim - 1
        )
        timer.stop("gather_all_nonzero_elements")
        timer.start("construct_schur_complement")
        sc_nnz = nonzero_rows.size
        sc_dim = block_matrix.get_row_size(self.block_dim - 1)
        self.sc_dim = sc_dim
        sc_values = np.zeros(sc_nnz, dtype=np.double)
        self.schur_complement = coo_matrix(
            (sc_values, (nonzero_rows, nonzero_cols)), shape=(sc_dim, sc_dim)
        )
        timer.stop("construct_schur_complement")
        timer.start("get_sc_data_slices")
        # TODO: This part can be very slow for problems with many compl. cars per partition
        # - however it saves time in iterations later, as sparsity is already computed
        # - can we do this more efficiently(vectorize instead of loop)?
        self.sc_data_slices = dict()
        for ndx in self.local_block_indices:
            self.sc_data_slices[ndx] = dict()
            border_matrix = self.border_matrices[ndx]
            for row_ndx in border_matrix.nonzero_rows:
                self.sc_data_slices[ndx][row_ndx] = np.bitwise_and(
                    nonzero_cols == row_ndx,
                    np.isin(nonzero_rows, border_matrix.nonzero_rows),
                ).nonzero()[0]
        timer.stop("get_sc_data_slices")

    def _form_sc_components(self, timer: HierarchicalTimer) -> None:
        timer.start("form_SC")
        self.schur_complement.data = np.zeros(
            self.schur_complement.data.size, dtype=np.double
        )
        for ndx in self.local_block_indices:
            border_matrix: _BorderMatrix = self.border_matrices[ndx]
            A = border_matrix.csr
            _rhs = np.zeros(A.shape[1], dtype=np.double)
            solver = self.subproblem_solvers[ndx]
            for row_ndx in border_matrix.nonzero_rows:
                for indptr in range(A.indptr[row_ndx], A.indptr[row_ndx + 1]):
                    col = A.indices[indptr]
                    val = A.data[indptr]
                    _rhs[col] += val
                timer.start("back_solve")
                contribution = solver.do_back_solve(_rhs)
                timer.stop("back_solve")
                timer.start("dot_product")
                contribution = A.dot(contribution)
                timer.stop("dot_product")
                self.schur_complement.data[self.sc_data_slices[ndx][row_ndx]] -= (
                    contribution[border_matrix.nonzero_rows]
                )
                for indptr in range(A.indptr[row_ndx], A.indptr[row_ndx + 1]):
                    col = A.indices[indptr]
                    val = A.data[indptr]
                    _rhs[col] -= val

        timer.start("communicate")
        sc = np.zeros(self.schur_complement.data.size, dtype=np.double)
        timer.start("Barrier")
        comm.Barrier()
        timer.stop("Barrier")
        timer.start("Allreduce")
        comm.Allreduce(self.schur_complement.data, sc)
        timer.stop("Allreduce")
        self.schur_complement.data = sc
        timer.stop("communicate")
        self.schur_complement = self.schur_complement + self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()
        timer.stop("form_SC")


class MPISpICholImplicitSchurComplementLinearSolver(ImplicitUsingFullSchurComplement):
    """Implicit Schur complement linear solver using sparse incomplete Cholesky preconditioner."""

    def __init__(
        self,
        subproblem_solvers: dict[int, LinearSolverInterface],
        pcg_options: PcgOptions,
        drop_tol: float = 1e-4,
        fill_factor: float = 10.0,
    ) -> None:
        super().__init__(subproblem_solvers=subproblem_solvers, pcg_options=pcg_options)
        try:
            self._ilupp = importlib.import_module("ilupp")
        except ImportError as e:
            raise ImportError(
                "Make sure the ilupp package is installed "
                "when using the incomplete Cholesky preconditioner"
            ) from e
        self.drop_tol = drop_tol
        self.fill_factor = fill_factor
        self._spich_precond: scipy.sparse.linalg.LinearOperator | None = None

    def _apply_preconditioner(self, r: np.ndarray) -> np.ndarray:
        return self._spich_precond(r)

    def _factorize_sc_components(self, timer: HierarchicalTimer):
        timer.start("spIChol_sc")
        A = self.schur_complement.tocsc()
        self._spich_precond = self._ilupp.ICholTPreconditioner(
            A,
            threshold=self.drop_tol,
            add_fill_in=int(self.fill_factor * (A.nnz / A.shape[0])),
        )
        timer.stop("spIChol_sc")
        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        return res


class MPISpILUImplicitSchurComplementLinearSolver(ImplicitUsingFullSchurComplement):
    """Implicit Schur complement linear solver using sparse incomplete LU preconditioner."""

    def __init__(
        self,
        subproblem_solvers: dict[int, LinearSolverInterface],
        pcg_options: PcgOptions,
        scipy_spilu_kwargs: dict | None = None,
    ) -> None:
        super().__init__(subproblem_solvers=subproblem_solvers, pcg_options=pcg_options)
        self.precond_options = scipy_spilu_kwargs if scipy_spilu_kwargs is not None else {}
        self._spilu_precond = None

    def _apply_preconditioner(self, r: np.ndarray) -> np.ndarray:
        return self._spilu_precond.solve(r)

    def _factorize_sc_components(self, timer: HierarchicalTimer) -> LinearSolverResults:
        timer.start("SpILU_SC")
        self._spilu_precond = scipy.sparse.linalg.spilu(
            self.schur_complement.tocsc(), **self.precond_options
        )
        timer.stop("SpILU_SC")
        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        return res


class MPIAdaptiveImplicitSchurComplementLinearSolver(ImplicitUsingFullSchurComplement):
    """Implicit Schur complement linear solver using adaptive refactorization as preconditioner."""

    def __init__(
        self,
        subproblem_solvers: dict[int, LinearSolverInterface],
        pcg_options: PcgOptions,
        schur_complement_solver: LinearSolverInterface,
        refactorization_iter_threshold: int,
    ) -> None:
        self.pcg_options = pcg_options
        super().__init__(subproblem_solvers=subproblem_solvers, pcg_options=pcg_options)
        self._flag_form_sc = True
        self._flag_factorize_sc = True
        self._sc_solver: LinearSolverInterface = schur_complement_solver
        self._pcg_iter_threshold = refactorization_iter_threshold

    def _update_preconditioner(self, pcg_sol: PcgSolution):
        if pcg_sol.num_iterations > self._pcg_iter_threshold:
            self._flag_form_sc = True
            self._flag_factorize_sc = True
        else:
            self._flag_form_sc = False
            self._flag_factorize_sc = False

    def _apply_preconditioner(self, r: np.ndarray) -> np.ndarray:
        return self._sc_solver.do_back_solve(r)

    def _form_sc_components(self, timer: HierarchicalTimer) -> None:
        if self._flag_form_sc:
            super()._form_sc_components(timer)

    def _factorize_sc_components(self, timer: HierarchicalTimer) -> LinearSolverResults:
        if self._flag_factorize_sc:
            res = self._sc_solver.do_symbolic_factorization(
                self.schur_complement, raise_on_error=False
            )
            sub_res = self._sc_solver.do_numeric_factorization(
                matrix=self.schur_complement, raise_on_error=False
            )
            _process_sub_results(res, sub_res)
        else:
            res = LinearSolverResults()
            res.status = LinearSolverStatus.successful
        return res


class MPILbfgsImplicitSchurComplementLinearSolver(
    MPIBaseImplicitSchurComplementLinearSolver
):
    """Implicit Schur complement linear solver using lbfgs approximation as preconditioner."""

    def __init__(
        self,
        subproblem_solvers: dict[int, LinearSolverInterface],
        pcg_options: PcgOptions,
    ) -> None:
        assert (
            pcg_options.lbfgs_approx_options.sampling != LbfgsSamplingOptions.disable
        ), (
            "LBFGS sampling must be enabled to use Adaptive Refactorization Preconditioner."
        )
        self.pcg_options: PcgOptions = pcg_options
        super().__init__(subproblem_solvers=subproblem_solvers, pcg_options=pcg_options)
        self._current_pcg_solution: PcgSolution | None = None

    def _update_preconditioner(self, pcg_sol: PcgSolution):
        self._current_pcg_solution = pcg_sol

    def _apply_preconditioner(self, r: np.ndarray) -> np.ndarray:
        if self._current_pcg_solution is None:
            # No prev solution available
            return r
        else:
            return self._current_pcg_solution.hess_approx.dot(r)

    def _get_sc_structure(
        self, block_matrix: MPIBlockMatrix, timer: HierarchicalTimer | None = None
    ) -> None:
        pass

    def _form_sc_components(self, timer: HierarchicalTimer | None = None) -> None:
        pass

    def _factorize_sc_components(
        self, timer: HierarchicalTimer | None = None
    ) -> LinearSolverResults:
        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        return res


class ImplicitUsingLocalSchurComplement(
    MPIBaseImplicitSchurComplementLinearSolver
):
    """Abstract class defining functionalities to form local Schur complements."""

    def __init__(
        self,
        subproblem_solvers: dict[int, LinearSolverInterface],
        pcg_options: PcgOptions,
        local_schur_complement_solvers: dict[int, LinearSolverInterface],
    ) -> None:
        super().__init__(subproblem_solvers, pcg_options)
        self.local_schur_complement_solvers: dict[int, LinearSolverInterface] = (
            local_schur_complement_solvers
        )
        self.local_schur_complements: dict[int, np.ndarray] = dict()

    def _get_sc_structure(self, block_matrix: MPIBlockMatrix, timer: HierarchicalTimer):
        pass

    def _form_local_schur_complements(self, timer: HierarchicalTimer):
        timer.start("form_local_SC")
        for ndx in self.local_block_indices:
            border_matrix: _BorderMatrix = self.border_matrices[ndx]
            local_sc_dim = border_matrix.num_nonzero_rows
            self.local_schur_complements[ndx] = np.zeros((local_sc_dim, local_sc_dim), dtype=np.double)
            A = border_matrix.csr
            Ar = border_matrix._get_reduced_matrix()
            _rhs = np.zeros(A.shape[1], dtype=np.double)
            solver = self.subproblem_solvers[ndx]
            for i, row_ndx in enumerate(border_matrix.nonzero_rows):
                timer.start("get_rhs")
                for indptr in range(A.indptr[row_ndx], A.indptr[row_ndx + 1]):
                    col = A.indices[indptr]
                    val = A.data[indptr]
                    _rhs[col] += val
                timer.stop("get_rhs")
                timer.start("block_back_solve")
                contribution = solver.do_back_solve(_rhs)
                timer.stop("block_back_solve")
                contribution = Ar.dot(contribution)
                self.local_schur_complements[ndx][i, :] -= contribution
                timer.start("get_rhs")
                for indptr in range(A.indptr[row_ndx], A.indptr[row_ndx + 1]):
                    col = A.indices[indptr]
                    val = A.data[indptr]
                    _rhs[col] -= val
                timer.stop("get_rhs")

        timer.start("Barrier")
        comm.Barrier()
        timer.stop("Barrier")
        timer.stop("form_local_SC")

    def _factorize_sc_components(self, timer: HierarchicalTimer) -> LinearSolverResults:
        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        timer.start("factor_SC")
        for ndx in self._local_block_indices_for_numeric_factorization:
            local_sc_coo = coo_matrix(self.local_schur_complements[ndx])
            timer.start("symbolic")
            sub_res = self.local_schur_complement_solvers[
                ndx
            ].do_symbolic_factorization(
                local_sc_coo, raise_on_error=False
            )
            _process_sub_results(res, sub_res)
            timer.stop("symbolic")
            if res.status not in {
                LinearSolverStatus.successful,
                LinearSolverStatus.warning,
            }:
                timer.stop("factor_SC")
                return res
            timer.start("numeric")
            sub_res = self.local_schur_complement_solvers[ndx].do_numeric_factorization(
                local_sc_coo, raise_on_error=False
            )
            _process_sub_results(res, sub_res)
            timer.stop("numeric")
        timer.stop("factor_SC")
        return res

    def _apply_preconditioner(
        self,
        r: np.ndarray,
    ) -> np.ndarray:
        result = np.zeros_like(r)
        for ndx in self.local_block_indices:
            r_local = r[self.border_matrices[ndx].nonzero_rows]
            x_local = self.local_schur_complement_solvers[ndx].do_back_solve(r_local)
            result[self.border_matrices[ndx].nonzero_rows] += x_local

        comm.Allreduce(MPI.IN_PLACE, result, op=MPI.SUM)
        # Note: we ignore this block (only non-zero if regularization was necessary),
        # and the entries in the diagonal are usually small.
        #diag_block = self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).diagonal()
        #result += r / diag_block

        return result


class BlockJacobiImplicitSchurComplementLinearSolver(ImplicitUsingLocalSchurComplement):
    """Implicit Schur Complement Linear Solver with Block-Jacobi Preconditioner."""
    def __init__(
        self,
        subproblem_solvers: dict[int, LinearSolverInterface],
        pcg_options: PcgOptions,
        local_schur_complement_solvers: dict[int, LinearSolverInterface],
    ) -> None:
        super().__init__(
            subproblem_solvers, pcg_options, local_schur_complement_solvers
        )

    def _form_sc_components(self, timer: HierarchicalTimer) -> None:
        self._form_local_schur_complements(timer)


class AdditiveSchwarzSchurComplementLinearSolver(ImplicitUsingLocalSchurComplement):
    """Implicit Schur Complement Linear Solver with Additive Schwarz Preconditioner."""
    def __init__(
        self,
        subproblem_solvers: dict[int, LinearSolverInterface],
        pcg_options: PcgOptions,
        local_schur_complement_solvers: dict[int, LinearSolverInterface],
    ) -> None:
        super().__init__(
            subproblem_solvers, pcg_options, local_schur_complement_solvers
        )
        self._local_windows = dict()
        self._windows_allocated = False
        self._neighboring_scs = dict()

    def _get_connectivity_info(self, timer: HierarchicalTimer):
        timer.start("get_connectivity_info")
        local_complicating_vars = np.unique(
            np.concatenate(
                [
                    self.border_matrices[ndx].nonzero_rows
                    for ndx in self.local_block_indices
                ]
            )
        ).ravel()
        local_var_indicator = np.zeros((self.sc_dim,), dtype=np.bool_)
        local_var_indicator[local_complicating_vars] = True
        global_var_indicator_mat = np.zeros((size, self.sc_dim), dtype=np.bool_)
        timer.start("Allgather")
        comm.Allgather(local_var_indicator, global_var_indicator_mat)
        timer.stop("Allgather")
        ranks_id, var_id = np.nonzero(global_var_indicator_mat)
        global_rank_to_var_map = dict()
        for grank in range(size):
            global_rank_to_var_map[grank] = var_id[ranks_id == grank]
        self.global_rank_to_var_map = global_rank_to_var_map

        timer.start("get_local_var_to_ranks_map")
        local_var_to_ranks_map = dict()
        neighboring_ranks = []
        for local_var in list(local_complicating_vars):
            connected_ranks = ranks_id[var_id == local_var]
            local_var_to_ranks_map[local_var] = connected_ranks
            neighboring_ranks.extend(connected_ranks)

        neighboring_ranks = list(np.unique(np.array(neighboring_ranks)))
        neighboring_ranks.remove(rank)
        timer.stop("get_local_var_to_ranks_map")
        # includes self rank
        timer.start("get_local_proc_to_shared_vars")
        local_proc_to_shared_vars = dict()
        for nrank in neighboring_ranks:
            local_proc_to_shared_vars[nrank] = np.intersect1d(
                var_id[ranks_id == nrank], local_complicating_vars
            )
        timer.stop("get_local_proc_to_shared_vars")

        self.local_var_to_ranks_map = local_var_to_ranks_map
        self.neighboring_ranks = neighboring_ranks
        self.neighboring_rank_to_shared_vars = local_proc_to_shared_vars
        timer.stop("get_connectivity_info")

        timer.start("get_ownership_map")
        ownership_map = np.diag(self.block_matrix.rank_ownership)[:-1]
        assert ownership_map.size == self.block_dim - 1
        self.ownership_map = ownership_map
        timer.stop("get_ownership_map")

        timer.start("create_groups")
        all_neighborring_ranks = comm.allreduce([(rank, neighboring_ranks)], op=MPI.SUM)
        all_neighborring_ranks = {r[0]: r[1] for r in all_neighborring_ranks}
        local_groups = {}
        local_comms = {}
        rel_local_group_ranks = np.empty(self.block_dim - 1, dtype=np.int64)
        for ndx in range(self.block_dim - 1):
            owning_rank = ownership_map[ndx]
            neighbors = all_neighborring_ranks[owning_rank]
            local_group_ranks = [*neighbors, owning_rank]
            local_groups[ndx] = comm.group.Incl(local_group_ranks)
            local_comms[ndx] = comm.Create_group(local_groups[ndx])
            rel_local_group_ranks[ndx] = local_comms[ndx].Get_rank()
        all_group_ranks = np.zeros((size, self.block_dim - 1), dtype=np.int64)
        comm.Allgather(rel_local_group_ranks, all_group_ranks)
        self.all_group_ranks = all_group_ranks
        self.local_comms = local_comms
        timer.stop("create_groups")

    def _get_sc_structure(
        self, block_matrix: MPIBlockMatrix, timer: HierarchicalTimer
    ) -> None:
        super()._get_sc_structure(block_matrix, timer)
        timer.start("get_connectivity_info")
        self._get_connectivity_info(timer)
        timer.stop("get_connectivity_info")

    def _create_windows(self, timer: HierarchicalTimer) -> None:
        timer.start("allocate_windows")
        local_windows = dict()
        datatype = MPI.DOUBLE
        itemsize = datatype.Get_size()
        for ndx in range(self.block_dim - 1):
            if ndx in self.local_block_indices:
                timer.start("allocate_local_window")
                local_sc_dim = self.border_matrices[ndx].num_nonzero_rows
                local_windows[ndx] = MPI.Win.Allocate(
                    size=local_sc_dim * local_sc_dim * itemsize,
                    disp_unit=itemsize,
                    comm=self.local_comms[ndx],
                )
                timer.stop("allocate_local_window")
            elif self.ownership_map[ndx] in self.neighboring_ranks:
                timer.start("allocate_empty_window")
                local_windows[ndx] = MPI.Win.Allocate(
                    size=0,
                    disp_unit=itemsize,
                    comm=self.local_comms[ndx],
                )
                timer.stop("allocate_empty_window")

            else:
                pass

        timer.start("Barrier")
        comm.Barrier()
        timer.stop("Barrier")
        self._local_windows = local_windows
        timer.stop("allocate_windows")

    def _put_windows(self, timer: HierarchicalTimer):
        timer.start("put_windows")
        datatype = MPI.DOUBLE
        np_dtype = dtlib.to_numpy_dtype(datatype)
        for ndx in self.local_block_indices:
            buf = self.local_schur_complements[ndx].astype(np_dtype).reshape(-1)
            rel_rank = self.all_group_ranks[rank, ndx]
            self._local_windows[ndx].Lock(rank=rel_rank)
            self._local_windows[ndx].Put(buf, target_rank=rel_rank)
            self._local_windows[ndx].Unlock(rank=rel_rank)

        timer.start("Barrier")
        comm.Barrier()
        timer.stop("Barrier")
        timer.stop("put_windows")

    def _get_windows(self, timer: HierarchicalTimer):
        timer.start("get_windows")
        datatype = MPI.DOUBLE
        np_dtype = dtlib.to_numpy_dtype(datatype)
        neighboring_scs = dict()
        for ndx in range(self.block_dim - 1):
            nrank = self.ownership_map[ndx]
            if nrank in self.neighboring_ranks:
                rel_nrank = self.all_group_ranks[nrank, ndx]
                n_sc_dim = len(self.global_rank_to_var_map[nrank])
                buf = np.empty((n_sc_dim, n_sc_dim), dtype=np_dtype).reshape(-1)
                n_sc_window = self._local_windows[ndx]
                n_sc_window.Lock(rel_nrank)
                n_sc_window.Get(buf, target_rank=rel_nrank)
                n_sc_window.Unlock(rel_nrank)
                neighboring_scs[nrank] = buf.reshape((n_sc_dim, n_sc_dim))

        timer.start("Barrier")
        comm.Barrier()
        timer.stop("Barrier")
        self._neighboring_scs = neighboring_scs
        timer.stop("get_windows")

    def _form_sc_components(self, timer: HierarchicalTimer) -> None:
        timer.start("form_SC")
        self._form_local_schur_complements(timer)
        # create mpi windows for local schur complements
        timer.start("communicate")
        if not self._windows_allocated:
            self._create_windows(timer)
            self._windows_allocated = True

        self._put_windows(timer)
        self._get_windows(timer)
        timer.stop("communicate")

        timer.start("assemble_local_SC")
        for ndx in self.local_block_indices:
            # also need to exchange overlap between local blocks

            # This exchanges overlaps between neighoring ranks
            for nrank in self.neighboring_ranks:
                local_sc_vars = self.border_matrices[ndx].nonzero_rows
                neighboring_sc_vars = self.global_rank_to_var_map[nrank]
                overlapping_vars_nmask = np.isin(neighboring_sc_vars, local_sc_vars)
                if not np.any(overlapping_vars_nmask):
                    continue
                overlapping_vars = neighboring_sc_vars[overlapping_vars_nmask]
                local_ov_idxs = []
                # TODO: can this be achieved without the loop?
                for ov in overlapping_vars:
                    local_ov_idxs.append(np.where(local_sc_vars == ov)[0][0])
                local_ov_idxs = np.array(local_ov_idxs, dtype=np.int64)
                self.local_schur_complements[ndx][np.ix_(local_ov_idxs, local_ov_idxs)] += self._neighboring_scs[
                    nrank
                ][np.ix_(overlapping_vars_nmask, overlapping_vars_nmask)]

        timer.start("Barrier")
        comm.Barrier()
        timer.stop("Barrier")
        timer.stop("assemble_local_SC")
        timer.stop("form_SC")


class DiagAdditiveSchwarzSchurComplementLinearSolver(ImplicitUsingLocalSchurComplement):
    """Implicit Schur Complement Linear Solver with Block-Jacobi Preconditioner + diagonal AS Overlap."""

    def __init__(
        self,
        subproblem_solvers: dict[int, LinearSolverInterface],
        pcg_options: PcgOptions,
        local_schur_complement_solvers: dict[int, LinearSolverInterface],
    ) -> None:
        super().__init__(
            subproblem_solvers, pcg_options, local_schur_complement_solvers
        )

    def _form_sc_components(self, timer: HierarchicalTimer) -> None:
        self._form_local_schur_complements(timer)
        timer.start("get_diag")
        sc_diag = np.zeros(self.sc_dim, dtype=np.double)
        for ndx in self._local_block_indices_for_numeric_factorization:
            border_matrix: _BorderMatrix = self.border_matrices[ndx]
            sc_diag[border_matrix.nonzero_rows] += np.diag(
                self.local_schur_complements[ndx]
            )
        timer.stop("get_diag")
        timer.start("communicate_diag")
        comm.Allreduce(MPI.IN_PLACE, sc_diag, op=MPI.SUM)
        timer.stop("communicate_diag")

        timer.start("set_global_diag")
        for ndx in self._local_block_indices_for_numeric_factorization:
            border_matrix: _BorderMatrix = self.border_matrices[ndx]
            local_sc_dim = border_matrix.num_nonzero_rows
            self.local_schur_complements[ndx][np.arange(local_sc_dim), np.arange(local_sc_dim)] = sc_diag[
                border_matrix.nonzero_rows
            ]
        timer.stop("set_global_diag")


def make_implicit_schur_complement_solver(
    preconditioner_type: ImplicitSchurComplementPreconditionerType,
    subproblem_solvers: dict[int, LinearSolverInterface],
    pcg_options: PcgOptions,
    preconditioner_options: dict | None = None,
) -> LinearSolverInterface:
    """Factory method to create implicit Schur complement linear solver."""
    if preconditioner_options is None:
        preconditioner_options = {}
    if (
        preconditioner_type
        == ImplicitSchurComplementPreconditionerType.incomplete_cholesky
    ):
        solver_cls = MPISpICholImplicitSchurComplementLinearSolver
    elif preconditioner_type == ImplicitSchurComplementPreconditionerType.incomplete_lu:
        solver_cls = MPISpILUImplicitSchurComplementLinearSolver
    elif preconditioner_type == ImplicitSchurComplementPreconditionerType.lbfgs:
        solver_cls = MPILbfgsImplicitSchurComplementLinearSolver
    elif (
        preconditioner_type
        == ImplicitSchurComplementPreconditionerType.adaptive_refactorization
    ):
        solver_cls = MPIAdaptiveImplicitSchurComplementLinearSolver
    elif preconditioner_type == ImplicitSchurComplementPreconditionerType.block_jacobi:
        solver_cls = BlockJacobiImplicitSchurComplementLinearSolver
    elif (
        preconditioner_type
        == ImplicitSchurComplementPreconditionerType.additive_schwarz
    ):
        solver_cls = AdditiveSchwarzSchurComplementLinearSolver
    elif (
        preconditioner_type
        == ImplicitSchurComplementPreconditionerType.diagonal_additive_schwarz
    ):
        solver_cls = DiagAdditiveSchwarzSchurComplementLinearSolver
    else:
        raise ValueError(
            f"Unknown implicit Schur complement preconditioner type: {preconditioner_type}"
        )
    try:
        solver = solver_cls(subproblem_solvers, pcg_options, **preconditioner_options)
    except TypeError as e:
        raise ValueError(
            f"Invalid arguments for {solver_cls.__name__}",
            "the necessary signature for __init__() is:",
            inspect.signature(solver_cls.__init__),
            "kwargs other than `subproblem_solvers` and `pcg_options` need to be included "
            "in `preconditioner_options` in `make_implicit_schur_complement_solver()`"
        ) from e
    return solver
