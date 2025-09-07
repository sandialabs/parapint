import enum
import importlib
import inspect
from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse.linalg
from mpi4py import MPI
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
from parapint.linalg.schur_complement.mpi_distributed_schur_complement import (
    AdditiveSchwarzSchurComplementLinearSolver,
    BlockJacobiImplicitSchurComplementLinearSolver,
    DiagAdditiveSchwarzSchurComplementLinearSolver,
)
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


class MPIBaseImplicitSchurComplementLinearSolver(ABC, LinearSolverInterface):
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
        return res

    def do_back_solve(
        self, rhs: MPIBlockVector, timer: HierarchicalTimer | None = None
    ) -> tuple[MPIBlockVector, PcgSolutionStatus]:
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
        self._current_pcg_solution = pcg_sol

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

        return result, pcg_sol.status

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


class ImplicitUsingFullSchurComplement(MPIBaseImplicitSchurComplementLinearSolver, ABC):
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
        self.schur_complement.data = scS
        timer.stop("communicate")
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
        scipy_spilu_kwargs: dict,
    ) -> None:
        super().__init__(subproblem_solvers=subproblem_solvers, pcg_options=pcg_options)
        self.precond_options = scipy_spilu_kwargs
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
        refactorization_iter_threshhold: int,
    ) -> None:
        self.pcg_options = pcg_options
        super().__init__(subproblem_solvers=subproblem_solvers, pcg_options=pcg_options)
        self._flag_form_sc = True
        self._flag_factorize_sc = True
        self._sc_solver: LinearSolverInterface = schur_complement_solver
        self._pcg_iter_threshhold = refactorization_iter_threshhold

    def _update_preconditioner(self, pcg_sol: PcgSolution):
        if pcg_sol.num_iterations > self._pcg_iter_threshhold:
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


def make_implicit_schur_complement_solver(
    preconditioner_type: ImplicitSchurComplementPreconditionerType,
    subproblem_solvers: dict[int, LinearSolverInterface],
    pcg_options: PcgOptions,
    preconditioner_options: dict,
) -> LinearSolverInterface:
    """Factory method to create implicit Schur complement linear solver."""
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
