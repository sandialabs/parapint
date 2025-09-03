import importlib
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
from parapint.linalg.schur_complement.utils import (
    _BorderMatrix,
    _gather_results,
    _get_all_nonzero_elements_in_sc_using_ix,
    _process_sub_results,
)

comm: MPI.Comm = MPI.COMM_WORLD
rank: int = comm.Get_rank()
size: int = comm.Get_size()


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
        # res = np.zeros(sc_dim, dtype='d')
        # comm.Allreduce(schur_complement_rhs, res)
        # schur_complement_rhs = rhs.get_block(self.block_dim - 1) + res
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
        #  - however it saves time in iterations later, as sparsity is already computed - can we do this more efficiently (vectorize instead of loop)?
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
        # self.schur_complement = self.schur_complement + self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()
        # self._current_schur_complement = self.schur_complement + self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()
        timer.stop("communicate")
        timer.stop("form_SC")


class MPISpICholImplicitSchurComplementLinearSolver(ImplicitUsingFullSchurComplement):
    """Implicit Schur complement linear solver using sparse incomplete Cholesky preconditioner."""

    def __init__(
        self, subproblem_solvers: dict[int, LinearSolverInterface], options: dict
    ) -> None:
        pcg_options = options.get("pcg_options", PcgOptions())
        super().__init__(subproblem_solvers=subproblem_solvers, pcg_options=pcg_options)
        self.precond_options = options.get("precond_options", dict())

        try:
            self._ilupp = importlib.import_module("ilupp")
        except ImportError as e:
            raise ImportError(
                "Make sure the ilupp package is installed "
                "when using the incomplete Cholesky preconditioner"
            ) from e
        self.drop_tol = self.precond_options.get("drop_tol", 1e-4)
        self.fill_factor = self.precond_options.get("fill_factor", 10)
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
        self, subproblem_solvers: dict[int, LinearSolverInterface], options: dict
    ) -> None:
        pcg_options = options.get("pcg_options", PcgOptions())
        super().__init__(subproblem_solvers=subproblem_solvers, pcg_options=pcg_options)
        self.precond_options = options.get("precond_options", dict())
        self._spilu_precond = None

    def _apply_preconditioner(self, r: np.ndarray) -> np.ndarray:
        return self._spilu_precond.solve(r)

    def _factorize_sc_components(self, timer: HierarchicalTimer) -> LinearSolverResults:
        timer.start("SpILU_SC")
        # options = dict(IterRefine="SINGLE")
        # TODO: might want to unpack precond_options for same argument names as spilu
        self._spilu_precond = scipy.sparse.linalg.spilu(
            self.schur_complement.tocsc(), options=self.precond_options
        )
        timer.stop("SpILU_SC")
        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        return res


class MPIAdaptiveImplicitSchurComplementLinearSolver(ImplicitUsingFullSchurComplement):
    """Implicit Schur complement linear solver using adaptive refactorization as preconditioner."""

    def __init__(
        self, subproblem_solvers: dict[int, LinearSolverInterface], options: dict
    ) -> None:
        pcg_options = options.get("pcg_options", PcgOptions())
        super().__init__(subproblem_solvers=subproblem_solvers, pcg_options=pcg_options)
        self.precond_options = options.get("precond_options", dict())
        self._flag_form_sc = True
        self._flag_factorize_sc = True
        self._sc_solver: LinearSolverInterface = self.precond_options[
            "schur_complement_solver"
        ]
        self._pcg_iter_threshhold = self.precond_options[
            "refactorization_iter_threshhold"
        ]

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
        self, subproblem_solvers: dict[int, LinearSolverInterface], options: dict
    ) -> None:
        pcg_options: PcgOptions = options.get("pcg_options", PcgOptions())
        assert (
            pcg_options.lbfgs_approx_options.sampling != LbfgsSamplingOptions.disable
        ), (
            "LBFGS sampling must be enabled to use Adaptive Refactorization Preconditioner."
        )
        super().__init__(subproblem_solvers=subproblem_solvers, pcg_options=pcg_options)

        self.precond_options = options.get("precond_options", dict())
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


# class _MPISchurComplementUtilMixin:
#     def __init__(self, subproblem_solvers: dict[int, LinearSolverInterface]):
#         self.subproblem_solvers: dict[int, LinearSolverInterface] = subproblem_solvers
#         self.block_dim = 0
#         self.block_matrix: MPIBlockMatrix = None
#         self.local_block_indices = list()
#         self._local_block_indices_for_numeric_factorization = list()
#         self.schur_complement = coo_matrix((0, 0))
#         # self._current_schur_complement = coo_matrix((0, 0))
#         self.border_matrices: Dict[int, _BorderMatrix] = dict()
#         self.sc_data_slices = dict()
#         self.sc_dim: int = 0
#         self.local_var_to_ranks_map = dict()
#         self.neighboring_ranks = []
#         self.neighboring_rank_to_shared_vars = dict()
#         self.global_rank_to_var_map = dict()
#         self.local_comms = dict()
#         self.local_group_ranks = dict()
#         self.all_group_ranks: np.ndarray = None
#         self.ownership_map: np.ndarray = None

#     def _get_connectivity_info(self, timer: HierarchicalTimer):
#         timer.start("get_connectivity_info")
#         local_complicating_vars = np.unique(
#             np.concatenate(
#                 [
#                     self.border_matrices[ndx].nonzero_rows
#                     for ndx in self.local_block_indices
#                 ]
#             )
#         ).ravel()
#         local_var_indicator = np.zeros((self.sc_dim,), dtype=np.bool_)
#         local_var_indicator[local_complicating_vars] = True
#         global_var_indicator_mat = np.zeros((size, self.sc_dim), dtype=np.bool_)
#         timer.start("Allgather")
#         comm.Allgather(local_var_indicator, global_var_indicator_mat)
#         timer.stop("Allgather")
#         ranks_id, var_id = np.nonzero(global_var_indicator_mat)
#         global_rank_to_var_map = dict()
#         for grank in range(size):
#             global_rank_to_var_map[grank] = var_id[ranks_id == grank]
#         self.global_rank_to_var_map = global_rank_to_var_map

#         timer.start("get_local_var_to_ranks_map")
#         local_var_to_ranks_map = dict()
#         neighboring_ranks = []
#         for local_var in list(local_complicating_vars):
#             connected_ranks = ranks_id[var_id == local_var]
#             local_var_to_ranks_map[local_var] = connected_ranks
#             neighboring_ranks.extend(connected_ranks)

#         neighboring_ranks = list(np.unique(np.array(neighboring_ranks)))
#         neighboring_ranks.remove(rank)
#         timer.stop("get_local_var_to_ranks_map")
#         # includes self rank
#         timer.start("get_local_proc_to_shared_vars")
#         local_proc_to_shared_vars = dict()
#         for nrank in neighboring_ranks:
#             local_proc_to_shared_vars[nrank] = np.intersect1d(
#                 var_id[ranks_id == nrank], local_complicating_vars
#             )
#         timer.stop("get_local_proc_to_shared_vars")

#         self.local_var_to_ranks_map = local_var_to_ranks_map
#         self.neighboring_ranks = neighboring_ranks
#         self.neighboring_rank_to_shared_vars = local_proc_to_shared_vars
#         timer.stop("get_connectivity_info")

#         timer.start("get_ownership_map")
#         ownership_map = np.diag(self.block_matrix.rank_ownership)[:-1]
#         assert ownership_map.size == self.block_dim - 1
#         self.ownership_map = ownership_map
#         timer.stop("get_ownership_map")

#         timer.start("create_groups")
#         all_neighborring_ranks = comm.allreduce([(rank, neighboring_ranks)], op=MPI.SUM)
#         all_neighborring_ranks = {r[0]: r[1] for r in all_neighborring_ranks}
#         local_groups = {}
#         local_comms = {}
#         rel_local_group_ranks = np.empty(self.block_dim - 1, dtype=np.int64)
#         for ndx in range(self.block_dim - 1):
#             owning_rank = ownership_map[ndx]
#             neighbors = all_neighborring_ranks[owning_rank]
#             local_group_ranks = [*neighbors, owning_rank]
#             local_groups[ndx] = comm.group.Incl(local_group_ranks)
#             local_comms[ndx] = comm.Create_group(local_groups[ndx])
#             rel_local_group_ranks[ndx] = local_comms[ndx].Get_rank()
#         all_group_ranks = np.zeros((size, self.block_dim - 1), dtype=np.int64)
#         comm.Allgather(rel_local_group_ranks, all_group_ranks)
#         self.all_group_ranks = all_group_ranks
#         self.local_comms = local_comms
#         timer.stop("create_groups")

#     def _get_full_sc_structure(
#         self, block_matrix: MPIBlockMatrix, timer: HierarchicalTimer
#     ):
#         """
#         Parameters
#         ----------
#         block_matrix: pyomo.contrib.pynumero.sparse.mpi_block_matrix.MPIBlockMatrix
#         """
#         timer.start("build_border_matrices")
#         self.border_matrices = dict()
#         for ndx in self.local_block_indices:
#             self.border_matrices[ndx] = _BorderMatrix(
#                 block_matrix.get_block(self.block_dim - 1, ndx)
#             )
#         timer.stop("build_border_matrices")
#         timer.start("gather_all_nonzero_elements")
#         nonzero_rows, nonzero_cols = _get_all_nonzero_elements_in_sc_using_ix(
#             self.border_matrices, self.local_block_indices, self.block_dim - 1
#         )
#         timer.stop("gather_all_nonzero_elements")
#         timer.start("construct_schur_complement")
#         sc_nnz = nonzero_rows.size
#         sc_dim = block_matrix.get_row_size(self.block_dim - 1)
#         self.sc_dim = sc_dim
#         sc_values = np.zeros(sc_nnz, dtype=np.double)
#         self.schur_complement = coo_matrix(
#             (sc_values, (nonzero_rows, nonzero_cols)), shape=(sc_dim, sc_dim)
#         )
#         timer.stop("construct_schur_complement")
#         timer.start("get_sc_data_slices")
#         # TODO: This part can be very slow for problems with many compl. cars per partition
#         #  - however it saves time in iterations later, as sparsity is already computed - can we do this more efficiently (vectorize instead of loop)?
#         self.sc_data_slices = dict()
#         for ndx in self.local_block_indices:
#             self.sc_data_slices[ndx] = dict()
#             border_matrix = self.border_matrices[ndx]
#             for row_ndx in border_matrix.nonzero_rows:
#                 self.sc_data_slices[ndx][row_ndx] = np.bitwise_and(
#                     nonzero_cols == row_ndx,
#                     np.isin(nonzero_rows, border_matrix.nonzero_rows),
#                 ).nonzero()[0]
#         timer.stop("get_sc_data_slices")

#     def _form_full_sc(self, timer) -> None:
#         timer.start("form_SC")
#         self.schur_complement.data = np.zeros(
#             self.schur_complement.data.size, dtype=np.double
#         )
#         for ndx in self.local_block_indices:
#             border_matrix: _BorderMatrix = self.border_matrices[ndx]
#             A = border_matrix.csr
#             _rhs = np.zeros(A.shape[1], dtype=np.double)
#             solver = self.subproblem_solvers[ndx]
#             for row_ndx in border_matrix.nonzero_rows:
#                 for indptr in range(A.indptr[row_ndx], A.indptr[row_ndx + 1]):
#                     col = A.indices[indptr]
#                     val = A.data[indptr]
#                     _rhs[col] += val
#                 timer.start("back_solve")
#                 contribution = solver.do_back_solve(_rhs)
#                 timer.stop("back_solve")
#                 timer.start("dot_product")
#                 contribution = A.dot(contribution)
#                 timer.stop("dot_product")
#                 self.schur_complement.data[self.sc_data_slices[ndx][row_ndx]] -= (
#                     contribution[border_matrix.nonzero_rows]
#                 )
#                 for indptr in range(A.indptr[row_ndx], A.indptr[row_ndx + 1]):
#                     col = A.indices[indptr]
#                     val = A.data[indptr]
#                     _rhs[col] -= val

#         timer.start("communicate")
#         sc = np.zeros(self.schur_complement.data.size, dtype=np.double)
#         timer.start("Barrier")
#         comm.Barrier()
#         timer.stop("Barrier")
#         timer.start("Allreduce")
#         comm.Allreduce(self.schur_complement.data, sc)
#         timer.stop("Allreduce")
#         self.schur_complement.data = sc
#         # self.schur_complement = self.schur_complement + self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()
#         # self._current_schur_complement = self.schur_complement + self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()
#         timer.stop("communicate")
#         timer.stop("form_SC")

#     def _symbolic_factorize_diag_blocks(
#         self, block_matrix: MPIBlockMatrix, timer: HierarchicalTimer
#     ) -> LinearSolverResults:
#         res = LinearSolverResults()
#         res.status = LinearSolverStatus.successful
#         timer.start("factorize_diag_blocks")
#         for ndx in self.local_block_indices:
#             sub_res = self.subproblem_solvers[ndx].do_symbolic_factorization(
#                 matrix=block_matrix.get_block(ndx, ndx), raise_on_error=False
#             )
#             _process_sub_results(res, sub_res)
#             if res.status not in {
#                 LinearSolverStatus.successful,
#                 LinearSolverStatus.warning,
#             }:
#                 break
#         timer.stop("factorize_diag_blocks")
#         res = _gather_results(res)
#         return res

#     def _numeric_factorize_diag_blocks(
#         self, block_matrix: MPIBlockMatrix, timer: HierarchicalTimer
#     ) -> LinearSolverResults:
#         res = LinearSolverResults()
#         res.status = LinearSolverStatus.successful

#         for ndx in self._local_block_indices_for_numeric_factorization:
#             timer.start("factorize_diag_blocks")
#             sub_res = self.subproblem_solvers[ndx].do_numeric_factorization(
#                 matrix=block_matrix.get_block(ndx, ndx), raise_on_error=False
#             )
#             timer.stop("factorize_diag_blocks")
#             _process_sub_results(res, sub_res)
#             if res.status not in {
#                 LinearSolverStatus.successful,
#                 LinearSolverStatus.warning,
#             }:
#                 break
#         res = _gather_results(res)
#         return res

#     def _symbolic_numeric_factorize_sc(
#         self, sc, sc_solver: LinearSolverInterface, timer
#     ) -> LinearSolverResults:
#         timer.start("factor_SC")
#         res = sc_solver.do_symbolic_factorization(sc, raise_on_error=False)
#         if res.status not in {
#             LinearSolverStatus.successful,
#             LinearSolverStatus.warning,
#         }:
#             timer.stop("factor_SC")
#             return res
#         sub_res = sc_solver.do_numeric_factorization(sc)
#         _process_sub_results(res, sub_res)
#         timer.stop("factor_SC")
#         return res


# class _MPIBaseImplicitSchurComplementLinearSolver(
#     LinearSolverInterface, _MPISchurComplementUtilMixin
# ):

#     def __init__(
#         self, subproblem_solvers: Dict[int, LinearSolverInterface], options: Dict
#     ):
#         super().__init__(subproblem_solvers=subproblem_solvers)

#         self.pcg_options: PcgOptions = options["pcg"]
#         self.precond_options: Dict = options["preconditioner"]
#         self._flag_form_sc = False
#         self._flag_factorize_sc = False
#         self._current_pcg_solution: PcgSolution = None
#         self.local_var_indices = []

#     def do_symbolic_factorization(
#         self,
#         matrix: MPIBlockMatrix,
#         raise_on_error: bool = True,
#         timer: Optional[HierarchicalTimer] = None,
#     ) -> LinearSolverResults:
#         if timer is None:
#             timer = HierarchicalTimer()

#         self.block_matrix = block_matrix = matrix
#         nbrows, nbcols = block_matrix.bshape
#         if nbrows != nbcols:
#             raise ValueError("The block matrix provided is not square.")
#         self.block_dim = nbrows

#         # split up the blocks between ranks
#         self.local_block_indices = list()
#         for ndx in range(self.block_dim - 1):
#             if (block_matrix.rank_ownership[ndx, ndx] == rank) or (
#                 block_matrix.rank_ownership[ndx, ndx] == -1 and rank == 0
#             ):
#                 self.local_block_indices.append(ndx)

#         res = self._symbolic_factorize_diag_blocks(block_matrix, timer)

#         if res.status not in {
#             LinearSolverStatus.successful,
#             LinearSolverStatus.warning,
#         }:
#             if raise_on_error:
#                 raise RuntimeError(
#                     "Symbolic factorization unsuccessful; status: " + str(res.status)
#                 )
#             else:
#                 return res

#         self._get_sc_structure(block_matrix, timer)

#         return res

#     def _get_sc_structure(
#         self, block_matrix: MPIBlockMatrix, timer: HierarchicalTimer
#     ) -> None:
#         if self._flag_form_sc:
#             timer.start("sc_structure")
#             self._get_full_sc_structure(block_matrix=block_matrix, timer=timer)
#             timer.stop("sc_structure")
#         else:
#             # always need border matrices
#             timer.start("build_border_matrices")
#             self.border_matrices = dict()
#             for ndx in self.local_block_indices:
#                 self.border_matrices[ndx] = _BorderMatrix(
#                     block_matrix.get_block(self.block_dim - 1, ndx)
#                 )
#                 self.local_var_indices.extend(
#                     self.border_matrices[ndx].nonzero_rows.tolist()
#                 )
#             self.local_var_indices = sorted(list(set(self.local_var_indices)))
#             timer.stop("build_border_matrices")

#     def _form_sc_components(self, timer: HierarchicalTimer) -> None:
#         if self._flag_form_sc:
#             self._form_full_sc(timer)

#     def _factorize_sc_components(self, timer: HierarchicalTimer) -> LinearSolverResults:
#         if self._flag_factorize_sc:
#             res = self._symbolic_numeric_factorize_sc(
#                 self.schur_complement, self.sc_solver, timer
#             )
#         else:
#             res = LinearSolverResults()
#             res.status = LinearSolverStatus.successful
#         return res

#     def do_numeric_factorization(
#         self,
#         matrix: MPIBlockMatrix,
#         raise_on_error: bool = True,
#         timer: Optional[HierarchicalTimer] = None,
#     ) -> LinearSolverResults:
#         """
#         Perform numeric factorization:
#           * perform numeric factorization on each diagonal block
#           * form and communicate the Schur-Complement
#           * factorize the schur-complement

#         This method should only be called after do_symbolic_factorization.

#         Parameters
#         ----------
#         matrix: MPIBlockMatrix
#             A Pynumero MPIBlockMatrix. This is the A matrix in Ax=b
#         raise_on_error: bool
#             If False, an error will not be raised if an error occurs during symbolic factorization. Instead the
#             status attribute of the results object will indicate an error ocurred.
#         timer: HierarchicalTimer
#             A timer for profiling.

#         Returns
#         -------
#         res: LinearSolverResults
#             The results object
#         """
#         if timer is None:
#             timer = HierarchicalTimer()

#         self.block_matrix = block_matrix = matrix

#         # factorize all local blocks
#         self._local_block_indices_for_numeric_factorization = self.local_block_indices
#         res = self._numeric_factorize_diag_blocks(block_matrix, timer)

#         if res.status not in {
#             LinearSolverStatus.successful,
#             LinearSolverStatus.warning,
#         }:
#             if raise_on_error:
#                 raise RuntimeError(
#                     "Numeric factorization unsuccessful; status: " + str(res.status)
#                 )
#             else:
#                 return res

#         self._form_sc_components(timer)

#         sub_res = self._factorize_sc_components(timer)

#         _process_sub_results(res, sub_res)

#         if res.status not in {
#             LinearSolverStatus.successful,
#             LinearSolverStatus.warning,
#         }:
#             if raise_on_error:
#                 raise RuntimeError(
#                     "Symbolic factorization unsuccessful; status: " + str(res.status)
#                 )

#         return res

#     def do_back_solve(
#         self, rhs, timer=None
#     ) -> Tuple[MPIBlockVector, PcgSolutionStatus]:
#         """
#         Performs a back solve with the factorized matrix. Should only be called after
#         do_numeric_factorixation.

#         Parameters
#         ----------
#         rhs: MPIBlockVector
#         timer: HierarchicalTimer

#         Returns
#         -------
#         result: MPIBlockVector
#         """
#         if timer is None:
#             timer = HierarchicalTimer()
#         timer.start("form_rhs_SC")
#         sc_dim = rhs.get_block(self.block_dim - 1).size
#         schur_complement_rhs = np.zeros(sc_dim, dtype="d")
#         for ndx in self.local_block_indices:
#             A = self.block_matrix.get_block(self.block_dim - 1, ndx)
#             contribution = self.subproblem_solvers[ndx].do_back_solve(
#                 rhs.get_block(ndx)
#             )
#             schur_complement_rhs -= A.tocsr().dot(contribution.flatten())
#         # res = np.zeros(sc_dim, dtype='d')
#         # comm.Allreduce(schur_complement_rhs, res)
#         # schur_complement_rhs = rhs.get_block(self.block_dim - 1) + res
#         comm.Allreduce(MPI.IN_PLACE, schur_complement_rhs, op=MPI.SUM)
#         schur_complement_rhs += rhs.get_block(self.block_dim - 1)

#         timer.stop("form_rhs_SC")

#         def _S(u):
#             timer.start("S_matvec")
#             r = self._sc_matvec(u, timer)
#             timer.stop("S_matvec")
#             return r

#         def _M(r):
#             timer.start("M_matvec")
#             u = self._apply_preconditioner(r, timer)
#             timer.stop("M_matvec")
#             return u

#         SC_linop = scipy.sparse.linalg.LinearOperator(
#             shape=(sc_dim, sc_dim), matvec=_S, dtype="d"
#         )
#         M_linop = scipy.sparse.linalg.LinearOperator(
#             shape=(sc_dim, sc_dim), matvec=_M, dtype="d"
#         )

#         if self.pcg_options.lbfgs_approx_options.distributed:
#             local_var_indices = self.local_var_indices
#         else:
#             local_var_indices = None
#         timer.start("pcg")
#         pcg_sol: PcgSolution = pcg_solve(
#             A=SC_linop,
#             b=schur_complement_rhs,
#             M=M_linop,
#             pcg_options=self.pcg_options,
#             local_var_indices=local_var_indices,
#         )
#         timer.stop("pcg")
#         coupling = pcg_sol.x
#         self._current_pcg_solution = pcg_sol
#         # if rank == 0:
#         #     print('# PCG iterations: ', pcg_sol.num_iterations)
#         self._update_preconditioner(pcg_sol)

#         timer.start("local_back_solve")
#         result = rhs.copy_structure()
#         for ndx in self.local_block_indices:
#             A = self.block_matrix.get_block(self.block_dim - 1, ndx)
#             result.set_block(
#                 ndx,
#                 self.subproblem_solvers[ndx].do_back_solve(
#                     rhs.get_block(ndx) - A.tocsr().transpose().dot(coupling.flatten())
#                 ),
#             )
#         timer.stop("local_back_solve")
#         result.set_block(self.block_dim - 1, coupling)

#         return result, pcg_sol.status

#     def get_inertia(self):
#         """
#         Get the inertia. Should only be called after do_numeric_factorization.

#         Returns
#         -------
#         num_pos: int
#             The number of positive eigenvalues of A
#         num_neg: int
#             The number of negative eigenvalues of A
#         num_zero: int
#             The number of zero eigenvalues of A
#         """
#         num_pos = 0
#         num_neg = 0
#         num_zero = 0

#         for ndx in self.local_block_indices:
#             _pos, _neg, _zero = self.subproblem_solvers[ndx].get_inertia()
#             num_pos += _pos
#             num_neg += _neg
#             num_zero += _zero

#         num_pos = comm.allreduce(num_pos)
#         num_neg = comm.allreduce(num_neg)
#         num_zero = comm.allreduce(num_zero)

#         # _pos, _neg, _zero = self.schur_complement_solver.get_inertia()
#         # num_pos += _pos
#         # num_neg += _neg
#         # num_zero += _zero

#         return num_pos, num_neg, num_zero

#     def increase_memory_allocation(self, factor):
#         """
#         Increases the memory allocation of each sub-solver. This method should only be called
#         if the results status from do_symbolic_factorization or do_numeric_factorization is
#         LinearSolverStatus.not_enough_memory.

#         Parameters
#         ----------
#         factor: float
#             The factor by which to increase memory allocation. Should be greater than 1.
#         """
#         for ndx in self.local_block_indices:
#             sub_solver = self.subproblem_solvers[ndx]
#             sub_solver.increase_memory_allocation(factor=factor)

#     def _sc_matvec(self, u: NDArray, timer) -> NDArray:
#         res = np.zeros_like(u)
#         for ndx in self.local_block_indices:
#             border_matrix: _BorderMatrix = self.border_matrices[ndx]
#             A = border_matrix.csr.transpose()
#             timer.start("dot_product")
#             v = A.dot(u)
#             timer.stop("dot_product")
#             timer.start("block_back_solve")
#             x = self.subproblem_solvers[ndx].do_back_solve(v)
#             timer.stop("block_back_solve")
#             timer.start("dot_product")
#             y = A.transpose().dot(x)
#             timer.stop("dot_product")
#             res -= y
#         # timer.start('communicate')
#         # res_global = np.empty(res.size)
#         # comm.Allreduce(res, res_global)
#         # timer.stop('communicate')
#         timer.start("communication")
#         comm.Allreduce(MPI.IN_PLACE, res, op=MPI.SUM)
#         timer.stop("communication")
#         # res_global += (self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()).dot(u)
#         return res

#     def _update_preconditioner(self, pcg_sol: PcgSolution):
#         raise NotImplementedError("This method should be implemented in a subclass")

#     def _apply_preconditioner(self, r: NDArray, timer=None) -> NDArray:
#         raise NotImplementedError("This method should be implemented in a subclass")

#     @property
#     def sc_solver(self) -> LinearSolverInterface:
#         raise NotImplementedError("This method should be implemented in a subclass")


# # TODO: Somewhat half-baked idea of collecting lbfgs approximations locally and constructing overall approx. with correct sparsity
# # As of now, collects all local variables, not partitioned by local block indices - as this is easier to realize in pcg
# # Check Nocedal & Wright, Sec. 7.4 for more details on this.
# class MPIDistributedLbfgsImplicitSchurComplementLinearSolver(
#     MPIBaseImplicitSchurComplementLinearSolver
# ):
#     """Implicit Schur complement linear solver using lbfgs approximation as preconditioner."""
#     def __init__(
#         self, subproblem_solvers: dict[int, LinearSolverInterface], options: dict
#     ) -> None:
#         pcg_options: PcgOptions = options.get("pcg_options", PcgOptions())
#         assert pcg_options.lbfgs_approx_options.sampling != LbfgsSamplingOptions.disable, \
#             "LBFGS sampling must be enabled to use Adaptive Refactorization Preconditioner."
#         super().__init__(subproblem_solvers=subproblem_solvers, pcg_options=pcg_options)

#         self.precond_options = options.get("precond_options", dict())
#         self._current_pcg_solution: PcgSolution | None = None

#     def _update_preconditioner(self, pcg_sol: PcgSolution):
#         self._current_pcg_solution = pcg_sol

#     def _apply_preconditioner(self, r: np.ndarray) -> np.ndarray:
#         if self._current_pcg_solution is None:
#             # No prev solution available
#             return r
#         else:
#              # TODO: assembly matrices should only be constructed once - eventually this should use selection matrices
#             sc_dim = len(r)
#             local_len = len(self.local_var_indices)
#             data = np.ones(local_len, dtype=np.int64)
#             row_idx = self.local_var_indices
#             col_idx = np.arange(local_len)  # Note: assumes linear ordering
#             coo_n = coo_matrix(
#                 (data, (row_idx, col_idx)), shape=(sc_dim, local_len)
#             ).toarray()
#             local_v = coo_n.dot(
#                 self._current_pcg_solution.hess_approx.dot(coo_n.T.dot(r))
#             )
#             # TODO: Use Allreduce
#             global_v = comm.allreduce(local_v)
#             return global_v

#     def _get_sc_structure(self,
#                           block_matrix: MPIBlockMatrix,
#                           timer: HierarchicalTimer | None = None
#                           ) -> None:
#         pass

#     def _form_sc_components(self, timer: HierarchicalTimer | None = None) -> None:
#         pass

#     def _factorize_sc_components(self, timer: HierarchicalTimer | None = None) -> LinearSolverResults:
#         res = LinearSolverResults()
#         res.status = LinearSolverStatus.successful
#         return res


# TODO: Factory method for different types of implicit schur complement solvers
