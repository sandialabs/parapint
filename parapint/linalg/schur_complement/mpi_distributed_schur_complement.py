from abc import ABC

import numpy as np
from mpi4py import MPI
from mpi4py.util import dtlib
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix

from parapint.linalg.base_linear_solver_interface import LinearSolverInterface
from parapint.linalg.iterative.pcg import PcgOptions
from parapint.linalg.results import LinearSolverResults, LinearSolverStatus
from parapint.linalg.schur_complement.mpi_implicit_schur_complement import (
    MPIBaseImplicitSchurComplementLinearSolver,
)
from parapint.linalg.schur_complement.utils import _BorderMatrix, _process_sub_results

comm: MPI.Comm = MPI.COMM_WORLD
rank: int = comm.Get_rank()
size: int = comm.Get_size()


class ImplicitUsingLocalSchurComplement(
    MPIBaseImplicitSchurComplementLinearSolver, ABC
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
            timer.start("symbolic")
            sub_res = self.local_schur_complement_solvers[
                ndx
            ].do_symbolic_factorization(
                self.local_schur_complements[ndx], raise_on_error=False
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
                self.local_schur_complements[ndx], raise_on_error=False
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
        timer.stop("form_SC")
