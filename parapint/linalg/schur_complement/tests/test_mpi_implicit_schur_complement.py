import unittest
import parapint
import pytest
import parapint.linalg.iterative
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
from parapint.linalg import ScipyInterface
from scipy.sparse import coo_matrix, spdiags
import numpy as np
from mpi4py import MPI
from parapint.examples.stochastic import Problem, Farmer


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_stoch_test_system(compute_sc=False):
    interface = Problem(farmer=Farmer())
    interface.set_barrier_parameter(1e-1)
    interface.set_bounds_relaxation_factor(1e-1)
    kkt = interface.evaluate_primal_dual_kkt_matrix()
    rhs = interface.evaluate_primal_dual_kkt_rhs()
    kkt = interface.regularize_hessian(kkt, coef=1, copy_kkt=False)
    kkt = interface.regularize_equality_gradient(kkt, coef=-1, copy_kkt=False)
    if compute_sc:
        M = kkt.to_local_array()
        sc_dim = 3
        A = M[:-sc_dim, :-sc_dim]
        D = M[-sc_dim:, -sc_dim:]
        B = M[-sc_dim:, :-sc_dim]
        sc = D - B @ np.linalg.inv(A) @ B.T
    else:
        sc = None

    return kkt, rhs, sc


class TestSchurComplement(unittest.TestCase):
    def _test_schur_complements(self, solver, type):
        A, _, sc = get_stoch_test_system(compute_sc=True)
        sc_dim = sc.shape[0]
        res_s = solver.do_symbolic_factorization(A)
        res_n = solver.do_numeric_factorization(A)
        if type == "explicit":
            full_sc = solver.schur_complement.toarray()
            full_sc += A.get_block(3, 3).toarray()
            assert np.allclose(full_sc, sc)
        elif type == "implicit-full":
            full_sc = solver.schur_complement.toarray()
            assert np.allclose(full_sc, sc)
        elif type == "implicit-local-bj":
            full_sc = np.zeros_like(sc)
            for ndx in range(len(solver.subproblem_solvers)):
                nk = solver.border_matrices[ndx].selection_matrix
                local_sc_contrib = nk @ solver.local_schur_complements[ndx] @ nk.T
                full_sc += local_sc_contrib
            full_sc += A.get_block(3, 3).toarray()
            assert np.allclose(full_sc, sc)
        elif type == "implicit-local-asd":
            full_sc = np.zeros_like(sc)
            true_sc_diag = np.diagonal(sc - A.get_block(3, 3).toarray())
            for ndx in range(len(solver.subproblem_solvers)):
                nk = solver.border_matrices[ndx].selection_matrix
                local_sc = solver.local_schur_complements[ndx]
                assert np.allclose(np.diagonal(local_sc), true_sc_diag)

    def _test_linear_solver(self, solver):
        A, rhs, sc = get_stoch_test_system(compute_sc=True)
        local_A = A.to_local_array()
        local_rhs = rhs.make_local_copy().flatten()
        x1 = np.linalg.solve(local_A, local_rhs)
        res_s = solver.do_symbolic_factorization(A)
        res_n = solver.do_numeric_factorization(A)
        x2 = solver.do_back_solve(rhs)
        self.assertTrue(np.allclose(x1, x2.make_local_copy().flatten()))

    def _test_inertia(self, solver):
        A, _, _ = get_stoch_test_system(compute_sc=False)
        local_A = A.to_local_array()
        res_s = solver.do_symbolic_factorization(A)
        res_n = solver.do_numeric_factorization(A)
        inertia1 = solver.get_inertia()
        eig = np.linalg.eigvals(local_A)
        pos = np.count_nonzero(eig > 0)
        neg = np.count_nonzero(eig < 0)
        zero = np.count_nonzero(eig == 0)
        inertia2 = (pos, neg, zero)
        self.assertEqual(inertia1, inertia2)

    @pytest.mark.parallel
    @pytest.mark.fast
    @pytest.mark.all_proc
    def test_explicit_schur_complement(self):
        subproblem_solvers = {
            ndx: ScipyInterface(compute_inertia=True) for ndx in range(3)
        }
        schur_complement_solver = ScipyInterface(compute_inertia=True)
        solver = parapint.linalg.MPISchurComplementLinearSolver(
            subproblem_solvers=subproblem_solvers,
            schur_complement_solver=schur_complement_solver,
        )
        self._test_linear_solver(solver)
        self._test_inertia(solver)
        self._test_schur_complements(solver, type="explicit")

    @pytest.mark.parallel
    @pytest.mark.fast
    @pytest.mark.all_proc
    def test_implicit_schur_complement_adaptive(self):
        preconditioner_type = parapint.linalg.ImplicitSchurComplementPreconditionerType.adaptive_refactorization
        subproblem_solvers = {
            ndx: ScipyInterface(compute_inertia=True) for ndx in range(3)
        }
        pcg_options = parapint.linalg.iterative.PcgOptions()
        pcg_options.atol = 1e-10
        pcg_options.rtol = 1e-10
        preconditioner_options = {
            "schur_complement_solver": ScipyInterface(compute_inertia=True),
            "refactorization_iter_threshold": 0,  # to ensure refactorization for each test
        }
        solver = parapint.linalg.make_implicit_schur_complement_solver(
            preconditioner_type=preconditioner_type,
            subproblem_solvers=subproblem_solvers,
            pcg_options=pcg_options,
            preconditioner_options=preconditioner_options,
        )
        self._test_linear_solver(solver)
        self._test_inertia(solver)
        self._test_schur_complements(solver, type="implicit-full")

    @pytest.mark.parallel
    @pytest.mark.fast
    @pytest.mark.all_proc
    def test_implicit_schur_complement_spilu(self):
        preconditioner_type = (
            parapint.linalg.ImplicitSchurComplementPreconditionerType.incomplete_lu
        )
        subproblem_solvers = {
            ndx: ScipyInterface(compute_inertia=True) for ndx in range(3)
        }
        pcg_options = parapint.linalg.iterative.PcgOptions()
        solver = parapint.linalg.make_implicit_schur_complement_solver(
            preconditioner_type=preconditioner_type,
            subproblem_solvers=subproblem_solvers,
            pcg_options=pcg_options,
        )
        self._test_linear_solver(solver)
        self._test_inertia(solver)
        self._test_schur_complements(solver, type="implicit-full")

    @pytest.mark.parallel
    @pytest.mark.fast
    @pytest.mark.all_proc
    @unittest.skipIf(not parapint.linalg.ilupp_is_available(), reason="ilupp package is not available")
    def test_implicit_schur_complement_spich(self):
        preconditioner_type = parapint.linalg.ImplicitSchurComplementPreconditionerType.incomplete_cholesky
        subproblem_solvers = {
            ndx: ScipyInterface(compute_inertia=True) for ndx in range(3)
        }
        pcg_options = parapint.linalg.iterative.PcgOptions()
        preconditioner_options = {"drop_tol": 1e-4, "fill_factor": 10}
        solver = parapint.linalg.make_implicit_schur_complement_solver(
            preconditioner_type=preconditioner_type,
            subproblem_solvers=subproblem_solvers,
            pcg_options=pcg_options,
            preconditioner_options=preconditioner_options,
        )
        self._test_linear_solver(solver)
        self._test_inertia(solver)
        self._test_schur_complements(solver, type="implicit-full")

    @pytest.mark.parallel
    @pytest.mark.fast
    @pytest.mark.all_proc
    def test_implicit_schur_complement_lbfgs(self):
        preconditioner_type = (
            parapint.linalg.ImplicitSchurComplementPreconditionerType.lbfgs
        )
        subproblem_solvers = {
            ndx: ScipyInterface(compute_inertia=True) for ndx in range(3)
        }
        pcg_options = parapint.linalg.iterative.PcgOptions()
        pcg_options.lbfgs_approx_options.sampling = (
            parapint.linalg.iterative.LbfgsSamplingOptions.uniform
        )
        solver = parapint.linalg.make_implicit_schur_complement_solver(
            preconditioner_type=preconditioner_type,
            subproblem_solvers=subproblem_solvers,
            pcg_options=pcg_options,
        )
        self._test_linear_solver(solver)
        self._test_inertia(solver)

    @pytest.mark.parallel
    @pytest.mark.fast
    @pytest.mark.all_proc
    def test_implicit_schur_complement_block_jacobi(self):
        preconditioner_type = (
            parapint.linalg.ImplicitSchurComplementPreconditionerType.block_jacobi
        )
        subproblem_solvers = {
            ndx: ScipyInterface(compute_inertia=True) for ndx in range(3)
        }
        pcg_options = parapint.linalg.iterative.PcgOptions()
        local_sc_solvers = {
            ndx: ScipyInterface(compute_inertia=True) for ndx in range(3)
        }
        preconditioner_options = {
            "local_schur_complement_solvers": local_sc_solvers,
        }
        solver = parapint.linalg.make_implicit_schur_complement_solver(
            preconditioner_type=preconditioner_type,
            subproblem_solvers=subproblem_solvers,
            pcg_options=pcg_options,
            preconditioner_options=preconditioner_options,
        )
        self._test_linear_solver(solver)
        self._test_inertia(solver)
        self._test_schur_complements(solver, type="implicit-local-bj")

    @pytest.mark.parallel
    @pytest.mark.fast
    @pytest.mark.all_proc
    def test_implicit_schur_complement_additive_schwarz(self):
        preconditioner_type = (
            parapint.linalg.ImplicitSchurComplementPreconditionerType.additive_schwarz
        )
        subproblem_solvers = {
            ndx: ScipyInterface(compute_inertia=True) for ndx in range(3)
        }
        pcg_options = parapint.linalg.iterative.PcgOptions()
        local_sc_solvers = {
            ndx: ScipyInterface(compute_inertia=True) for ndx in range(3)
        }
        preconditioner_options = {
            "local_schur_complement_solvers": local_sc_solvers,
        }
        solver = parapint.linalg.make_implicit_schur_complement_solver(
            preconditioner_type=preconditioner_type,
            subproblem_solvers=subproblem_solvers,
            pcg_options=pcg_options,
            preconditioner_options=preconditioner_options,
        )
        self._test_linear_solver(solver)
        self._test_inertia(solver)
        # TODO: AS only works when using MPI with one block per rank
        # self._test_schur_complements(solver, type='implicit-local-as')

    @pytest.mark.parallel
    @pytest.mark.fast
    @pytest.mark.all_proc
    def test_implicit_schur_complement_diag_additive_schwarz(self):
        preconditioner_type = parapint.linalg.ImplicitSchurComplementPreconditionerType.diagonal_additive_schwarz
        subproblem_solvers = {
            ndx: ScipyInterface(compute_inertia=True) for ndx in range(3)
        }
        pcg_options = parapint.linalg.iterative.PcgOptions()
        local_sc_solvers = {
            ndx: ScipyInterface(compute_inertia=True) for ndx in range(3)
        }
        preconditioner_options = {
            "local_schur_complement_solvers": local_sc_solvers,
        }
        solver = parapint.linalg.make_implicit_schur_complement_solver(
            preconditioner_type=preconditioner_type,
            subproblem_solvers=subproblem_solvers,
            pcg_options=pcg_options,
            preconditioner_options=preconditioner_options,
        )
        self._test_linear_solver(solver)
        self._test_inertia(solver)
        self._test_schur_complements(solver, type="implicit-local-asd")
