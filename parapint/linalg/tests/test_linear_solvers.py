import unittest
from pyomo.common.dependencies import attempt_import
import numpy as np
import scipy.sparse as sps
from scipy.sparse import coo_matrix, tril
import parapint
import pytest
import os
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
from parapint.linalg.mkl_pardiso import MKLPardisoInterface

ma27_available = MA27Interface.available()
mumps, mumps_available = attempt_import('mumps')
mkl_pardiso_available = MKLPardisoInterface.available()


def get_base_matrix(use_tril):
    if use_tril:
        row = [0, 1, 1, 2, 2]
        col = [0, 0, 1, 0, 2]
        data = [1, 7, 4, 3, 6]
    else:
        row = [0, 0, 0, 1, 1, 2, 2]
        col = [0, 1, 2, 0, 1, 0, 2]
        data = [1, 7, 3, 7, 4, 3, 6]
    mat = coo_matrix((data, (row, col)), shape=(3,3), dtype=np.double)
    return mat


def get_base_matrix_wrong_order(use_tril):
    if use_tril:
        row = [1, 0, 1, 2, 2]
        col = [0, 0, 1, 0, 2]
        data = [7, 1, 4, 3, 6]
    else:
        row = [1, 0, 0, 0, 1, 2, 2]
        col = [0, 1, 2, 0, 1, 0, 2]
        data = [7, 7, 3, 1, 4, 3, 6]
    mat = coo_matrix((data, (row, col)), shape=(3,3), dtype=np.double)
    return mat

def get_schur_complement_matrices(n=10, m=5, seed=0):
    # define invertible matrix with structure
    # [A   B]
    # [B^T D]
    # and return the Schur complement
    rng = np.random.default_rng(seed)
    H = sps.random(n, n, density=0.3, random_state=rng)
    H = H + H.T + 10 * np.eye(n)
    B = sps.random(n, m, density=0.3, random_state=rng).todense() + np.eye(N=m, M=n).T
    A = np.zeros((n + m, n + m))
    D = 0.1 * np.eye(m)
    A[:n, :n] = H
    A[:n, n:] = B.A
    A[n:, :n] = B.A.T
    A[n:, n:] = D
    
    S = D - B.T @ np.linalg.inv(H) @ B

    A = coo_matrix(A)
    H = coo_matrix(H)
    S = coo_matrix(S)
    return A, H, S

class TestTrilBehavior(unittest.TestCase):
    """
    Some of the other tests in this file depend on
    the behavior of tril that is tested in this
    test, namely the tests in TestWrongNonzeroOrdering.
    """
    @pytest.mark.serial
    @pytest.mark.fast
    def test_tril_behavior(self):
        mat = get_base_matrix(use_tril=True)
        mat2 = tril(mat)
        self.assertTrue(np.all(mat.row == mat2.row))
        self.assertTrue(np.all(mat.col == mat2.col))
        self.assertTrue(np.allclose(mat.data, mat2.data))

        mat = get_base_matrix_wrong_order(use_tril=True)
        self.assertFalse(np.all(mat.row == mat2.row))
        self.assertFalse(np.allclose(mat.data, mat2.data))
        mat2 = tril(mat)
        self.assertTrue(np.all(mat.row == mat2.row))
        self.assertTrue(np.all(mat.col == mat2.col))
        self.assertTrue(np.allclose(mat.data, mat2.data))


class TestLinearSolvers(unittest.TestCase):
    def _test_linear_solvers(self, solver):
        mat = get_base_matrix(use_tril=False)
        # Note: As of now, MKL Pardiso needs filled matrix even for symbolic factorization,
        # as zero diagonal elements are not allowed.
        # Also, MKL Pardiso need triu not tril - for now just disabling it
        #zero_mat = mat.copy()
        #zero_mat.data.fill(0)
        stat = solver.do_symbolic_factorization(mat)
        self.assertEqual(stat.status, parapint.linalg.LinearSolverStatus.successful)
        stat = solver.do_numeric_factorization(mat)
        self.assertEqual(stat.status, parapint.linalg.LinearSolverStatus.successful)
        x_true = np.array([1, 2, 3], dtype=np.double)
        rhs = mat * x_true
        x = solver.do_back_solve(rhs)
        self.assertTrue(np.allclose(x, x_true))
        x_true = np.array([4, 2, 3], dtype=np.double)
        rhs = mat * x_true
        x = solver.do_back_solve(rhs)
        self.assertTrue(np.allclose(x, x_true))

    def _test_inertia_computation(self, solver):
        mat = get_base_matrix(use_tril=False)
        #zero_mat = mat.copy()
        #zero_mat.data.fill(0)
        stat = solver.do_symbolic_factorization(mat)
        self.assertEqual(stat.status, parapint.linalg.LinearSolverStatus.successful)
        stat = solver.do_numeric_factorization(mat)
        self.assertEqual(stat.status, parapint.linalg.LinearSolverStatus.successful)
        inertia = solver.get_inertia()
        self.assertEqual(inertia[0], 2)
        self.assertEqual(inertia[1], 1)
        self.assertEqual(inertia[2], 0)

    @pytest.mark.serial
    @pytest.mark.fast
    def test_scipy(self):
        solver = parapint.linalg.ScipyInterface(compute_inertia=True)
        self._test_linear_solvers(solver)
        self._test_inertia_computation(solver)

    @pytest.mark.serial
    @pytest.mark.fast
    @unittest.skipIf(not mumps_available, 'mumps is needed for interior point mumps tests')
    def test_mumps(self):
        solver = parapint.linalg.MumpsInterface()
        self._test_linear_solvers(solver)
        self._test_inertia_computation(solver)

    @pytest.mark.serial
    @pytest.mark.fast
    @unittest.skipIf(not ma27_available, 'MA27 is needed for interior point MA27 tests')
    def test_ma27(self):
        solver = parapint.linalg.InteriorPointMA27Interface()
        self._test_linear_solvers(solver)
        self._test_inertia_computation(solver)

    @pytest.mark.serial
    @pytest.mark.fast
    @unittest.skipIf(not mkl_pardiso_available, 'MKL Pardiso is needed for interior point MKL Pardiso tests')
    def test_mkl_pardiso(self):
        os.environ['MKL_NUM_THREADS'] = '1'
        solver = parapint.linalg.InteriorPointMKLPardisoInterface()
        self._test_linear_solvers(solver)
        self._test_inertia_computation(solver)


@unittest.skip('This does not work yet')
class TestWrongNonzeroOrdering(unittest.TestCase):
    def _test_solvers(self, solver, use_tril):
        mat = get_base_matrix(use_tril=use_tril)
        wrong_order_mat = get_base_matrix_wrong_order(use_tril=use_tril)
        stat = solver.do_symbolic_factorization(mat)
        stat = solver.do_numeric_factorization(wrong_order_mat)
        x_true = np.array([1, 2, 3], dtype=np.double)
        rhs = mat * x_true
        x = solver.do_back_solve(rhs)
        self.assertTrue(np.allclose(x, x_true))

    @pytest.mark.serial
    @pytest.mark.fast
    def test_scipy(self):
        solver = parapint.linalg.ScipyInterface()
        self._test_solvers(solver, use_tril=True)

    @pytest.mark.serial
    @pytest.mark.fast
    @unittest.skipIf(not mumps_available, 'mumps is needed for interior point mumps tests')
    def test_mumps(self):
        solver = parapint.linalg.MumpsInterface()
        self._test_solvers(solver, use_tril=True)

    @pytest.mark.serial
    @pytest.mark.fast
    @unittest.skipIf(not ma27_available, 'MA27 is needed for interior point MA27 tests')
    def test_ma27(self):
        solver = parapint.linalg.InteriorPointMA27Interface()
        self._test_solvers(solver, use_tril=False)

    @pytest.mark.serial
    @pytest.mark.fast
    @unittest.skipIf(not mkl_pardiso_available, 'MKL Pardiso is needed for interior point MKL Pardiso tests')
    def test_mkl_pardiso(self):
        solver = parapint.linalg.InteriorPointMKLPardisoInterface()
        self._test_solvers(solver, use_tril=True)


class TestSchurComplementSolver(unittest.TestCase):

    def _test_schur_solvers(self, solver):
        A, H, S = get_schur_complement_matrices()
        #nnz_S = S.nnz
        stat = solver.do_symbolic_factorization(A, dim_schur=S.shape[0])
        self.assertEqual(stat.status, parapint.linalg.LinearSolverStatus.successful)
        #self.assertEqual(nnz_S, solver.get_schur_complement_nnz())
        stat = solver.do_numeric_factorization(A)
        self.assertEqual(stat.status, parapint.linalg.LinearSolverStatus.successful)
        self.assertTrue(np.allclose(S.toarray(), solver.get_schur_complement()))
        x_true = np.ones(H.shape[0])
        rhs = H * x_true
        x = solver.do_back_solve(rhs)
        self.assertTrue(np.allclose(x, x_true))

    @pytest.mark.serial
    @pytest.mark.fast
    @unittest.skipIf(not mkl_pardiso_available, 'MKL Pardiso is needed for interior point MKL Pardiso tests')
    def test_mkl_pardiso(self):
        solver = parapint.linalg.InteriorPointMKLPardisoSchurInterface()
        self._test_schur_solvers(solver)


if __name__ == '__main__':
    unittest.main()
