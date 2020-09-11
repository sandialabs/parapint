import unittest
import parapint
from nose.plugins.attrib import attr
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
from parapint.linalg import ScipyInterface
from scipy.sparse import coo_matrix
import numpy as np


@attr(parallel=False, speed='fast')
class TestSchurComplement(unittest.TestCase):
    def test_schur_complement(self):
        A = BlockMatrix(4, 4)
        A.set_block(0, 0, coo_matrix(np.array([[1, 1],
                                               [0, 1]], dtype=np.double)))
        A.set_block(1, 1, coo_matrix(np.array([[1, 0],
                                               [0, 1]], dtype=np.double)))
        A.set_block(2, 2, coo_matrix(np.array([[1, 0],
                                               [1, 1]], dtype=np.double)))
        A.set_block(3, 3, coo_matrix(np.array([[0, 0],
                                               [0, 0]], dtype=np.double)))
        A.set_block(3, 0, coo_matrix(np.array([[0, -1],
                                               [0, 0]], dtype=np.double)))
        A.set_block(3, 1, coo_matrix(np.array([[-1, 0],
                                               [0, -1]], dtype=np.double)))
        A.set_block(3, 2, coo_matrix(np.array([[0, 0],
                                               [-1, 0]], dtype=np.double)))
        A_upper = A.copy_structure()
        A_upper.set_block(0, 3, A.get_block(3, 0).transpose(copy=True))
        A_upper.set_block(1, 3, A.get_block(3, 1).transpose(copy=True))
        A_upper.set_block(2, 3, A.get_block(3, 2).transpose(copy=True))

        rhs = BlockVector(4)
        rhs.set_block(0, np.array([1, 0], dtype=np.double))
        rhs.set_block(1, np.array([0, 0], dtype=np.double))
        rhs.set_block(2, np.array([0, 1], dtype=np.double))
        rhs.set_block(3, np.array([1, 1], dtype=np.double))

        x1 = np.linalg.solve((A + A_upper).toarray(), rhs.flatten())

        sc_solver = parapint.linalg.SchurComplementLinearSolver(subproblem_solvers={ndx: ScipyInterface(compute_inertia=True) for ndx in range(3)},
                                                               schur_complement_solver=ScipyInterface(compute_inertia=True))
        sc_solver.do_symbolic_factorization(A)
        sc_solver.do_numeric_factorization(A)
        x2 = sc_solver.do_back_solve(rhs)

        inertia1 = sc_solver.get_inertia()
        eig = np.linalg.eigvals((A + A_upper).toarray())
        pos = np.count_nonzero(eig > 0)
        neg = np.count_nonzero(eig < 0)
        zero = np.count_nonzero(eig == 0)
        inertia2 = (pos, neg, zero)
        self.assertTrue(np.allclose(x1, x2.flatten()))
        self.assertEqual(inertia1, inertia2)
