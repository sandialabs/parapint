import unittest
import parapint
from nose.plugins.attrib import attr
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
from parapint.linalg import ScipyInterface
from scipy.sparse import coo_matrix
import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class TestSchurComplement(unittest.TestCase):
    @attr(parallel=True, speed='fast', n_procs='all')
    def test_mpi_schur_complement(self):
        rank_by_index = list()
        for ndx in range(3):
            for _rank in range(size):
                if (ndx - _rank) % size == 0:
                    rank_by_index.append(_rank)
        rank_by_index.append(-1)

        A = MPIBlockMatrix(nbrows=4,
                           nbcols=4,
                           rank_ownership=[rank_by_index,
                                           rank_by_index,
                                           rank_by_index,
                                           rank_by_index],
                           mpi_comm=comm)
        if rank_by_index[0] == rank:
            A.set_block(0, 0, coo_matrix(np.array([[1, 1],
                                                   [0, 1]], dtype=np.double)))
        if rank_by_index[1] == rank:
            A.set_block(1, 1, coo_matrix(np.array([[1, 0],
                                                   [0, 1]], dtype=np.double)))
        if rank_by_index[2] == rank:
            A.set_block(2, 2, coo_matrix(np.array([[1, 0],
                                                   [1, 1]], dtype=np.double)))
        A.set_block(3, 3, coo_matrix(np.array([[0, 0],
                                               [0, 1]], dtype=np.double)))
        if rank_by_index[0] == rank:
            A.set_block(3, 0, coo_matrix(np.array([[0, -1],
                                                   [0, 0]], dtype=np.double)))
        if rank_by_index[1] == rank:
            A.set_block(3, 1, coo_matrix(np.array([[-1, 0],
                                                   [0, -1]], dtype=np.double)))
        if rank_by_index[2] == rank:
            A.set_block(3, 2, coo_matrix(np.array([[0, 0],
                                                   [-1, 0]], dtype=np.double)))
        A.broadcast_block_sizes()

        local_A = BlockMatrix(4, 4)
        local_A.set_block(0, 0, coo_matrix(np.array([[1, 1],
                                                     [0, 1]], dtype=np.double)))
        local_A.set_block(1, 1, coo_matrix(np.array([[1, 0],
                                                     [0, 1]], dtype=np.double)))
        local_A.set_block(2, 2, coo_matrix(np.array([[1, 0],
                                                     [1, 1]], dtype=np.double)))
        local_A.set_block(3, 3, coo_matrix(np.array([[0, 0],
                                                     [0, 1]], dtype=np.double)))
        local_A.set_block(3, 0, coo_matrix(np.array([[0, -1],
                                                     [0, 0]], dtype=np.double)))
        local_A.set_block(3, 1, coo_matrix(np.array([[-1, 0],
                                                     [0, -1]], dtype=np.double)))
        local_A.set_block(3, 2, coo_matrix(np.array([[0, 0],
                                                     [-1, 0]], dtype=np.double)))
        local_A.set_block(0, 3, local_A.get_block(3, 0).transpose(copy=True))
        local_A.set_block(1, 3, local_A.get_block(3, 1).transpose(copy=True))
        local_A.set_block(2, 3, local_A.get_block(3, 2).transpose(copy=True))

        rhs = MPIBlockVector(nblocks=4, rank_owner=rank_by_index, mpi_comm=comm)
        if rank_by_index[0] == rank:
            rhs.set_block(0, np.array([1, 0], dtype=np.double))
        if rank_by_index[1] == rank:
            rhs.set_block(1, np.array([0, 0], dtype=np.double))
        if rank_by_index[2] == rank:
            rhs.set_block(2, np.array([0, 1], dtype=np.double))
        rhs.set_block(3, np.array([1, 1], dtype=np.double))
        rhs.broadcast_block_sizes()

        local_rhs = BlockVector(4)
        local_rhs.set_block(0, np.array([1, 0], dtype=np.double))
        local_rhs.set_block(1, np.array([0, 0], dtype=np.double))
        local_rhs.set_block(2, np.array([0, 1], dtype=np.double))
        local_rhs.set_block(3, np.array([1, 1], dtype=np.double))

        x1 = np.linalg.solve(local_A.toarray(), local_rhs.flatten())

        solver_class = parapint.linalg.MPISchurComplementLinearSolver
        sc_solver = solver_class(subproblem_solvers={ndx: ScipyInterface(compute_inertia=True) for ndx in range(3)},
                                 schur_complement_solver=ScipyInterface(compute_inertia=True))
        sc_solver.do_symbolic_factorization(A)
        sc_solver.do_numeric_factorization(A)
        x2 = sc_solver.do_back_solve(rhs)

        self.assertTrue(np.allclose(x1, x2.make_local_copy().flatten()))

        inertia1 = sc_solver.get_inertia()
        eig = np.linalg.eigvals(local_A.toarray())
        pos = np.count_nonzero(eig > 0)
        neg = np.count_nonzero(eig < 0)
        zero = np.count_nonzero(eig == 0)
        inertia2 = (pos, neg, zero)
        self.assertEqual(inertia1, inertia2)

        sc_solver.do_numeric_factorization(A)
        x2 = sc_solver.do_back_solve(rhs)
        self.assertTrue(np.allclose(x1, x2.make_local_copy().flatten()))
