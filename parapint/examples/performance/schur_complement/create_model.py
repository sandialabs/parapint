import numpy as np
from scipy.sparse import coo_matrix, identity
from .utils import get_random_n_diagonal_matrix, get_ownership_map, distribute_blocks
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
import math
from mpi4py import MPI


class Block(object):
    def __init__(self, n_y, n_q, seed, theta, A, P):
        np.random.seed(seed)
        self.n_coupling = theta.size
        self.A = A
        self.q = np.random.normal(loc=5, scale=2, size=n_q)
        self.q[0:self.n_coupling] = theta
        self.y_hat = self.A * self.q
        noise = np.random.normal(loc=0, scale=0.01 * np.abs(self.y_hat).max(), size=n_y)
        self.y_hat += noise
        self.P = P

    def build_sub_kkt(self):
        n_y = self.y_hat.size
        n_q = self.q.size
        n_coupling = self.n_coupling

        kkt = BlockMatrix(nbrows=4, nbcols=4)

        kkt.set_row_size(0, n_y)
        kkt.set_row_size(1, n_q)
        kkt.set_row_size(2, n_y)
        kkt.set_row_size(3, n_coupling)
        kkt.set_col_size(0, n_y)
        kkt.set_col_size(1, n_q)
        kkt.set_col_size(2, n_y)
        kkt.set_col_size(3, n_coupling)

        kkt.set_block(0, 0, 2*identity(n_y, format='coo'))
        kkt.set_block(0, 2, identity(n_y, format='coo'))
        kkt.set_block(1, 2, -self.A.transpose())
        kkt.set_block(1, 3, self.P.transpose())
        kkt.set_block(2, 0, identity(n_y, format='coo'))
        kkt.set_block(2, 1, -self.A)
        kkt.set_block(3, 1, self.P)

        return kkt.tocoo()

    def build_sub_rhs(self):
        n_y = self.y_hat.size
        n_q = self.q.size
        n_coupling = self.n_coupling

        rhs = BlockVector(2)
        rhs.set_block(0, 2 * self.y_hat)
        rhs.set_block(1, np.zeros(n_q + n_y + n_coupling))

        return rhs.flatten()

    def check_result(self, sol):
        n_y = self.y_hat.size
        n_q = self.q.size
        q_estimate = sol[n_y:n_y+n_q]
        return abs(q_estimate - self.q).max()


class Model(object):
    def __init__(self, n_blocks, n_q_per_block, n_y_multiplier, n_theta, A_nnz_per_row):
        assert type(n_y_multiplier) is int
        assert n_y_multiplier > 1
        self.n_blocks = n_blocks
        self.n_y_per_block = n_y_per_block = n_q_per_block * n_y_multiplier
        self.n_q_per_block = n_q_per_block
        self.blocks = dict()
        np.random.seed(0)
        seed = np.random.randint(low=0, high=1000000)
        np.random.seed(seed)
        self.A = BlockMatrix(nbrows=n_y_multiplier, nbcols=1)
        for i in range(n_y_multiplier):
            self.A.set_block(i, 0, get_random_n_diagonal_matrix(n=n_q_per_block, nnz_per_row=A_nnz_per_row))
        self.A = self.A.tocoo()
        self.theta = np.random.normal(loc=5, scale=2, size=n_theta)
        self.P = BlockMatrix(nbrows=1, nbcols=2)
        self.P.set_block(0, 0, identity(n=n_theta, format='coo'))
        self.P.set_col_size(1, n_q_per_block - n_theta)
        self.P = self.P.tocoo()
        self.P_d = identity(n=n_theta, format='coo')
        for ndx in range(n_blocks):
            self.blocks[ndx] = Block(n_y=n_y_per_block, n_q=n_q_per_block, seed=ndx,
                                     theta=self.theta, A=self.A, P=self.P)

    def build_kkt(self):
        n_y = self.n_y_per_block
        n_q = self.n_q_per_block
        last_row_block = BlockMatrix(nbrows=1, nbcols=2)
        last_row_block.set_block(0, 1, -self.P_d.transpose())
        last_row_block.set_col_size(0, n_y + n_q + n_y)
        last_row_block = last_row_block.tocoo()

        kkt = BlockMatrix(nbrows=self.n_blocks + 1, nbcols=self.n_blocks + 1)
        for ndx in range(self.n_blocks):
            block = self.blocks[ndx].build_sub_kkt()
            kkt.set_block(ndx, ndx, block)
            kkt.set_block(self.n_blocks, ndx, last_row_block)
            kkt.set_block(ndx, self.n_blocks, last_row_block.transpose())

        kkt.set_block(self.n_blocks, self.n_blocks, coo_matrix((self.theta.size, self.theta.size)))

        return kkt

    def build_rhs(self):
        rhs = BlockVector(self.n_blocks + 1)
        for ndx in range(self.n_blocks):
            block = self.blocks[ndx].build_sub_rhs()
            rhs.set_block(ndx, block)
        rhs.set_block(self.n_blocks, np.zeros(self.theta.size))

        return rhs

    def check_result(self, sol):
        max_err = 0
        for ndx in range(self.n_blocks):
            tmp = self.blocks[ndx].check_result(sol.get_block(ndx))
            if tmp > max_err:
                max_err = tmp
        tmp = abs(sol.get_block(self.n_blocks) - self.theta).max()
        if tmp > max_err:
            max_err = tmp
        return max_err


class MPIModel(object):
    def __init__(self, n_blocks, n_q_per_block, n_y_multiplier, n_theta, A_nnz_per_row):
        assert type(n_y_multiplier) is int
        assert n_y_multiplier > 1
        self.n_blocks = n_blocks
        self.n_y_per_block = n_y_per_block = n_q_per_block * n_y_multiplier
        self.n_q_per_block = n_q_per_block
        self.blocks = dict()
        np.random.seed(0)
        seed = np.random.randint(low=0, high=1000000)
        np.random.seed(seed)
        self.A = BlockMatrix(nbrows=n_y_multiplier, nbcols=1)
        for i in range(n_y_multiplier):
            self.A.set_block(i, 0, get_random_n_diagonal_matrix(n=n_q_per_block, nnz_per_row=A_nnz_per_row))
        self.A = self.A.tocoo()
        self.theta = np.random.normal(loc=5, scale=2, size=n_theta)
        self.P = BlockMatrix(nbrows=1, nbcols=2)
        self.P.set_block(0, 0, identity(n=n_theta, format='coo'))
        self.P.set_col_size(1, n_q_per_block - n_theta)
        self.P = self.P.tocoo()
        self.P_d = identity(n=n_theta, format='coo')
        comm: MPI.Comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        self.local_blocks = distribute_blocks(num_blocks=n_blocks, rank=rank, size=size)
        for ndx in self.local_blocks:
            self.blocks[ndx] = Block(n_y=n_y_per_block, n_q=n_q_per_block, seed=ndx,
                                     theta=self.theta, A=self.A, P=self.P)

    def build_kkt(self):
        n_y = self.n_y_per_block
        n_q = self.n_q_per_block
        last_row_block = BlockMatrix(nbrows=1, nbcols=2)
        last_row_block.set_block(0, 1, -self.P_d.transpose())
        last_row_block.set_col_size(0, n_y + n_q + n_y)
        last_row_block = last_row_block.tocoo()

        comm: MPI.Comm = MPI.COMM_WORLD
        size = comm.Get_size()
        ownership_map = get_ownership_map(num_blocks=self.n_blocks, size=size)
        rank_owner = -np.ones((self.n_blocks+1, self.n_blocks+1), dtype=np.int64)
        for ndx in range(self.n_blocks):
            rank_owner[ndx, ndx] = ownership_map[ndx]
            rank_owner[self.n_blocks, ndx] = ownership_map[ndx]
            rank_owner[ndx, self.n_blocks] = ownership_map[ndx]
        kkt = MPIBlockMatrix(nbrows=self.n_blocks + 1,
                             nbcols=self.n_blocks + 1,
                             rank_ownership=rank_owner,
                             mpi_comm=comm,
                             assert_correct_owners=False)
        for ndx in self.local_blocks:
            block = self.blocks[ndx].build_sub_kkt()
            kkt.set_block(ndx, ndx, block)
            kkt.set_block(self.n_blocks, ndx, last_row_block)
            kkt.set_block(ndx, self.n_blocks, last_row_block.transpose())

        kkt.set_block(self.n_blocks, self.n_blocks, coo_matrix((self.theta.size, self.theta.size)))

        return kkt

    def build_rhs(self):
        comm: MPI.Comm = MPI.COMM_WORLD
        size = comm.Get_size()
        ownership_map = get_ownership_map(num_blocks=self.n_blocks, size=size)
        rank_owner = -np.ones(self.n_blocks+1, dtype=np.int64)
        for ndx in range(self.n_blocks):
            rank_owner[ndx] = ownership_map[ndx]
        rhs = MPIBlockVector(nblocks=self.n_blocks + 1, rank_owner=rank_owner, mpi_comm=comm, assert_correct_owners=False)
        for ndx in self.local_blocks:
            block = self.blocks[ndx].build_sub_rhs()
            rhs.set_block(ndx, block)
        rhs.set_block(self.n_blocks, np.zeros(self.theta.size))

        return rhs

    def check_result(self, sol):
        max_err = 0
        for ndx in self.local_blocks:
            tmp = self.blocks[ndx].check_result(sol.get_block(ndx))
            if tmp > max_err:
                max_err = tmp
        tmp = abs(sol.get_block(self.n_blocks) - self.theta).max()
        if tmp > max_err:
            max_err = tmp
        comm: MPI.Comm = MPI.COMM_WORLD
        max_err = comm.allreduce(max_err, MPI.MAX)
        # print(self.theta)
        # print(sol.get_block(self.n_blocks))
        return max_err
