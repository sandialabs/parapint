import unittest
from pyomo.common.dependencies import attempt_import
import numpy as np
from scipy.sparse import coo_matrix
import parapint
mumps, mumps_available = attempt_import('mumps')


class TestReallocation(unittest.TestCase):
    @unittest.skipIf(not mumps_available, 'mumps is not available')
    def test_reallocate_memory_mumps(self):

        # Create a tri-diagonal matrix with small entries on the diagonal
        n = 10000
        small_val = 1e-7
        big_val = 1e2
        irn = []
        jcn = []
        ent = []
        for i in range(n-1):
            irn.extend([i+1, i, i])
            jcn.extend([i, i, i+1])
            ent.extend([big_val,small_val,big_val])
        irn.append(n-1)
        jcn.append(n-1)
        ent.append(small_val)
        irn = np.array(irn)
        jcn = np.array(jcn)
        ent = np.array(ent)

        matrix = coo_matrix((ent, (irn, jcn)), shape=(n,n))

        linear_solver = parapint.linalg.MumpsInterface()
        linear_solver.do_symbolic_factorization(matrix)

        predicted = linear_solver.get_infog(16)

        res = linear_solver.do_numeric_factorization(matrix, raise_on_error=False)
        self.assertEqual(res.status, parapint.linalg.LinearSolverStatus.not_enough_memory)

        linear_solver.do_symbolic_factorization(matrix)

        factor = 2
        linear_solver.increase_memory_allocation(factor)

        res = linear_solver.do_numeric_factorization(matrix)
        self.assertEqual(res.status, parapint.linalg.LinearSolverStatus.successful)

        # Expected memory allocation (MB)
        self.assertEqual(linear_solver._prev_allocation, 6)

        actual = linear_solver.get_infog(18)

        # Sanity checks:
        # Make sure actual memory usage is greater than initial guess
        self.assertTrue(predicted < actual)
        # Make sure memory allocation is at least as much as was used
        self.assertTrue(actual <= linear_solver._prev_allocation)


if __name__ == '__main__':
    test_realloc = TestReallocation()
    test_realloc.test_reallocate_memory_mumps()
