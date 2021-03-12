import pyomo.environ as pe
from pyomo.core.base.block import _BlockData
import math
from pyomo.opt.results import assert_optimal_termination
import unittest
from nose.plugins.attrib import attr
from typing import Tuple, Sequence
from pyomo.core.base.var import _GeneralVarData
import parapint
from pyomo.contrib.pynumero.sparse import BlockVector
import numpy as np
from mpi4py import MPI
from parapint.interfaces.schur_complement.mpi_sc_ip_interface import _get_ownership_map, _distribute_blocks
from scipy.sparse import coo_matrix


comm: MPI.Comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


"""
Test problem is

min integral from t0 to tf of [x(t) - sin(t) - 1]**2
s.t.
dx/dt = p(t) - x(t)

Using trapezoid rule for integral and implicit euler for differential equation
"""


def build_time_block(t0: int,
                     delta_t: int,
                     num_finite_elements: int,
                     constant_control_duration: int,
                     time_scale: float,
                     with_bounds: bool = False,
                     with_ineq: bool = False) -> _BlockData:
    """
    Parameters
    ----------
    t0: int
        start time
    delta_t: float
        end time
    num_finite_elements:
        number of finite elements
    constant_control_duration:
        number of finite elements over which the
    time_scale: float
        coefficient of t within the sin function
    with_bounds: bool
    with_ineq: bool

    Returns
    -------
    m: _BlockData
        The Pyomo model
    """
    assert constant_control_duration >= delta_t
    assert constant_control_duration % delta_t == 0
    assert (num_finite_elements * delta_t) % constant_control_duration == 0

    def finite_element_ndx_to_start_t_x(ndx):
        return t0 + ndx * delta_t

    def finite_element_ndx_to_end_t_x(ndx):
        return t0 + (ndx + 1) * delta_t

    def finite_element_ndx_to_start_t_p(ndx):
        return t0 + (math.floor(ndx / (constant_control_duration / delta_t))) * constant_control_duration

    m = pe.Block(concrete=True)
    m.x_time_points = pe.Set(initialize=[t for t in range(t0, t0+delta_t*(num_finite_elements + 1), delta_t)])
    m.x = pe.Var(m.x_time_points)
    num_p_elements = int((num_finite_elements * delta_t) / constant_control_duration)
    m.p_time_points = pe.Set(initialize=[t for t in range(t0,
                                                          t0+constant_control_duration*num_p_elements,
                                                          constant_control_duration)])
    if with_bounds:
        bnds = (None, 1.75)
    else:
        bnds = (None, None)
    m.p = pe.Var(m.p_time_points, bounds=bnds)

    obj_expr = 0
    for fe_ndx in range(num_finite_elements):
        start_t_x = finite_element_ndx_to_start_t_x(fe_ndx)
        end_t_x = finite_element_ndx_to_end_t_x(fe_ndx)
        obj_expr += 0.5 * delta_t * ((m.x[start_t_x] - (math.sin(time_scale*start_t_x) + 1))**2 +
                                     (m.x[end_t_x] - (math.sin(time_scale*end_t_x) + 1))**2)
    m.obj = pe.Objective(expr=obj_expr)

    m.con_indices = pe.Set(initialize=[t for t in m.x_time_points if t > t0])
    m.cons = pe.Constraint(m.con_indices)
    for fe_ndx in range(num_finite_elements):
        start_t_x = finite_element_ndx_to_start_t_x(fe_ndx)
        end_t_x = finite_element_ndx_to_end_t_x(fe_ndx)
        start_t_p = finite_element_ndx_to_start_t_p(fe_ndx)
        m.cons[end_t_x] = m.x[end_t_x] - (m.x[start_t_x] + delta_t * (m.p[start_t_p] - m.x[end_t_x])) == 0

    if with_ineq:
        m.p_ub = pe.Constraint(m.p_time_points)
        for t in m.p_time_points:
            m.p_ub[t] = m.p[t] <= 1.75

    return m


class Problem(parapint.interfaces.MPIDynamicSchurComplementInteriorPointInterface):
    def __init__(self,
                 t0=0,
                 delta_t=1,
                 num_finite_elements=90,
                 constant_control_duration=10,
                 time_scale=0.1,
                 num_time_blocks=3,
                 with_bounds=False,
                 with_ineq=False):
        assert num_finite_elements % num_time_blocks == 0
        self.t0: int = t0
        self.delta_t: int = delta_t
        self.num_finite_elements: int = num_finite_elements
        self.constant_control_duration: int = constant_control_duration
        self.time_scale: float = time_scale
        self.num_time_blocks: int = num_time_blocks
        self.tf = self.t0 + self.delta_t*self.num_finite_elements
        self.with_bounds = with_bounds
        self.with_ineq = with_ineq
        super(Problem, self).__init__(start_t=self.t0,
                                      end_t=self.tf,
                                      num_time_blocks=self.num_time_blocks,
                                      comm=MPI.COMM_WORLD)

    def build_model_for_time_block(self, ndx: int,
                                   start_t: float,
                                   end_t: float,
                                   add_init_conditions: bool) -> Tuple[_BlockData,
                                                                       Sequence[_GeneralVarData],
                                                                       Sequence[_GeneralVarData]]:
        assert int(start_t) == start_t
        assert int(end_t) == end_t
        assert end_t == start_t + self.delta_t*(self.num_finite_elements/self.num_time_blocks)
        start_t = int(start_t)
        end_t = int(end_t)
        m = build_time_block(t0=start_t,
                             delta_t=self.delta_t,
                             num_finite_elements=int(self.num_finite_elements/self.num_time_blocks),
                             constant_control_duration=self.constant_control_duration,
                             time_scale=self.time_scale,
                             with_bounds=self.with_bounds,
                             with_ineq=self.with_ineq)
        start_states = [m.x[start_t]]
        end_states = [m.x[end_t]]
        return m, start_states, end_states


@attr(parallel=True, speed='fast', n_procs=[1, 2, 3])
class TestSCIPInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.t0 = 0
        cls.delta_t = 1
        cls.num_finite_elements = 6
        cls.constant_control_duration = 2
        cls.time_scale = 1
        cls.num_time_blocks = 3
        cls.with_bounds = True
        cls.with_ineq = False
        cls.barrier_parameter = 0.1
        cls.interface = Problem(t0=cls.t0,
                                delta_t=cls.delta_t,
                                num_finite_elements=cls.num_finite_elements,
                                constant_control_duration=cls.constant_control_duration,
                                time_scale=cls.time_scale,
                                num_time_blocks=cls.num_time_blocks,
                                with_bounds=cls.with_bounds,
                                with_ineq=cls.with_ineq)
        interface = cls.interface
        num_time_blocks = cls.num_time_blocks

        primals = BlockVector(num_time_blocks + 1)
        duals_eq = BlockVector(num_time_blocks)
        duals_ineq = BlockVector(num_time_blocks)
        duals_primals_lb = BlockVector(num_time_blocks + 1)
        duals_primals_ub = BlockVector(num_time_blocks + 1)
        duals_slacks_lb = BlockVector(num_time_blocks)
        duals_slacks_ub = BlockVector(num_time_blocks)

        ownership_map = _get_ownership_map(num_time_blocks, size)

        val_map = pe.ComponentMap()

        if ownership_map[0] == rank:
            m = interface.pyomo_model(0)
            val_map[m.x[0]] = 0
            val_map[m.x[1]] = 1
            val_map[m.x[2]] = 2
            val_map[m.p[0]] = 0.5
            val_map[m.cons[1]] = 1
            val_map[m.cons[2]] = 2

        if ownership_map[1] == rank:
            m = interface.pyomo_model(1)
            val_map[m.x[2]] = 2
            val_map[m.x[3]] = 3
            val_map[m.x[4]] = 4
            val_map[m.p[2]] = 1
            val_map[m.cons[3]] = 3
            val_map[m.cons[4]] = 4

        if ownership_map[2] == rank:
            m = interface.pyomo_model(2)
            val_map[m.x[4]] = 4
            val_map[m.x[5]] = 5
            val_map[m.x[6]] = 6
            val_map[m.p[4]] = 1.5
            val_map[m.cons[5]] = 5
            val_map[m.cons[6]] = 6

        for ndx in range(num_time_blocks):
            primals.set_block(ndx, np.zeros(4, dtype=np.double))
            duals_primals_lb.set_block(ndx, np.zeros(4, dtype=np.double))
            duals_primals_ub.set_block(ndx, np.zeros(4, dtype=np.double))
            duals_ineq.set_block(ndx, np.zeros(0, dtype=np.double))
            duals_slacks_lb.set_block(ndx, np.zeros(0, dtype=np.double))
            duals_slacks_ub.set_block(ndx, np.zeros(0, dtype=np.double))
            sub_duals_eq = BlockVector(3)
            sub_duals_eq.set_block(0, np.zeros(2, dtype=np.double))
            if ndx == 0:
                sub_duals_eq.set_block(1, np.zeros(0, dtype=np.double))
            else:
                sub_duals_eq.set_block(1, np.zeros(1, dtype=np.double))
            if ndx == num_time_blocks - 1:
                sub_duals_eq.set_block(2, np.zeros(0, dtype=np.double))
            else:
                sub_duals_eq.set_block(2, np.zeros(1, dtype=np.double))
            duals_eq.set_block(ndx, sub_duals_eq)
        primals.set_block(num_time_blocks, np.zeros(2, dtype=np.double))
        duals_primals_lb.set_block(num_time_blocks, np.zeros(2, dtype=np.double))
        duals_primals_ub.set_block(num_time_blocks, np.zeros(2, dtype=np.double))

        local_block_indices = _distribute_blocks(num_time_blocks, rank, size)
        for ndx in local_block_indices:
            primals.set_block(ndx, np.array([val_map[i] for i in interface.get_pyomo_variables(ndx)], dtype=np.double))
            duals_primals_ub.set_block(ndx, np.array([0, 0, 0, ndx], dtype=np.double))
            sub_duals_eq = duals_eq.get_block(ndx)
            sub_duals_eq.set_block(0, np.array([val_map[i] for i in interface.get_pyomo_constraints(ndx)], dtype=np.double))
            if ndx == 0:
                sub_duals_eq.set_block(1, np.zeros(0, dtype=np.double))
            else:
                sub_duals_eq.set_block(1, np.ones(1, dtype=np.double) * ndx)
            if ndx == num_time_blocks - 1:
                sub_duals_eq.set_block(2, np.zeros(0, dtype=np.double))
            else:
                sub_duals_eq.set_block(2, np.ones(1, dtype=np.double) * ndx)

        primals_flat = primals.flatten()
        res = np.zeros(primals_flat.size, dtype=np.double)
        comm.Allreduce(primals_flat, res)
        primals.copyfrom(res)

        duals_primals_lb_flat = duals_primals_lb.flatten()
        res = np.zeros(duals_primals_lb_flat.size, dtype=np.double)
        comm.Allreduce(duals_primals_lb_flat, res)
        duals_primals_lb.copyfrom(res)

        duals_primals_ub_flat = duals_primals_ub.flatten()
        res = np.zeros(duals_primals_ub_flat.size, dtype=np.double)
        comm.Allreduce(duals_primals_ub_flat, res)
        duals_primals_ub.copyfrom(res)

        duals_eq_flat = duals_eq.flatten()
        res = np.zeros(duals_eq_flat.size, dtype=np.double)
        comm.Allreduce(duals_eq_flat, res)
        duals_eq.copyfrom(res)

        primals.set_block(num_time_blocks, np.array([3, 6], dtype=np.double))
        duals_primals_lb.set_block(num_time_blocks, np.zeros(2, dtype=np.double))
        duals_primals_ub.set_block(num_time_blocks, np.zeros(2, dtype=np.double))
        interface.set_primals(primals)
        interface.set_duals_eq(duals_eq)
        interface.set_duals_ineq(duals_ineq)
        interface.set_duals_slacks_lb(duals_slacks_lb)
        interface.set_duals_slacks_ub(duals_slacks_ub)
        interface.set_duals_primals_lb(duals_primals_lb)
        interface.set_duals_primals_ub(duals_primals_ub)
        interface.set_barrier_parameter(cls.barrier_parameter)

    def test_n_primals(self):
        self.assertEqual(self.interface.n_primals(), 14)

    def test_primals_lb(self):
        expected = np.zeros(14, dtype=np.double)
        expected.fill(-np.inf)
        got = self.interface.primals_lb().make_local_copy().flatten()
        self.assertTrue(np.allclose(expected, got))

    def test_primals_ub(self):
        expected = np.array([np.inf, np.inf, np.inf, 1.75, np.inf, np.inf, np.inf, 1.75, np.inf, np.inf, np.inf, 1.75, np.inf, np.inf], dtype=np.double)
        got = self.interface.primals_ub().make_local_copy().flatten()
        self.assertTrue(np.allclose(expected, got))

    def test_get_primals(self):
        expected = np.array([0, 1, 2, 0.5, 2, 3, 4, 1, 4, 5, 6, 1.5, 3, 6], dtype=np.double)
        got = self.interface.get_primals().make_local_copy().flatten()
        self.assertTrue(np.allclose(expected, got))

    def test_evaluate_objective(self):
        ts = self.time_scale
        expected = (0.5 * (0 - math.sin(ts*0) - 1)**2 +
                    0.5 * (1 - math.sin(ts*1) - 1)**2 +
                    0.5 * (1 - math.sin(ts*1) - 1)**2 +
                    0.5 * (2 - math.sin(ts*2) - 1)**2 +
                    0.5 * (2 - math.sin(ts*2) - 1)**2 +
                    0.5 * (3 - math.sin(ts*3) - 1)**2 +
                    0.5 * (3 - math.sin(ts*3) - 1)**2 +
                    0.5 * (4 - math.sin(ts*4) - 1)**2 +
                    0.5 * (4 - math.sin(ts*4) - 1)**2 +
                    0.5 * (5 - math.sin(ts*5) - 1)**2 +
                    0.5 * (5 - math.sin(ts*5) - 1)**2 +
                    0.5 * (6 - math.sin(ts*6) - 1)**2)
        got = self.interface.evaluate_objective()
        self.assertAlmostEqual(expected, got)

    def test_evaluate_grad_objective(self):
        # -----block 0----------  -------block 1--------  ------block 2---------  -coupling-
        # x[0]  x[1]  x[2]  p[0]  x[2]  x[3]  x[4]  p[2]  x[4]  x[5]  x[6]  p[4]  x[2]  x[4]
        ts = self.time_scale
        expected = np.array([1 * (0 - math.sin(ts*0) - 1),
                             2 * (1 - math.sin(ts*1) - 1),
                             1 * (2 - math.sin(ts*2) - 1),
                             0,
                             1 * (2 - math.sin(ts*2) - 1),
                             2 * (3 - math.sin(ts*3) - 1),
                             1 * (4 - math.sin(ts*4) - 1),
                             0,
                             1 * (4 - math.sin(ts*4) - 1),
                             2 * (5 - math.sin(ts*5) - 1),
                             1 * (6 - math.sin(ts*6) - 1),
                             0,
                             0,
                             0],
                            dtype=np.double)
        got = self.interface.evaluate_grad_objective().make_local_copy().flatten()
        self.assertTrue(np.allclose(expected, got))

    def test_n_eq_constraints(self):
        self.assertEqual(self.interface.n_eq_constraints(), 10)

    def test_n_ineq_constraints(self):
        self.assertEqual(self.interface.n_ineq_constraints(), 0)

    def test_ineq_lb(self):
        res = self.interface.ineq_lb()
        self.assertEqual(res.nblocks, 3)
        self.assertEqual(res.size, 0)

    def test_ineq_ub(self):
        res = self.interface.ineq_ub()
        self.assertEqual(res.nblocks, 3)
        self.assertEqual(res.size, 0)

    def test_get_duals_eq(self):
        expected = np.array([1, 2, 0, 3, 4, 1, 1, 5, 6, 2], dtype=np.double)
        got = self.interface.get_duals_eq().make_local_copy().flatten()
        self.assertTrue(np.allclose(expected, got))

    def test_get_duals_ineq(self):
        res = self.interface.get_duals_ineq()
        self.assertEqual(res.nblocks, 3)
        self.assertEqual(res.size, 0)

    def test_evaluate_eq_constraints(self):
        expected = np.array([1 - (0 + (0.5 - 1)),
                             2 - (1 + (0.5 - 2)),
                             2 - 3,
                             3 - (2 + (1 - 3)),
                             4 - (3 + (1 - 4)),
                             2 - 3,
                             4 - 6,
                             5 - (4 + (1.5 - 5)),
                             6 - (5 + (1.5 - 6)),
                             4 - 6],
                            dtype=np.double)
        got = self.interface.evaluate_eq_constraints()
        got = got.make_local_copy().flatten()
        self.assertTrue(np.allclose(expected, got))

    def test_evaluate_ineq_constraints(self):
        res = self.interface.evaluate_ineq_constraints()
        self.assertEqual(res.nblocks, 3)
        self.assertEqual(res.size, 0)

    def test_evaluate_jacobian_eq(self):
        #            -----block 0----------  -------block 1--------  ------block 2---------  -coupling-
        #            x[0]  x[1]  x[2]  p[0]  x[2]  x[3]  x[4]  p[2]  x[4]  x[5]  x[6]  p[4]  x[2]  x[4]
        expected = [[-1,   2,    0,    -1,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],  # x[1] - (x[0] + (p[0] - x[1]))  block 0
                    [0,    -1,   2,    -1,   0,    0,    0,    0,    0,    0,    0,    0,    0,    0 ],  # x[2] - (x[1] + (p[0] - x[2]))  block 0
                    [0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    -1,   0 ],  # x[2] - coupling[0]             block 0
                    [0,    0,    0,    0,    -1,   2,    0,    -1,   0,    0,    0,    0,    0,    0 ],  # x[3] - (x[2] + (p[2] - x[3]))  block 1
                    [0,    0,    0,    0,    0,    -1,   2,    -1,   0,    0,    0,    0,    0,    0 ],  # x[4] - (x[3] + (p[2] - x[4]))  block 1
                    [0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    -1,   0 ],  # x[2] - coupling[0]             block 1
                    [0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    -1],  # x[4] - coupling[1]             block 1
                    [0,    0,    0,    0,    0,    0,    0,    0,    -1,   2,    0,    -1,   0,    0 ],  # x[5] - (x[4] + (p[4] - x[5]))  block 2
                    [0,    0,    0,    0,    0,    0,    0,    0,    0,    -1,   2,    -1,   0,    0 ],  # x[6] - (x[5] + (p[4] - x[6]))  block 2
                    [0,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    -1],  # x[4] - coupling[1]             block 2
                    ]
        expected = np.array(expected, dtype=np.double)
        got = self.interface.evaluate_jacobian_eq().to_local_array()
        self.assertTrue(np.allclose(expected, got))

    def test_evaluate_jacobian_ineq(self):
        res = self.interface.evaluate_jacobian_ineq()
        nrows, ncols = res.shape
        self.assertEqual(nrows, 0)
        self.assertEqual(ncols, 14)

    def test_get_slacks(self):
        res = self.interface.get_slacks()
        self.assertEqual(res.nblocks, 3)
        self.assertEqual(res.size, 0)

    def test_get_duals_primals_lb(self):
        expected = np.zeros(14, dtype=np.double)
        got = self.interface.get_duals_primals_lb().make_local_copy().flatten()
        self.assertTrue(np.allclose(expected, got))

    def test_get_duals_primals_ub(self):
        expected = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0], dtype=np.double)
        got = self.interface.get_duals_primals_ub().make_local_copy().flatten()
        self.assertTrue(np.allclose(expected, got))

    def test_get_duals_slacks_lb(self):
        res = self.interface.get_duals_slacks_lb()
        self.assertEqual(res.nblocks, 3)
        self.assertEqual(res.size, 0)

    def test_get_duals_slacks_ub(self):
        res = self.interface.get_duals_slacks_ub()
        self.assertEqual(res.nblocks, 3)
        self.assertEqual(res.size, 0)

    def test_evaluate_primal_dual_kkt_rhs(self):
        got = self.interface.evaluate_primal_dual_kkt_rhs().make_local_copy().flatten()
        self.assertEqual(got.size, 24)
        ts = self.time_scale
        barrier = self.barrier_parameter
        expected = np.array([1 * (0 - math.sin(ts*0) - 1) + (-1)*1 - 0 + 0,        # grad lag wrt x[0]              block 0
                             2 * (1 - math.sin(ts*1) - 1) + 2*1 + (-1)*2 - 0 + 0,  # grad lag wrt x[1]              block 0
                             1 * (2 - math.sin(ts*2) - 1) + 2*2 + 1*0 - 0 + 0,     # grad lag wrt x[2]              block 0
                             0 + (-1)*1 + (-1)*2 - 0 + barrier/(1.75 - 0.5),       # grad lag wrt p[0]              block 0
                             1 - (0 + (0.5 - 1)),                                  # x[1] - (x[0] + (p[0] - x[1]))  block 0
                             2 - (1 + (0.5 - 2)),                                  # x[2] - (x[1] + (p[0] - x[2]))  block 0
                             1 * (2 - math.sin(ts*2) - 1) + (-1)*3 + 1*1 - 0 + 0,  # grad lag wrt x[2]              block 1
                             2 * (3 - math.sin(ts*3) - 1) + 2*3 + (-1)*4,          # grad lag wrt x[3]              block 1
                             1 * (4 - math.sin(ts*4) - 1) + 2*4 + 1*1,             # grad lag wrt x[4]              block 1
                             0 + (-1)*3 + (-1)*4 - 0 + barrier/(1.75 - 1.0),       # grad lag wrt p[2]              block 1
                             3 - (2 + (1 - 3)),                                    # x[3] - (x[2] + (p[2] - x[3]))  block 1
                             4 - (3 + (1 - 4)),                                    # x[4] - (x[3] + (p[2] - x[4]))  block 1
                             2 - 3,                                                # x[2] - coupling[0]             block 1
                             1 * (4 - math.sin(ts*4) - 1) + (-1)*5 + 1*2 - 0 + 0,  # grad lag wrt x[4]              block 2
                             2 * (5 - math.sin(ts*5) - 1) + 2*5 + (-1)*6 - 0 + 0,  # grad lag wrt x[5]              block 2
                             1 * (6 - math.sin(ts*6) - 1) + 2*6 - 0 + 0,           # grad lag wrt x[6]              block 2
                             0 + (-1)*5 + (-1)*6 - 0 + barrier/(1.75 - 1.5),       # grad lag wrt p[4]              block 2
                             5 - (4 + (1.5 - 5)),                                  # x[5] - (x[4] + (p[4] - x[5]))  block 2
                             6 - (5 + (1.5 - 6)),                                  # x[6] - (x[5] + (p[4] - x[6]))  block 2
                             4 - 6,                                                # x[4] - coupling[1]             block 2
                             2 - 3,                                                # x[2] - coupling[0]             block 0
                             4 - 6,                                                # x[4] - coupling[1]             block 1
                             0 + (-1)*1 + (-1)*0,                                  # grad lag wrt coupling[0]
                             0 + (-1)*2 + (-1)*1],                                 # grad lag wrt coupling[1]
                            dtype=np.double)
        expected *= -1
        self.assertTrue(np.allclose(expected, got))


@attr(parallel=True, speed='medium', n_procs=[1, 2, 3])
class TestSCIPInterfaceWithSolve(unittest.TestCase):
    def test_kkt_system(self):
        t0 = 0
        delta_t = 1
        num_finite_elements = 90
        constant_control_duration = 10
        time_scale = 0.1
        num_time_blocks = 3
        interface = Problem(t0=t0,
                            delta_t=delta_t,
                            num_finite_elements=num_finite_elements,
                            constant_control_duration=constant_control_duration,
                            time_scale=time_scale,
                            num_time_blocks=num_time_blocks)
        interface.set_primals(interface.init_primals())
        interface.set_slacks(interface.init_slacks())
        interface.set_duals_eq(interface.init_duals_eq())
        interface.set_duals_ineq(interface.init_duals_ineq())
        interface.set_duals_primals_lb(interface.init_duals_primals_lb())
        interface.set_duals_primals_ub(interface.init_duals_primals_ub())
        interface.set_duals_slacks_lb(interface.init_duals_slacks_lb())
        interface.set_duals_slacks_ub(interface.init_duals_slacks_ub())
        interface.set_barrier_parameter(0)
        kkt = interface.evaluate_primal_dual_kkt_matrix()
        rhs = interface.evaluate_primal_dual_kkt_rhs()
        linear_solver = parapint.linalg.ScipyInterface()
        local_kkt = coo_matrix(kkt.to_local_array())
        linear_solver.do_symbolic_factorization(local_kkt)
        linear_solver.do_numeric_factorization(local_kkt)
        sol = linear_solver.do_back_solve(rhs.make_local_copy())
        interface.set_primal_dual_kkt_solution(sol)
        interface.set_primals(interface.get_primals() + interface.get_delta_primals())
        interface.set_slacks(interface.get_slacks() + interface.get_delta_slacks())
        interface.set_duals_eq(interface.get_duals_eq() + interface.get_delta_duals_eq())
        interface.set_duals_ineq(interface.get_duals_ineq() + interface.get_delta_duals_ineq())
        interface.set_duals_primals_lb(interface.get_duals_primals_lb() + interface.get_delta_duals_primals_lb())
        interface.set_duals_primals_ub(interface.get_duals_primals_ub() + interface.get_delta_duals_primals_ub())
        interface.set_duals_slacks_lb(interface.get_duals_slacks_lb() + interface.get_delta_duals_slacks_lb())
        interface.set_duals_slacks_ub(interface.get_duals_slacks_ub() + interface.get_delta_duals_slacks_ub())
        interface.load_primals_into_pyomo_model()
        x = dict()
        p = dict()
        local_block_indices = _distribute_blocks(num_time_blocks, rank, size)
        for ndx in local_block_indices:
            m = interface.pyomo_model(ndx)
            for t in m.x_time_points:
                if t in x:
                    self.assertAlmostEqual(x[t], m.x[t].value)
                else:
                    x[t] = m.x[t].value
            for t in m.p_time_points:
                p[t] = m.p[t].value

        m = build_time_block(t0=t0,
                             delta_t=delta_t,
                             num_finite_elements=num_finite_elements,
                             constant_control_duration=constant_control_duration,
                             time_scale=time_scale)
        opt = pe.SolverFactory('ipopt')
        res = opt.solve(m)
        assert_optimal_termination(res)
        for _t, _x in x.items():
            self.assertAlmostEqual(_x, m.x[_t].value)
        for _t, _p in p.items():
            self.assertAlmostEqual(_p, m.p[_t].value)

    def _ip_helper(self, with_bounds, with_ineq, linear_solver):
        t0 = 0
        delta_t = 1
        num_finite_elements = 90
        constant_control_duration = 10
        time_scale = 0.1
        num_time_blocks = 3
        interface = Problem(t0=t0,
                            delta_t=delta_t,
                            num_finite_elements=num_finite_elements,
                            constant_control_duration=constant_control_duration,
                            time_scale=time_scale,
                            num_time_blocks=num_time_blocks,
                            with_bounds=with_bounds,
                            with_ineq=with_ineq)
        options = parapint.algorithms.IPOptions()
        options.linalg.solver = linear_solver
        status = parapint.algorithms.ip_solve(interface=interface, options=options)
        self.assertEqual(status, parapint.algorithms.InteriorPointStatus.optimal)
        interface.load_primals_into_pyomo_model()
        x = dict()
        p = dict()
        local_block_indices = _distribute_blocks(num_time_blocks, rank, size)
        for ndx in local_block_indices:
            m = interface.pyomo_model(ndx)
            for t in m.x_time_points:
                if t in x:
                    self.assertAlmostEqual(x[t], m.x[t].value)
                else:
                    x[t] = m.x[t].value
            for t in m.p_time_points:
                p[t] = m.p[t].value

        m = build_time_block(t0=t0,
                             delta_t=delta_t,
                             num_finite_elements=num_finite_elements,
                             constant_control_duration=constant_control_duration,
                             time_scale=time_scale,
                             with_bounds=with_bounds,
                             with_ineq=with_ineq)
        opt = pe.SolverFactory('ipopt')
        res = opt.solve(m, tee=True)
        assert_optimal_termination(res)
        for _t, _x in x.items():
            self.assertAlmostEqual(_x, m.x[_t].value)
        for _t, _p in p.items():
            self.assertAlmostEqual(_p, m.p[_t].value)

    def test_interface_with_ip_sc(self):
        linear_solver = parapint.linalg.MPISchurComplementLinearSolver(subproblem_solvers={0: parapint.linalg.ScipyInterface(compute_inertia=True),
                                                                                          1: parapint.linalg.ScipyInterface(compute_inertia=True),
                                                                                          2: parapint.linalg.ScipyInterface(compute_inertia=True)},
                                                                      schur_complement_solver=parapint.linalg.ScipyInterface(compute_inertia=True))
        self._ip_helper(with_bounds=False, with_ineq=False, linear_solver=linear_solver)

    def test_interface_with_ip_bounds_sc(self):
        linear_solver = parapint.linalg.MPISchurComplementLinearSolver(subproblem_solvers={0: parapint.linalg.ScipyInterface(compute_inertia=True),
                                                                                          1: parapint.linalg.ScipyInterface(compute_inertia=True),
                                                                                          2: parapint.linalg.ScipyInterface(compute_inertia=True)},
                                                                      schur_complement_solver=parapint.linalg.ScipyInterface(compute_inertia=True))
        self._ip_helper(with_bounds=True, with_ineq=False, linear_solver=linear_solver)

    def test_interface_with_ip_ineq_sc(self):
        linear_solver = parapint.linalg.MPISchurComplementLinearSolver(subproblem_solvers={0: parapint.linalg.ScipyInterface(compute_inertia=True),
                                                                                          1: parapint.linalg.ScipyInterface(compute_inertia=True),
                                                                                          2: parapint.linalg.ScipyInterface(compute_inertia=True)},
                                                                      schur_complement_solver=parapint.linalg.ScipyInterface(compute_inertia=True))
        self._ip_helper(with_bounds=False, with_ineq=True, linear_solver=linear_solver)


if __name__ == '__main__':
    unittest.main()
