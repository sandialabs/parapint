import pyomo.environ as pe
from pyomo.core.base.block import _BlockData
import math
from typing import Tuple, Sequence
from pyomo.core.base.var import _GeneralVarData
import parapint
from mpi4py import MPI
import logging
import numpy as np
import matplotlib.pyplot as plt


"""
The example problem is

min integral from t0 to tf of [x(t) - sin(time_scale * t) - 1]**2
s.t.
dx/dt = p(t) - x(t)
p(t) <= 2

Using trapezoid rule for integral and implicit euler for differential equation

To run the example, use

mpirun -np 3 python -m mpi4py schur_complement.py
"""


comm: MPI.Comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    logging.basicConfig(level=logging.INFO)


def build_time_block(t0: int,
                     delta_t: int,
                     num_finite_elements: int,
                     constant_control_duration: int,
                     time_scale: float) -> _BlockData:
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
        number of finite elements over which the control (p) is constant
    time_scale: float
        coefficient of t within the sin function

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
    bnds = (None, 2)
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

    return m


class Problem(parapint.interfaces.MPIDynamicSchurComplementInteriorPointInterface):
    def __init__(self,
                 t0=0,
                 delta_t=1,
                 num_finite_elements=90,
                 constant_control_duration=10,
                 time_scale=0.1,
                 num_time_blocks=3):
        assert num_finite_elements % num_time_blocks == 0
        self.t0: int = t0
        self.delta_t: int = delta_t
        self.num_finite_elements: int = num_finite_elements
        self.constant_control_duration: int = constant_control_duration
        self.time_scale: float = time_scale
        self.num_time_blocks: int = num_time_blocks
        self.tf = self.t0 + self.delta_t*self.num_finite_elements
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
                             time_scale=self.time_scale)
        start_states = [m.x[start_t]]
        end_states = [m.x[end_t]]
        return m, start_states, end_states


def main(subproblem_solver_class, subproblem_solver_options, show_plot=True):
    t0: int = 0
    delta_t: int = 1
    num_finite_elements: int = 90
    constant_control_duration: int = 10
    time_scale: float = 0.1
    num_time_blocks: int = 3
    interface = Problem(t0=t0,
                        delta_t=delta_t,
                        num_finite_elements=num_finite_elements,
                        constant_control_duration=constant_control_duration,
                        time_scale=time_scale,
                        num_time_blocks=num_time_blocks)
    linear_solver = parapint.linalg.MPISchurComplementLinearSolver(
        subproblem_solvers={ndx: subproblem_solver_class(**subproblem_solver_options) for ndx in range(num_time_blocks)},
        schur_complement_solver=subproblem_solver_class(**subproblem_solver_options))
    options = parapint.algorithms.IPOptions()
    options.linalg.solver = linear_solver
    status = parapint.algorithms.ip_solve(interface=interface, options=options)
    assert status == parapint.algorithms.InteriorPointStatus.optimal
    interface.load_primals_into_pyomo_model()

    # gather the results and plot
    t_x = np.array(list(range(t0, t0+delta_t*(num_finite_elements + 1), delta_t)), dtype=np.int64)
    num_p_elements = int((num_finite_elements * delta_t) / constant_control_duration)
    t_p = np.array(list(range(t0, t0+constant_control_duration*num_p_elements, constant_control_duration)))
    x = np.zeros(t_x.size, dtype=np.double)
    p = np.zeros(t_p.size, dtype=np.double)
    t_x_map = {int(t): ndx for ndx, t in enumerate(t_x)}
    t_p_map = {int(t): ndx for ndx, t in enumerate(t_p)}
    for ndx in interface.local_block_indices:
        m = interface.pyomo_model(ndx)
        _t_x = list(m.x_time_points)
        if ndx != 0:
            _t_x = _t_x[1:]
        for t in _t_x:
            x[t_x_map[t]] = m.x[t].value
        for t in m.p_time_points:
            p[t_p_map[t]] = m.p[t].value
    global_x = np.zeros(x.size, dtype=np.double)
    comm.Allreduce(x, global_x)
    global_p = np.zeros(p.size, dtype=np.double)
    comm.Allreduce(p, global_p)
    if rank == 0:
        plt.plot(t_x, global_x, label='x(t)')
        t_goal = np.linspace(t0, t0+delta_t*(num_finite_elements+1), 1000)
        x_goal = np.sin(time_scale*t_goal) + 1
        plt.plot(t_goal, x_goal, label='sin({0}t) + 1'.format(time_scale))
        plt.step(t_p, global_p, where='post', label='p(t)')
        plt.xlabel('t')
        plt.legend()
        if show_plot:
            plt.show()
        plt.close()

    return interface


if __name__ == '__main__':
    # cntl[1] is the MA27 pivot tolerance
    main(subproblem_solver_class=parapint.linalg.InteriorPointMA27Interface,
         subproblem_solver_options={'cntl_options': {1: 1e-6}})
