import pyomo.environ as pe
from pyomo import dae
import parapint
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from mpi4py import MPI
import math
import logging
import argparse


"""
Run this example with, e.g., 

mpirun -np 4 python -m mpi4py burgers.py --nfe_x 50 --nfe_t 200 --nblocks 4

If you run it with the --plot, make sure you don't use too many finite elements, or it will take forever to plot.
"""

comm: MPI.Comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    logging.basicConfig(level=logging.INFO)


class Args(object):
    def __init__(self):
        self.nfe_x = 50
        self.nfe_t = 200
        self.nblocks = 4
        self.plot = True
        self.show_plot = True

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--nfe_x', type=int, required=True, help='number of finite elements for x')
        parser.add_argument('--nfe_t', type=int, required=True, help='number of finite elements for t')
        parser.add_argument('--nblocks', type=int, required=True, help='number of time blocks for schur complement')
        parser.add_argument('--no_plot', action='store_true')
        parser.add_argument('--no_show_plot', action='store_true')
        args = parser.parse_args()
        self.nfe_x = args.nfe_x
        self.nfe_t = args.nfe_t
        self.nblocks = args.nblocks
        self.plot = not args.no_plot
        self.show_plot = not args.no_show_plot


class BurgersInterface(parapint.interfaces.MPIDynamicSchurComplementInteriorPointInterface):
    def __init__(self, start_t, end_t, num_time_blocks, nfe_t, nfe_x):
        self.nfe_x = nfe_x
        self.dt = (end_t - start_t) / float(nfe_t)
        self.last_t = None
        super(BurgersInterface, self).__init__(start_t=start_t,
                                               end_t=end_t,
                                               num_time_blocks=num_time_blocks,
                                               comm=comm)

    def build_burgers_model(self, nfe_x=50, nfe_t=50, start_t=0, end_t=1, add_init_conditions=True):
        dt = (end_t - start_t) / float(nfe_t)

        start_x = 0
        end_x = 1
        dx = (end_x - start_x) / float(nfe_x)

        m = pe.Block(concrete=True)
        m.omega = pe.Param(initialize=0.02)
        m.v = pe.Param(initialize=0.01)
        m.r = pe.Param(initialize=0)

        m.x = dae.ContinuousSet(bounds=(start_x, end_x))
        m.t = dae.ContinuousSet(bounds=(start_t, end_t))

        m.y = pe.Var(m.x, m.t)
        m.dydt = dae.DerivativeVar(m.y, wrt=m.t)
        m.dydx = dae.DerivativeVar(m.y, wrt=m.x)
        m.dydx2 = dae.DerivativeVar(m.y, wrt=(m.x, m.x))

        m.u = pe.Var(m.x, m.t)

        def _y_init_rule(m, x):
            if x <= 0.5 * end_x:
                return 1
            return 0

        m.y0 = pe.Param(m.x, default=_y_init_rule)

        def _upper_x_bound(m, t):
            return m.y[end_x, t] == 0

        m.upper_x_bound = pe.Constraint(m.t, rule=_upper_x_bound)

        def _lower_x_bound(m, t):
            return m.y[start_x, t] == 0

        m.lower_x_bound = pe.Constraint(m.t, rule=_lower_x_bound)

        def _upper_x_ubound(m, t):
            return m.u[end_x, t] == 0

        m.upper_x_ubound = pe.Constraint(m.t, rule=_upper_x_ubound)

        def _lower_x_ubound(m, t):
            return m.u[start_x, t] == 0

        m.lower_x_ubound = pe.Constraint(m.t, rule=_lower_x_ubound)

        def _lower_t_bound(m, x):
            if x == start_x or x == end_x:
                return pe.Constraint.Skip
            return m.y[x, start_t] == m.y0[x]

        def _lower_t_ubound(m, x):
            if x == start_x or x == end_x:
                return pe.Constraint.Skip
            return m.u[x, start_t] == 0

        if add_init_conditions:
            m.lower_t_bound = pe.Constraint(m.x, rule=_lower_t_bound)
            m.lower_t_ubound = pe.Constraint(m.x, rule=_lower_t_ubound)

        # PDE
        def _pde(m, x, t):
            if t == start_t or x == end_x or x == start_x:
                e = pe.Constraint.Skip
            else:
                # print(foo.last_t, t-dt, abs(foo.last_t - (t-dt)))
                # assert math.isclose(foo.last_t, t - dt, abs_tol=1e-6)
                e = m.dydt[x, t] - m.v * m.dydx2[x, t] + m.dydx[x, t] * m.y[x, t] == m.r + m.u[x, self.last_t]
            self.last_t = t
            return e

        m.pde = pe.Constraint(m.x, m.t, rule=_pde)

        # Discretize Model
        disc = pe.TransformationFactory('dae.finite_difference')
        disc.apply_to(m, nfe=nfe_t, wrt=m.t, scheme='BACKWARD')
        disc.apply_to(m, nfe=nfe_x, wrt=m.x, scheme='CENTRAL')

        # Solve control problem using Pyomo.DAE Integrals
        def _intX(m, x, t):
            return (m.y[x, t] - m.y0[x]) ** 2 + m.omega * m.u[x, t] ** 2

        m.intX = dae.Integral(m.x, m.t, wrt=m.x, rule=_intX)

        def _intT(m, t):
            return m.intX[t]

        m.intT = dae.Integral(m.t, wrt=m.t, rule=_intT)

        def _obj(m):
            e = 0.5 * m.intT
            for x in sorted(m.x):
                if x == start_x or x == end_x:
                    pass
                else:
                    e += 0.5 * 0.5 * dx * dt * m.omega * m.u[x, start_t] ** 2
            return e

        m.obj = pe.Objective(rule=_obj)

        return m

    def build_model_for_time_block(self, ndx, start_t, end_t, add_init_conditions):
        dt = self.dt
        nfe_t = math.ceil((end_t - start_t) / dt)
        m = self.build_burgers_model(nfe_x=self.nfe_x, nfe_t=nfe_t, start_t=start_t, end_t=end_t,
                                     add_init_conditions=add_init_conditions)

        return (m,
                ([m.y[x, start_t] for x in sorted(m.x) if x not in {0, 1}]),
                ([m.y[x, end_t] for x in sorted(m.x) if x not in {0, 1}]))

    def plot_results(self, show_plot=True):
        y_pts = list()
        u_pts = list()
        for block_ndx in self.local_block_indices:
            m = self.pyomo_model(block_ndx)
            for x in m.x:
                for t in m.t:
                    y_pts.append((x, t, m.y[x, t].value))
                    u_pts.append((x, t, m.u[x, t].value))
        y_pts = comm.allgather(y_pts)
        u_pts = comm.allgather(u_pts)
        if rank == 0:
            _tmp_y = list()
            _tmp_u = list()
            for i in y_pts:
                _tmp_y.extend(i)
            for i in u_pts:
                _tmp_u.extend(i)
            y_pts = _tmp_y
            u_pts = _tmp_u
            y_pts.sort(key=lambda x: x[0])
            y_pts.sort(key=lambda x: x[1])
            u_pts.sort(key=lambda x: x[0])
            u_pts.sort(key=lambda x: x[1])
            x_set = set()
            t_set = set()
            y_dict = dict()
            u_dict = dict()
            for x, t, y in y_pts:
                x_set.add(x)
                t_set.add(t)
                y_dict[x, t] = y
            for x, t, u in u_pts:
                u_dict[x, t] = u
            x_list = list(x_set)
            t_list = list(t_set)
            x_list.sort()
            t_list.sort()
            y_list = list()
            u_list = list()
            all_x = list()
            all_t = list()
            for x in x_list:
                tmp_y = list()
                tmp_u = list()
                tmp_x = list()
                tmp_t = list()
                for t in t_list:
                    tmp_y.append(y_dict[x, t])
                    tmp_u.append(u_dict[x, t])
                    tmp_x.append(x)
                    tmp_t.append(t)
                y_list.append(tmp_y)
                u_list.append(tmp_u)
                all_x.append(tmp_x)
                all_t.append(tmp_t)

            colors = cm.jet(y_list)
            rcount, ccount, _ = colors.shape
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(np.array(all_x), np.array(all_t), np.array(y_list), rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
            surf.set_facecolor((0, 0, 0, 0))
            ax.set_xlabel('x')
            ax.set_ylabel('t')
            ax.set_zlabel('y')
            if show_plot:
                plt.show()
            plt.close()

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(np.array(all_x), np.array(all_t), np.array(u_list), rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
            surf.set_facecolor((0, 0, 0, 0))
            ax.set_xlabel('x')
            ax.set_ylabel('t')
            ax.set_zlabel('u')
            if show_plot:
                plt.show()
            plt.close()


def main(args, subproblem_solver_class, subproblem_solver_options):
    interface = BurgersInterface(start_t=0,
                                 end_t=1,
                                 num_time_blocks=args.nblocks,
                                 nfe_t=args.nfe_t,
                                 nfe_x=args.nfe_x)
    linear_solver = parapint.linalg.MPISchurComplementLinearSolver(
        subproblem_solvers={ndx: subproblem_solver_class(**subproblem_solver_options) for ndx in range(args.nblocks)},
        schur_complement_solver=subproblem_solver_class(**subproblem_solver_options))
    options = parapint.algorithms.IPOptions()
    options.linalg.solver = linear_solver
    status = parapint.algorithms.ip_solve(interface=interface, options=options)
    assert status == parapint.algorithms.InteriorPointStatus.optimal
    interface.load_primals_into_pyomo_model()

    if args.plot:
        interface.plot_results(show_plot=args.show_plot)

    return interface


if __name__ == '__main__':
    args = Args()
    args.parse_arguments()
    # cntl[1] is the MA27 pivot tolerance
    main(args=args,
         subproblem_solver_class=parapint.linalg.InteriorPointMA27Interface,
         subproblem_solver_options={'cntl_options': {1: 1e-6}})
