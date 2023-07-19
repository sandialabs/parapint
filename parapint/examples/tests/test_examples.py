import parapint
from parapint import examples
import unittest
import pytest
from mpi4py import MPI
from parapint.examples.performance.schur_complement import main as sc_main


class TestExamples(unittest.TestCase):
    @pytest.mark.serial
    @pytest.mark.fast
    def test_interior_point_example(self):
        linear_solver = parapint.linalg.ScipyInterface(compute_inertia=True)
        m = examples.interior_point.main(linear_solver=linear_solver)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

    @pytest.mark.parallel
    @pytest.mark.medium
    @pytest.mark.one_proc
    @pytest.mark.two_proc
    @pytest.mark.three_proc
    def test_stochastic(self):
        comm: MPI.Comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        farmer = examples.stochastic.Farmer()
        interface = examples.stochastic.main(farmer=farmer,
                                             subproblem_solver_class=parapint.linalg.ScipyInterface,
                                             subproblem_solver_options={'compute_inertia': True})
        self.assertAlmostEqual(interface.pyomo_model(farmer.scenarios[rank]).devoted_acreage['CORN'].value, 80, 5)
        self.assertAlmostEqual(interface.pyomo_model(farmer.scenarios[rank]).devoted_acreage['SUGAR_BEETS'].value, 250, 5)
        self.assertAlmostEqual(interface.pyomo_model(farmer.scenarios[rank]).devoted_acreage['WHEAT'].value, 170, 5)

    @pytest.mark.parallel
    @pytest.mark.medium
    @pytest.mark.three_proc
    def test_dynamics(self):
        comm: MPI.Comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        self.assertEqual(size, 3)
        interface = examples.dynamics.main(subproblem_solver_class=parapint.linalg.ScipyInterface,
                                           subproblem_solver_options={'compute_inertia': True},
                                           show_plot=False)
        if rank == 0:
            self.assertAlmostEqual(interface.pyomo_model(ndx=0).p[0].value, 1.6046242850486279)
            self.assertAlmostEqual(interface.pyomo_model(ndx=0).p[10].value, 2.0)
            self.assertAlmostEqual(interface.pyomo_model(ndx=0).p[20].value, 1.4792062911745605)
        if rank == 1:
            self.assertAlmostEqual(interface.pyomo_model(ndx=1).p[30].value, 0.5082444341496647)
            self.assertAlmostEqual(interface.pyomo_model(ndx=1).p[40].value, -0.009859487375413882)
            self.assertAlmostEqual(interface.pyomo_model(ndx=1).p[50].value, 0.40043954978583834)
        if rank == 2:
            self.assertAlmostEqual(interface.pyomo_model(ndx=2).p[60].value, 1.3619861771562247)
            self.assertAlmostEqual(interface.pyomo_model(ndx=2).p[70].value, 1.99059057528143)
            self.assertAlmostEqual(interface.pyomo_model(ndx=2).p[80].value, 1.7102013685364827)

    @pytest.mark.parallel
    @pytest.mark.slow
    @pytest.mark.all_proc
    def test_burgers(self):
        comm: MPI.Comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        args = examples.burgers.Args()
        args.nblocks = 4
        args.nfe_x = 10
        args.nfe_t = 12
        args.plot = True
        args.show_plot = False
        interface = examples.burgers.main(args=args,
                                          subproblem_solver_class=parapint.linalg.ScipyInterface,
                                          subproblem_solver_options={'compute_inertia': True})

    @pytest.mark.medium
    @pytest.mark.serial
    def test_schur_complement_fs(self):
        class Args:
            def __init__(self):
                self.method = 'fs'
                self.n_blocks = 3

        args = Args()
        max_err = sc_main.run(args, linear_solver_str='scipy', n_q_per_block=500, n_y_multiplier=12)
        self.assertAlmostEqual(max_err, 0.3163456780448639)

    @pytest.mark.parallel
    @pytest.mark.medium
    @pytest.mark.all_proc
    def test_schur_complement_psc(self):
        class Args:
            def __init__(self):
                self.method = 'psc'
                self.n_blocks = 3

        args = Args()
        max_err = sc_main.run(args, linear_solver_str='scipy', n_q_per_block=500, n_y_multiplier=12)
        self.assertAlmostEqual(max_err, 0.3163456780448639)
