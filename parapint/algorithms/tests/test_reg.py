import unittest
import pyomo.environ as pe
from pyomo.core.base import ConcreteModel, Var, Constraint, Objective
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.pynumero.asl import AmplInterface
import parapint
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
from parapint.algorithms.interior_point import numeric_factorization
ma27_available = MA27Interface.available()
mumps, mumps_available = attempt_import('mumps', 'Interior point requires mumps')
asl_available = AmplInterface.available()
if not asl_available:
    raise unittest.SkipTest('Regularization tests require ASL')


def make_model():
    m = ConcreteModel()
    m.x = Var([1,2,3], initialize=0)
    m.f = Var([1,2,3], initialize=0)
    m.F = Var(initialize=0)
    m.f[1].fix(1)
    m.f[2].fix(2)

    m.sum_con = Constraint(expr= 
            (1 == m.x[1] + m.x[2] + m.x[3]))
    def bilin_rule(m, i):
        return m.F*m.x[i] == m.f[i]
    m.bilin_con = Constraint([1,2,3], rule=bilin_rule)

    m.obj = Objective(expr=m.F**2)

    return m


def make_model_2():
    m = ConcreteModel()
    m.x = Var(initialize=0.1, bounds=(0, 1))
    m.y = Var(initialize=0.1, bounds=(0, 1))
    m.obj = Objective(expr=-m.x**2 - m.y**2)
    m.c = Constraint(expr=m.y <= pe.exp(-m.x))
    return m


class TestRegularization(unittest.TestCase):
    def _test_regularization(self, linear_solver):
        m = make_model()
        interface = parapint.interfaces.InteriorPointInterface(m)
        options = parapint.algorithms.IPOptions()
        options.linalg.solver = linear_solver

        interface.set_barrier_parameter(1e-1)

        # Evaluate KKT matrix before any iterations
        kkt = interface.evaluate_primal_dual_kkt_matrix()
        linear_solver.do_symbolic_factorization(kkt)
        reg_coef = numeric_factorization(interface=interface, kkt=kkt, options=options,
                                         inertia_coef=options.inertia_correction.init_coef)

        # Expected regularization coefficient:
        self.assertAlmostEqual(reg_coef, 1e-4)

        desired_n_neg_evals = (interface.n_eq_constraints() +
                               interface.n_ineq_constraints())

        # Expected inertia:
        n_pos_evals, n_neg_evals, n_null_evals = linear_solver.get_inertia()
        self.assertEqual(n_null_evals, 0)
        self.assertEqual(n_neg_evals, desired_n_neg_evals)

    @unittest.skipIf(not mumps_available, 'Mumps is not available')
    def test_mumps(self):
        solver = parapint.linalg.MumpsInterface()
        self._test_regularization(solver)

    def test_scipy(self):
        solver = parapint.linalg.ScipyInterface(compute_inertia=True)
        self._test_regularization(solver)

    @unittest.skipIf(not ma27_available, 'MA27 is not available')
    def test_ma27(self):
        solver = parapint.linalg.InteriorPointMA27Interface(icntl_options={1: 0, 2: 0})
        self._test_regularization(solver)

    def _test_regularization_2(self, linear_solver):
        m = make_model_2()
        interface = parapint.interfaces.InteriorPointInterface(m)
        options = parapint.algorithms.IPOptions()
        options.linalg.solver = linear_solver

        status = parapint.algorithms.ip_solve(interface=interface,
                                              options=options)
        self.assertEqual(status, parapint.algorithms.InteriorPointStatus.optimal)
        interface.load_primals_into_pyomo_model()
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, pe.exp(-1))

    @unittest.skipIf(not mumps_available, 'Mumps is not available')
    def test_mumps_2(self):
        solver = parapint.linalg.MumpsInterface()
        self._test_regularization_2(solver)

    def test_scipy_2(self):
        solver = parapint.linalg.ScipyInterface(compute_inertia=True)
        self._test_regularization_2(solver)

    @unittest.skipIf(not ma27_available, 'MA27 is not available')
    def test_ma27_2(self):
        solver = parapint.linalg.InteriorPointMA27Interface(icntl_options={1: 0, 2: 0})
        self._test_regularization_2(solver)


if __name__ == '__main__':
    #
    unittest.main()
    # test_reg = TestRegularization()
    # test_reg.test_regularize_mumps()
    # test_reg.test_regularize_scipy()
    
