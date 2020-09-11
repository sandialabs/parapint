import pyomo.environ as pe
import parapint


def main():
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.y = pe.Var()
    m.obj = pe.Objective(expr=m.x**2 + m.y**2)
    m.c1 = pe.Constraint(expr=m.y >= (m.x - 1)**2)
    m.c2 = pe.Constraint(expr=m.y == pe.exp(m.x))

    interface = parapint.interfaces.InteriorPointInterface(m)
    options = {1: 1e-6}  # MA27 pivot tolerance
    linear_solver = parapint.linalg.InteriorPointMA27Interface(cntl_options=options)
    opt = parapint.interior_point.InteriorPointSolver(linear_solver=linear_solver)
    status = opt.solve(interface)
    assert status == parapint.interior_point.InteriorPointStatus.optimal
    interface.load_primals_into_pyomo_model()
    m.x.pprint()
    m.y.pprint()


if __name__ == '__main__':
    main()
