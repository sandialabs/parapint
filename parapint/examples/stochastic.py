import pyomo.environ as pe
from pyomo.core.base.block import _BlockData
import math
from typing import Tuple, Sequence, Any, Dict
from pyomo.core.base.var import _GeneralVarData
import parapint
from mpi4py import MPI
import logging
import numpy as np
import matplotlib.pyplot as plt


"""
To run this example:

mpirun -np 3 python -m mpi4py stochastic.py
"""


class Farmer(object):
    def __init__(self):
        self.crops = ['WHEAT', 'CORN', 'SUGAR_BEETS']
        self.total_acreage = 500
        self.PriceQuota = {'WHEAT': 100000.0, 'CORN': 100000.0, 'SUGAR_BEETS': 6000.0}
        self.SubQuotaSellingPrice = {'WHEAT': 170.0, 'CORN': 150.0, 'SUGAR_BEETS': 36.0}
        self.SuperQuotaSellingPrice = {'WHEAT': 0.0, 'CORN': 0.0, 'SUGAR_BEETS': 10.0}
        self.CattleFeedRequirement = {'WHEAT': 200.0, 'CORN': 240.0, 'SUGAR_BEETS': 0.0}
        self.PurchasePrice = {'WHEAT': 238.0, 'CORN': 210.0, 'SUGAR_BEETS': 100000.0}
        self.PlantingCostPerAcre = {'WHEAT': 150.0, 'CORN': 230.0, 'SUGAR_BEETS': 260.0}
        self.scenarios = ['BelowAverageScenario', 'AverageScenario', 'AboveAverageScenario']
        self.crop_yield = dict()
        self.crop_yield['BelowAverageScenario'] = {'WHEAT': 2.0, 'CORN': 2.4, 'SUGAR_BEETS': 16.0}
        self.crop_yield['AverageScenario'] = {'WHEAT': 2.5, 'CORN': 3.0, 'SUGAR_BEETS': 20.0}
        self.crop_yield['AboveAverageScenario'] = {'WHEAT': 3.0, 'CORN': 3.6, 'SUGAR_BEETS': 24.0}
        self.scenario_probabilities = dict()
        self.scenario_probabilities['BelowAverageScenario'] = 0.3333
        self.scenario_probabilities['AverageScenario'] = 0.3334
        self.scenario_probabilities['AboveAverageScenario'] = 0.3333


def create_scenario(farmer: Farmer, scenario: str):
    m = pe.ConcreteModel()

    m.crops = pe.Set(initialize=farmer.crops)
    m.devoted_acreage = pe.Var(m.crops, bounds=(0, farmer.total_acreage))
    m.total_acreage_con = pe.Constraint(expr=sum(m.devoted_acreage.values()) <= farmer.total_acreage)

    m.QuantitySubQuotaSold = pe.Var(m.crops, bounds=(0.0, None))
    m.QuantitySuperQuotaSold = pe.Var(m.crops, bounds=(0.0, None))
    m.QuantityPurchased = pe.Var(m.crops, bounds=(0.0, None))

    def EnforceCattleFeedRequirement_rule(m, i):
        return (farmer.CattleFeedRequirement[i] <= (farmer.crop_yield[scenario][i] * m.devoted_acreage[i]) +
                m.QuantityPurchased[i] - m.QuantitySubQuotaSold[i] - m.QuantitySuperQuotaSold[i])
    m.EnforceCattleFeedRequirement = pe.Constraint(m.crops, rule=EnforceCattleFeedRequirement_rule)

    def LimitAmountSold_rule(m, i):
        return (m.QuantitySubQuotaSold[i] +
                m.QuantitySuperQuotaSold[i] -
                (farmer.crop_yield[scenario][i] * m.devoted_acreage[i]) <= 0.0)
    m.LimitAmountSold = pe.Constraint(m.crops, rule=LimitAmountSold_rule)

    def EnforceQuotas_rule(m, i):
        return 0.0, m.QuantitySubQuotaSold[i], farmer.PriceQuota[i]
    m.EnforceQuotas = pe.Constraint(m.crops, rule=EnforceQuotas_rule)

    obj_expr = sum(farmer.PurchasePrice[crop] * m.QuantityPurchased[crop] for crop in m.crops)
    obj_expr -= sum(farmer.SubQuotaSellingPrice[crop] * m.QuantitySubQuotaSold[crop] for crop in m.crops)
    obj_expr -= sum(farmer.SuperQuotaSellingPrice[crop] * m.QuantitySuperQuotaSold[crop] for crop in m.crops)
    obj_expr += sum(farmer.PlantingCostPerAcre[crop] * m.devoted_acreage[crop] for crop in m.crops)

    m.obj = pe.Objective(expr=farmer.scenario_probabilities[scenario] * obj_expr)

    return m


comm: MPI.Comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    logging.basicConfig(level=logging.INFO)


class Problem(parapint.interfaces.MPIStochasticSchurComplementInteriorPointInterface):
    def __init__(self,
                 farmer: Farmer):
        self.farmer: Farmer = farmer
        first_stage_var_ids = [('devoted_acreage', i) for i in self.farmer.crops]
        super(Problem, self).__init__(scenarios=self.farmer.scenarios,
                                      nonanticipative_var_identifiers=first_stage_var_ids,
                                      comm=comm)

    def build_model_for_scenario(self,
                                 scenario_identifier: Any) -> Tuple[_BlockData, Dict[Any, _GeneralVarData]]:
        m = create_scenario(farmer=self.farmer, scenario=scenario_identifier)
        first_stage_vars = {('devoted_acreage', i): m.devoted_acreage[i] for i in self.farmer.crops}
        return m, first_stage_vars


def main(farmer: Farmer, subproblem_solver_class, subproblem_solver_options):
    interface = Problem(farmer=farmer)
    linear_solver = parapint.linalg.MPISchurComplementLinearSolver(
        subproblem_solvers={ndx: subproblem_solver_class(**subproblem_solver_options) for ndx in range(len(farmer.scenarios))},
        schur_complement_solver=subproblem_solver_class(**subproblem_solver_options))
    options = parapint.algorithms.IPOptions()
    options.linalg.solver = linear_solver
    status = parapint.algorithms.ip_solve(interface=interface, options=options)
    assert status == parapint.algorithms.InteriorPointStatus.optimal
    interface.load_primals_into_pyomo_model()

    # gather the results and plot
    if rank == 0:
        interface.pyomo_model(scenario_id=farmer.scenarios[0]).devoted_acreage.display()

    return interface


if __name__ == '__main__':
    # cntl[1] is the MA27 pivot tolerance
    farmer = Farmer()
    main(farmer=farmer,
         subproblem_solver_class=parapint.linalg.InteriorPointMA27Interface,
         subproblem_solver_options={'cntl_options': {1: 1e-6}})
