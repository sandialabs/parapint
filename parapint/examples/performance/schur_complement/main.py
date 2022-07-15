from parapint.examples.performance.schur_complement.create_model import Model, MPIModel
from parapint.linalg import (
    SchurComplementLinearSolver,
    InteriorPointMA27Interface,
    MPISchurComplementLinearSolver,
    ScipyInterface,
)
import time
from mpi4py import MPI
import argparse


"""
run with, for example,

python main.py --method fs --n_blocks 4

or 

mpirun -np 4 python -m mpi4py main.py --method psc --n_blocks 4
"""


class Result(object):
    def __init__(self):
        self.max_err = None
        self.symbolic_time = None
        self.numeric_time = None
        self.back_solve_time = None
        self.total_time = None


def helper(m, solver):
    comm: MPI.Comm = MPI.COMM_WORLD

    kkt = m.build_kkt()
    rhs = m.build_rhs()

    comm.Barrier()
    t0 = time.time()
    solver.do_symbolic_factorization(kkt)
    t1 = time.time()
    solver.do_numeric_factorization(kkt)
    t2 = time.time()
    x = solver.do_back_solve(rhs)
    t3 = time.time()

    res = Result()
    res.max_err = m.check_result(x)
    res.symbolic_time = t1 - t0
    res.numeric_time = t2 - t1
    res.back_solve_time = t3 - t2
    res.total_time = t3 - t0

    res.symbolic_time = comm.allreduce(res.symbolic_time, MPI.MAX)
    res.numeric_time = comm.allreduce(res.numeric_time, MPI.MAX)
    res.back_solve_time = comm.allreduce(res.back_solve_time, MPI.MAX)
    res.total_time = comm.allreduce(res.total_time, MPI.MAX)

    return res


def run(args, linear_solver_str="ma27", n_q_per_block=5000, n_y_multiplier=120):
    comm: MPI.Comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if args.method != "psc":
        if size != 1:
            raise RuntimeError("running serial code with multiple processes")

    n_blocks = args.n_blocks
    n_theta = 10
    A_nnz_per_row = 3

    if linear_solver_str == "ma27":
        linear_solver_class = InteriorPointMA27Interface
        linear_solver_options = dict(cntl_options={1: 1e-6})
    else:
        if linear_solver_str != "scipy":
            raise ValueError("linear_solver_str should be ma27 or scipy")
        linear_solver_class = ScipyInterface
        linear_solver_options = dict(compute_inertia=False)

    if args.method == "fs":
        model_class = Model
        solver = linear_solver_class(**linear_solver_options)
    elif args.method == "ssc":
        model_class = Model
        solver = SchurComplementLinearSolver(
            subproblem_solvers={
                i: linear_solver_class(**linear_solver_options) for i in range(n_blocks)
            },
            schur_complement_solver=linear_solver_class(**linear_solver_options),
        )
    else:
        model_class = MPIModel
        solver = MPISchurComplementLinearSolver(
            subproblem_solvers={
                i: linear_solver_class(**linear_solver_options) for i in range(n_blocks)
            },
            schur_complement_solver=linear_solver_class(**linear_solver_options),
        )

    m = model_class(
        n_blocks=n_blocks,
        n_q_per_block=n_q_per_block,
        n_y_multiplier=n_y_multiplier,
        n_theta=n_theta,
        A_nnz_per_row=A_nnz_per_row,
    )

    res = helper(m, solver)

    method_map = {
        "fs": "Full Space",
        "ssc": "Serial Schur-Complement",
        "psc": "Parallel Schur-Complement",
    }

    if rank == 0:
        print(
            f"{'method':<30}"
            f"{'# processes':<15}"
            f"{'# blocks':<15}"
            f"{'n_q_per_block':<15}"
            f"{'n_y_multiplier':<15}"
            f"{'n_theta':<15}"
            f"{'A NNZ per row':<15}"
            f"{'Est Err':<15}"
            f"{'Symb Fact (s)':<15}"
            f"{'Num Fact (s)':<15}"
            f"{'Back Solve (s)':<15}"
            f"{'Total Time (s)':<15}"
        )
        print(
            f"{method_map[args.method]:<30}"
            f"{size:<15}"
            f"{n_blocks:<15}"
            f"{n_q_per_block:<15}"
            f"{n_y_multiplier:<15}"
            f"{n_theta:<15}"
            f"{A_nnz_per_row:<15}"
            f"{res.max_err:<15.3f}"
            f"{res.symbolic_time:<15.3f}"
            f"{res.numeric_time:<15.3f}"
            f"{res.back_solve_time:<15.3f}"
            f"{res.total_time:<15.3f}"
        )

    return res.max_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", type=str, required=True, choices={"fs", "ssc", "psc"}
    )
    parser.add_argument("--n_blocks", type=int, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
