Solving Dynamic Optimization Problems with Schur-Complement Decomposition
=========================================================================

In order to solve a dynamic optimization problem with schur-complement
decomposition, you must create a class which inherits from
:py:class:`~parapint.interfaces.schur_complement.mpi_sc_ip_interface.MPIDynamicSchurComplementInteriorPointInterface`.
This class must implement the method
:py:meth:`~parapint.interfaces.schur_complement.mpi_sc_ip_interface.MPIDynamicSchurComplementInteriorPointInterface.build_model_for_time_block`::

    import parapint

    class Problem(parapint.interfaces.MPIDynamicSchurComplementInteriorPointInterface):
        def __init__(self, your_arguments):
	    # do anything you need to here
	    super(Problem, self).__init__(start_t, end_t, num_time_blocks, mpi_comm)

	def build_model_for_time_block(self, ndx, start_t, end_t, add_init_conditions):
	    # build the dynamic optimization problem with Pyomo over the time horizon
	    # [start_t, end_t] and return the model along with two lists. The first
	    # list should be a list of pyomo variables corresponding to the states at
	    # start_t. The second list should be a list of pyomo variables
	    # corresponding to the states at end_t. Continuity will be enforced
	    # between the states at end_t for one time block
	    # and the states at start_t for the next time block. It is very important for
	    # the ordering of the state variables to be the same for every time block.

	    return model, start_states, end_states

    problem = Problem(some_arguments)

The
:py:meth:`~parapint.interfaces.schur_complement.mpi_sc_ip_interface.MPIDynamicSchurComplementInteriorPointInterface.build_model_for_time_block`
method will be called once for every time block. It will be called
within the call to
:py:meth:`~parapint.interfaces.schur_complement.mpi_sc_ip_interface.MPIDynamicSchurComplementInteriorPointInterface.__init__`
of the super class
(:py:class:`~parapint.interfaces.schur_complement.mpi_sc_ip_interface.MPIDynamicSchurComplementInteriorPointInterface`).
Therefore, if you override the `__init__` method, it is very important
to still call the `__init__` method of the base class as shown above.
There is an example class in schur_complement.py in the examples directory within
Parapint.

In addition to the implementation of the class described above, you
must create an instance of 
:py:class:`~parapint.linalg.schur_complement.mpi_explicit_schur_complement.MPISchurComplementLinearSolver`.
This linear solver requires a sub-solver for every time block::

    cntl_options = {1: 1e-6}  # the pivot tolerance
    sub_solvers = {ndx: parapint.linalg.InteriorPointMA27Interface(cntl_options=cntl_options) for ndx in range(num_time_blocks)}
    schur_complement_solver = parapint.linalg.InteriorPointMA27Interface(cntl_options=cntl_options)
    linear_solver = parapint.linalg.MPISchurComplementLinearSolver(subproblem_solvers=sub_solvers,
                                                                  schur_complement_solver=schur_complement_solver)

The linear solver and interface instances can then be used with the interior point algorithm::

    options = parapint.algorithms.IPOptions()
    options.linalg.solver = linear_solver
    status = parapint.algorithms.ip_solve(interface, options)
    assert status == parapint.interior_point.InteriorPointStatus.optimal
    problem.load_primals_into_pyomo_model()
    for ndx in problem.local_block_indices:
        model = problem.pyomo_model(ndx)
	model.pprint()
