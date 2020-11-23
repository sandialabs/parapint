from .base_linear_solver_interface import LinearSolverInterface
from .results import LinearSolverStatus, LinearSolverResults
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
from scipy.sparse import isspmatrix_coo, tril
from pyomo.contrib.pynumero.sparse import BlockVector
from pyomo.common.timing import HierarchicalTimer


class InteriorPointMA27Interface(LinearSolverInterface):
    """
    An interface to HSL's MA27 routines for use with Parapint's interior point algorithm.
    See http://www.hsl.rl.ac.uk/archive/specs/ma27.pdf for details on the use of MA27.

    .. note::
       The pivot tolerance, cntl(1), should be selected carefully. Larger values result in better precision but
       smaller values result in better performance.

    Parameters
    ----------
    cntl_options: dict
        See http://www.hsl.rl.ac.uk/archive/specs/ma27.pdf
    icntl_options: dict
        See http://www.hsl.rl.ac.uk/archive/specs/ma27.pdf
    iw_factor: float
        The factor for memory allocation of the integer working arrays used by MA27.
        This value is increased by the increase_memory_allocation method.
    a_factor: float
        The factor for memory allocation of the A array used by MA28.
        This value is increased by the increase_memory_allocation_method.
    """

    @classmethod
    def getLoggerName(cls):
        return 'ma27'

    def __init__(self, cntl_options=None, icntl_options=None, iw_factor=1.2, a_factor=2):
        self._ma27 = MA27Interface(iw_factor=iw_factor, a_factor=a_factor)

        if cntl_options is None:
            cntl_options = dict()
        if icntl_options is None:
            icntl_options = dict()

        for k, v in cntl_options.items():
            self.set_cntl(k, v)
        for k, v in icntl_options.items():
            self.set_icntl(k, v)

        self._dim = None
        self._num_status = None

    def do_symbolic_factorization(self, matrix, raise_on_error=True, timer=None):
        """
        Perform symbolic factorization. This calls the MA27A/AD routines.

        Parameters
        ----------
        matrix: scipy.sparse.spmatrix or pyomo.contrib.pynumero.sparse.block_matrix.BlockMatrix
            The matrix to factorize
        raise_on_error: bool
            If False, an error will not be raised if an error occurs during symbolic factorization. Instead the
            status attribute of the results object will indicate an error ocurred.
        timer: HierarchicalTimer

        Returns
        -------
        res: LinearSolverResults
            A LinearSolverResults object with a status attribute for the LinearSolverStatus
        """
        self._num_status = None
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        matrix = tril(matrix)
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError('Matrix must be square')
        self._dim = nrows

        stat = self._ma27.do_symbolic_factorization(dim=self._dim, irn=matrix.row, icn=matrix.col)
        res = LinearSolverResults()
        if stat == 0:
            res.status = LinearSolverStatus.successful
        else:
            if raise_on_error:
                raise RuntimeError('Symbolic factorization was not successful; return code: ' + str(stat))
            if stat in {-3, -4}:
                res.status = LinearSolverStatus.not_enough_memory
            elif stat in {-5, 3}:
                res.status = LinearSolverStatus.singular
            else:
                res.status = LinearSolverStatus.error
        return res

    def do_numeric_factorization(self, matrix, raise_on_error=True, timer=None):
        """
        Perform numeric factorization. This calls the MA27B/BD routines.

        Parameters
        ----------
        matrix: scipy.sparse.spmatrix or pyomo.contrib.pynumero.sparse.block_matrix.BlockMatrix
            The matrix to factorize
        raise_on_error: bool
            If False, an error will not be raised if an error occurs during numeric factorization. Instead the
            status attribute of the results object will indicate an error ocurred.
        timer: HierarchicalTimer

        Returns
        -------
        res: LinearSolverResults
            A LinearSolverResults object with a status attribute for the LinearSolverStatus
        """
        if self._dim is None:
            raise RuntimeError('Perform symbolic factorization first!')
        if not isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        matrix = tril(matrix)
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError('Matrix must be square')
        if nrows != self._dim:
            raise ValueError('Matrix dimensions do not match the dimensions of '
                             'the matrix used for symbolic factorization')

        stat = self._ma27.do_numeric_factorization(irn=matrix.row, icn=matrix.col, dim=self._dim, entries=matrix.data)
        res = LinearSolverResults()
        if stat == 0:
            res.status = LinearSolverStatus.successful
        else:
            if raise_on_error:
                raise RuntimeError('Numeric factorization was not successful; return code: ' + str(stat))
            if stat in {-3, -4}:
                res.status = LinearSolverStatus.not_enough_memory
            elif stat in {-5, 3}:
                res.status = LinearSolverStatus.singular
            else:
                res.status = LinearSolverStatus.error

        self._num_status = res.status

        return  res

    def increase_memory_allocation(self, factor):
        """
        Increas the memory allocation for factorization. This method should only be called
        if the results status from do_symbolic_factorization or do_numeric_factorization is
        LinearSolverStatus.not_enough_memory.

        Parameters
        ----------
        factor: float
            The factor by which to increase memory allocation. Should be greater than 1.
        """
        self._ma27.iw_factor *= factor
        self._ma27.a_factor *= factor

    def do_back_solve(self, rhs):
        """
        Performs a back solve with the factorized matrix. Should only be called after
        do_numeric_factorization.

        Parameters
        ----------
        rhs: numpy.ndarray or BlockVector

        Returns
        -------
        result: numpy.ndarray or BlockVector
        """
        if isinstance(rhs, BlockVector):
            _rhs = rhs.flatten()
            result = _rhs
        else:
            result = rhs.copy()

        result = self._ma27.do_backsolve(result)

        if isinstance(rhs, BlockVector):
            _result = rhs.copy_structure()
            _result.copyfrom(result)
            result = _result

        return result

    def get_inertia(self):
        """
        Get the inertia. Should only be called after do_numeric_factorization.

        Returns
        -------
        num_pos: int
            The number of positive eigenvalues of A
        num_neg: int
            The number of negative eigenvalues of A
        num_zero: int
            The number of zero eigenvalues of A
        """
        if self._num_status is None:
            raise RuntimeError('Must call do_numeric_factorization before inertia can be computed')
        if self._num_status != LinearSolverStatus.successful:
            raise RuntimeError('Can only compute inertia if the numeric factorization was successful.')
        num_negative_eigenvalues = self.get_info(15)
        num_positive_eigenvalues = self._dim - num_negative_eigenvalues
        return (num_positive_eigenvalues, num_negative_eigenvalues, 0)

    def set_icntl(self, key, value):
        """
        Set the value for an icntl option.

        Parameters
        ----------
        key: int
        value: int
        """
        self._ma27.set_icntl(key, value)

    def set_cntl(self, key, value):
        """
        Set the value for a cntl option.

        Parameters
        ----------
        key: int
        value: float
        """
        self._ma27.set_cntl(key, value)

    def get_icntl(self, key):
        """
        Get the value for an icntl option.

        Parameters
        ----------
        key: int

        Returns
        -------
        val: int
        """
        return self._ma27.get_icntl(key)

    def get_cntl(self, key):
        """
        Get the value for a cntl option.

        Parameters
        ----------
        key: int

        Returns
        -------
        val: float
        """
        return self._ma27.get_cntl(key)

    def get_info(self, key):
        return self._ma27.get_info(key)
