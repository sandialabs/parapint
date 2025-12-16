from .base_linear_solver_interface import LinearSolverInterface
from .results import LinearSolverStatus, LinearSolverResults
from .mkl_pardiso import MKLPardisoInterface
import scipy.sparse as sps
from scipy.sparse import isspmatrix_coo, isspmatrix_csr, tril, triu, spmatrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from pyomo.common.timing import HierarchicalTimer
from typing import Union, Tuple, Optional
import numpy as np


class InteriorPointMKLPardisoInterface(LinearSolverInterface):
    def __init__(self, iparm=None, msglvl: int = 0):
        self._pardiso = MKLPardisoInterface()
        if iparm is not None:
            for k, v in iparm.items():
                self.set_iparm(k, v)
        self._dim = None
        self._num_status = None
        self._pardiso.set_msglvl(msglvl)

    @classmethod
    def getLoggerName(cls):
        return 'mkl_pardiso'
    
    def _convert_matrix(self, matrix: Union[spmatrix, BlockMatrix]
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(matrix, BlockMatrix):
            smat = triu(matrix.tocsr())
        else:
            smat = triu(matrix)
        # This could be optimized
        zero_diag_idx = np.where(smat.diagonal() == 0)[0]
        diag_perturb = np.zeros(smat.shape[0])
        diag_perturb[zero_diag_idx] = 1
        smat2 = (smat + sps.diags(diag_perturb, format='csr')).sorted_indices()
        a2, ia2, ja2 = smat2.data, smat2.indptr, smat2.indices
        a2[ia2[zero_diag_idx]] = 0
        return a2, ia2 + 1, ja2 + 1

    def do_symbolic_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        self._num_status = None
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError("Matrix must be square")
        self._dim = nrows

        _a, _ia, _ja = self._convert_matrix(matrix)

        stat = self._pardiso.do_symbolic_factorization(a=_a, ia=_ia, ja=_ja)
        res = LinearSolverResults()
        if stat == 0:
            res.status = LinearSolverStatus.successful
        else:
            if raise_on_error:
                raise RuntimeError(
                    "Symbolic factorization was not successful; return code: "
                    + str(stat)
                )
            if stat in {-2}:
                res.status = LinearSolverStatus.not_enough_memory
            elif stat in {-7}:
                res.status = LinearSolverStatus.singular
            else:
                res.status = LinearSolverStatus.error
        return res
    


    def do_numeric_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError("Matrix must be square")
        if nrows != self._dim:
            raise ValueError(
                "Matrix dimensions do not match the dimensions of "
                "the matrix used for symbolic factorization"
            )
        _a, _ia, _ja = self._convert_matrix(matrix)
        stat = self._pardiso.do_numeric_factorization(a=_a, ia=_ia, ja=_ja)
        res = LinearSolverResults()
        if stat == 0:
            res.status = LinearSolverStatus.successful
        else:
            if raise_on_error:
                raise RuntimeError(
                    "Numeric factorization was not successful; return code: "
                    + str(stat)
                )
            if stat in {-2}:
                res.status = LinearSolverStatus.not_enough_memory
            elif stat in {-7}:
                res.status = LinearSolverStatus.singular
            else:
                res.status = LinearSolverStatus.error

        self._num_status = res.status

        return res
    

    
    def increase_memory_allocation(self, factor):
        raise NotImplementedError("increase_memory_allocation not implemented for MKL Pardiso")

    def do_back_solve(
        self, rhs: Union[np.ndarray, BlockVector], raise_on_error: bool = True
    ) -> Union[np.ndarray, BlockVector]:
        if self._num_status is None:
            raise RuntimeError('Must call do_numeric_factorization before do_back_solve can be called')
        if self._num_status != LinearSolverStatus.successful:
            raise RuntimeError('Can only call do_back_solve if the numeric factorization was successful.')
        
        if isinstance(rhs, BlockVector):
            _rhs = rhs.flatten()
            result = _rhs
        else:
            result = rhs.copy()

        result = self._pardiso.do_backsolve(result, copy=False)

        if isinstance(rhs, BlockVector):
            _result = rhs.copy_structure()
            _result.copyfrom(result)
            result = _result

        return result

    def set_iparm(self, key, value):
        self._pardiso.set_iparm(key, value)

    def get_iparm(self, key):
        return self._pardiso.get_iparm(key)

    def get_inertia(self):
        if self._num_status is None:
            raise RuntimeError('Must call do_numeric_factorization before inertia can be computed')
        if self._num_status != LinearSolverStatus.successful:
            raise RuntimeError('Can only compute inertia if the numeric factorization was successful.')
        num_negative_eigenvalues = self.get_iparm(23)
        num_positive_eigenvalues = self.get_iparm(22)
        return (num_positive_eigenvalues, num_negative_eigenvalues, 0)
    


class InteriorPointMKLPardisoSchurInterface(InteriorPointMKLPardisoInterface):

    def get_schur_complement(self):
        return self._pardiso.get_schur_complement()
    
    def do_symbolic_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], dim_schur: int, raise_on_error: bool = True
    ) -> LinearSolverResults:
        self._num_status = None
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError("Matrix must be square")
        self._dim = nrows

        _a, _ia, _ja = self._convert_matrix(matrix)

        stat = self._pardiso.do_symbolic_factorization_schur(a=_a, ia=_ia, ja=_ja, dim_schur=dim_schur)
        
        res = LinearSolverResults()
        if stat == 0:
            res.status = LinearSolverStatus.successful
        else:
            if raise_on_error:
                raise RuntimeError(
                    "Symbolic factorization was not successful; return code: "
                    + str(stat)
                )
            if stat in {-2}:
                res.status = LinearSolverStatus.not_enough_memory
            elif stat in {-7}:
                res.status = LinearSolverStatus.singular
            else:
                res.status = LinearSolverStatus.error

        return res
    
    def do_numeric_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError("Matrix must be square")
        if nrows != self._dim:
            raise ValueError(
                "Matrix dimensions do not match the dimensions of "
                "the matrix used for symbolic factorization"
            )
        _a, _ia, _ja = self._convert_matrix(matrix)
        stat = self._pardiso.do_numeric_factorization_schur(a=_a, ia=_ia, ja=_ja)
        res = LinearSolverResults()
        if stat == 0:
            res.status = LinearSolverStatus.successful
        else:
            if raise_on_error:
                raise RuntimeError(
                    "Numeric factorization was not successful; return code: "
                    + str(stat)
                )
            if stat in {-2}:
                res.status = LinearSolverStatus.not_enough_memory
            elif stat in {-7}:
                res.status = LinearSolverStatus.singular
            else:
                res.status = LinearSolverStatus.error

        self._num_status = res.status

        return res

    def do_back_solve(
        self, rhs: Union[np.ndarray, BlockVector],
        ) -> Union[np.ndarray, BlockVector]:
        if self._num_status is None:
            raise RuntimeError('Must call do_numeric_factorization before do_back_solve can be called')
        if self._num_status != LinearSolverStatus.successful:
            raise RuntimeError('Can only call do_back_solve if the numeric factorization was successful.')
        
        if isinstance(rhs, BlockVector):
            _rhs = rhs.flatten()
            result = _rhs
        else:
            result = rhs.copy()

        result = self._pardiso.do_backsolve_schur(result, copy=False)

        if isinstance(rhs, BlockVector):
            _result = rhs.copy_structure()
            _result.copyfrom(result)
            result = _result

        return result
