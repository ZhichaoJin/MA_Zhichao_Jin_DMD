import logging
from collections import deque
from typing import Tuple
import scipy.sparse as ss
import numpy as np
import functions_dmd as dfc

logger = logging.getLogger(__name__)


class WindowDMD:
    """WindowDMD is a class that implements window dynamic mode decomposition
    The time complexity (multiply–add operation for one iteration) is O(8n^2),
    and space complexity is O(2wn+2n^2), where n is the state dimension, w is
    the window size.

    Algorithm description:
        At time step t, define two matrix X(t) = [x(t-w+1),x(t-w+2),...,x(t)],
        Y(t) = [y(t-w+1),y(t-w+2),...,y(t)], that contain the recent w snapshot
        pairs from a finite time window, where x(t), y(t) are the n dimensional
        state vector, y(t) = f(x(t)) is the image of x(t), f() is the dynamics.

        Here, if the (discrete-time) dynamics are given by z(t) = f(z(t-1)),
        then x(t), y(t) should be measurements correponding to consecutive
        states z(t-1) and z(t).

        At time t+1, we need to forget the old snapshot pair xold = x(t-w+1),
        yold = y(t-w+1), and remember the new snapshot pair x = x(t+1),
        y = y(t+1).

        We would like to update the DMD matrix A(t)
        by efficient rank-2 updating window DMD algrithm.
        An exponential weighting factor can be used to place more weight on
        recent data.

    Usage:
        wdmd = WindowDMD(n, w)
        wdmd.initialize(Xw, Yw) # this is necessary for window DMD
        wdmd.update(x, y)
        evals, modes = wdmd.computemodes()

    properties:
        n: state dimension
        w: window size, we must have w >= 2*n
        weighting: weighting factor in (0,1]
        timestep: number of snapshot pairs processed (i.e., current time step)
        Xw: recent w snapshots x stored in Xw, size n by w
        Yw: recent w snapshots y stored in Yw, size n by w
        A: DMD matrix, size n by n
        P: Matrix that contains information about recent w snapshots, size n by n

    methods:
        initialize(Xw, Yw), initialize window DMD algorithm with w snapshot pairs, this is necessary
        update(x, y), update DMD computation by adding a new snapshot pair
        computemodes(), compute and return DMD eigenvalues and DMD modes

    Authors:
        Hao Zhang
        Clarence W. Rowley

    References:
        Zhang, Hao, Clarence W. Rowley, Eric A. Deem, and Louis N. Cattafesta.
        "Online dynamic mode decomposition for time-varying systems."
        SIAM Journal on Applied Dynamical Systems 18, no. 3 (2019): 1586-1609.

    Date created: April 2017
    """

    def __init__(self, n: int, w: int) -> None:
        """
        Creat an object for window DMD
        Usage: wdmd = WindowDMD(n, w, weighting), we must have w >= 2*n
        """
        # input check
        assert n >= 1 and isinstance(n, int)
        assert w >= 1 and isinstance(w, int)
        

        self.n = n
        self.w = w
        self.timestep = 0
        self.Xw = deque()
        self.Yw = deque()
        self.A =deque()
        self.P =deque()
        self.r = 0 #Threshold
        self._svd_rank = 0
        # need to call initialize before update() and computemodes()
        self.ready = False

    def initialize(self, Xw: np.ndarray, Yw: np.ndarray) -> None:
        """Initialize window DMD with first w snapshot pairs stored in (Xw, Yw)
        Usage: wdmd.initialize(Xw, Yw)

        Args:
            Xw (np.ndarray): 2D array, shape (n, w), matrix [x(1),x(2),...x(w)]
            Yw (np.ndarray): 2D array, shape (n, w), matrix [y(1),y(2),...y(w)]
        """
        # input check
        assert Xw is not None and Yw is not None
        Xw, Yw = np.array(Xw), np.array(Yw)
        assert Xw.shape == Yw.shape
        

        # initialize Xw, Yw queue
        for i in range(self.w):
            self.Xw.append(Xw[:, i])
            self.Yw.append(Yw[:, i])
        



        # # initialie A
        self.A, self.r = self._compute_operator(Xw, Yw)

        # timestep
        self.timestep += self.w

        # mark the model as ready
        self.ready = True

    def _compute_operator(self, X, Y):
        """
        Compute the low-rank operator.

        :param numpy.ndarray X: matrix containing the snapshots x0,..x{n-1} by
            column.
        :param numpy.ndarray Y: matrix containing the snapshots x1,..x{n} by
            column.
        :return: the (truncated) left-singular vectors matrix, the (truncated)
            singular values array, the (truncated) right-singular vectors
            matrix of X.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """

        U, s, V, rank= self._compute_svd(X)
        print(U.shape,s.shape,V.shape)
        atilde = self._least_square_operator(U, s, V, Y)



        return atilde, rank

    def _least_square_operator(self, U, s, V, Y):
        """
        Private method that computes the lowrank operator from the singular
        value decomposition of matrix X and the matrix Y.

        .. math::

            \\mathbf{\\tilde{A}} =
            \\mathbf{U}^* \\mathbf{Y} \\mathbf{X}^\\dagger \\mathbf{U} =
            \\mathbf{U}^* \\mathbf{Y} \\mathbf{V} \\mathbf{S}^{-1}

        :param numpy.ndarray U: 2D matrix that contains the left-singular
            vectors of X, stored by column.
        :param numpy.ndarray s: 1D array that contains the singular values of X.
        :param numpy.ndarray V: 2D matrix that contains the right-singular
            vectors of X, stored by row.
        :param numpy.ndarray Y: input matrix Y.
        :return: the lowrank operator
        :rtype: numpy.ndarray
        """
        S_diag = np.diag(s)
        Atilde = np.dot(np.dot(np.dot(np.transpose(U),Y),V),np.linalg.inv(S_diag))
        return Atilde

    def _compute_svd(self, X, svd_rank=None):
        """
        Truncated Singular Value Decomposition.

        :param numpy.ndarray X: the matrix to decompose.
        :param svd_rank: the rank for the truncation; If 0, the method computes
            the optimal rank and uses it for truncation; if positive interger,
            the method uses the argument for the truncation; if float between 0
            and 1, the rank is the number of the biggest singular values that
            are needed to reach the 'energy' specified by `svd_rank`; if -1,
            the method does not compute truncation. If None, self._svd_rank is
            used.
        :type svd_rank: int or float
        :return: the truncated left-singular vectors matrix, the truncated
            singular values array, the truncated right-singular vectors matrix.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray

        References:
        Gavish, Matan, and David L. Donoho, The optimal hard threshold for
        singular values is, IEEE Transactions on Information Theory 60.8
        (2014): 5040-5053.
        """
        U, s, V = np.linalg.svd(X, full_matrices=False)
        V = V.conj().T
        if svd_rank is None:
            svd_rank = self._svd_rank

        if svd_rank == 0:
            omega = lambda x: 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43
            beta = np.divide(*sorted(X.shape))
            tau = np.median(s) * omega(beta)
            rank = np.sum(s > tau)
        elif svd_rank > 0 and svd_rank < 1:
            cumulative_energy = np.cumsum(s**2 / (s**2).sum())
            rank = np.searchsorted(cumulative_energy, svd_rank) + 1
        elif svd_rank >= 1 and isinstance(svd_rank, int):
            rank = min(svd_rank, U.shape[1])
        else:
            rank = X.shape[1]

        U = U[:, :rank]
        V = V[:, :rank]
        s =s[:rank]

        return U, s, V, rank 

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Update the DMD computation by sliding the finite time window forward
        Forget the oldest pair of snapshots (xold, yold), and remembers the newest
        pair of snapshots (x, y) in the new time window. If the new finite
        time window at time step t+1 includes recent w snapshot pairs as
        X(t+1) = [x(t-w+2),x(t-w+3),...,x(t+1)], Y(t+1) = [y(t-w+2),y(t-w+3),
        ...,y(t+1)], where y(t) = f(x(t)) and f is the dynamics, then we should
        take x = x(t+1), y = y(t+1)
        Usage: wdmd.update(x, y)

        Args:
            x (np.ndarray): 1D array, shape (n, ), x(t) as in y(t) = f(t, x(t))
            y (np.ndarray): 1D array, shape (n, ), x(t) as in y(t) = f(t, x(t))

        Raises:
            Exception: if Not initialized yet! Need to call self.initialize(Xw, Yw)
        """
        if not self.ready:
            raise Exception(
                "Not initialized yet! Need to call self.initialize(Xw, Yw)")

        # input check
        assert x is not None and y is not None
        x, y = np.array(x), np.array(y)

        assert np.array(x).shape == np.array(y).shape
        assert np.array(x).shape[0] == self.n

        # define old snapshots to be discarded
        # define old snapshots to be discarded
        xold, yold = self.Xw.popleft(), self.Yw.popleft()
        # Update recent w snapshots
        self.Xw.append(x)
        self.Yw.append(y)
        
        # direct rank-2 update
        # define matrices
        
        
        # update A
        
        self.A,self.r= self._compute_operator(np.asmatrix(self.Xw), np.asmatrix(self.Yw))

        # time step + 1
        self.timestep += 1

    def computemodes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute and return DMD eigenvalues and DMD modes at current time step
        Usage: evals, modes = wdmd.computemodes()

        Raises:
            Exception: if Not initialized yet! Need to call self.initialize(Xw, Yw)

        Returns:
            Tuple[np.ndarray, np.ndarray]: DMD eigenvalues and DMD modes
        """
        if not self.ready:
            raise Exception(
                "Not initialized yet! Need to call self.initialize(Xw, Yw)")
        self._eigenvalues, self._eigenvectors = np.linalg.eig(self.A)
        return self._eigenvalues, self._eigenvectors
    @property
    def eigenvalues(self):
        return self._eigenvalues

    @property
    def eigenvectors(self):
        return self._eigenvectors

    @property
    def rank(self):
        return self.r
    
