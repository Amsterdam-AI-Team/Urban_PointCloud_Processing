# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

# Adapted from https://photutils.readthedocs.io/en/stable/_modules/photutils/
#              utils/interpolation.html#ShepardIDWInterpolator
#
# Licensed under a 3-clause BSD style license
#
# Copyright (c) 2011-2020, Photutils developers
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
#     Neither the name of the Photutils Team nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IM-
# PLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This module provides tools for interpolating data.
"""

import numpy as np


class SpatialInterpolator:
    """
    Class to perform either Inverse Distance Weighted (IDW) interpolation,
    or Maximum-based interpolation.

    The IDW interpolator uses a modified version of `Shepard's method
    <https://en.wikipedia.org/wiki/Inverse_distance_weighting>`_ (see
    the Notes section for details).

    The Maximum-based interpolator does not technically "interpolate",
    but simply returns the maximum value among the neighbouring points.

    Parameters
    ----------
    coordinates : float, 1D array-like, or NxM-array-like
        Coordinates of the known data points. In general, it is expected
        that these coordinates are in a form of a NxM-like array where N
        is the number of points and M is dimension of the coordinate
        space. When M=1 (1D space), then the ``coordinates`` parameter
        may be entered as a 1D array or, if only one data point is
        available, ``coordinates`` can be a scalar number representing
        the 1D coordinate of the data point.

        .. note::
            If the dimensionality of ``coordinates`` is larger than 2,
            e.g., if it is of the form N1 x N2 x N3 x ... x Nn x M, then
            it will be flattened to form an array of size NxM where N =
            N1 * N2 * ... * Nn.

    values : float or 1D array-like
        Values of the data points corresponding to each coordinate
        provided in ``coordinates``. In general a 1D array is expected.
        When a single data point is available, then ``values`` can be a
        scalar number.

        .. note::
            If the dimensionality of ``values`` is larger than 1 then it
            will be flattened.

    weights : float or 1D array-like, optional
        Weights to be associated with each data value. These weights, if
        provided, will be combined with inverse distance weights (see
        the Notes section for details). When ``weights`` is `None`
        (default), then only inverse distance weights will be used. When
        provided, this input parameter must have the same form as
        ``values``.

    method : str, optional (ADDED PARAMETER)
        Which interpolation method to use. Options are 'idw' for Inverse
        Distance Weighting (default), or 'max' for Maximum-based
        interpolation.

    leafsize : float, optional
        The number of points at which the k-d tree algorithm switches
        over to brute-force. ``leafsize`` must be positive.  See
        `scipy.spatial.cKDTree` for further information.

    Notes
    -----
    The IDW interpolator uses a slightly modified version of `Shepard's
    method <https://en.wikipedia.org/wiki/Inverse_distance_weighting>`_.
    The essential difference is the introduction of a "regularization"
    parameter (``reg``) that is used when computing the inverse distance
    weights:

    .. math::
        w_i = 1 / (d(x, x_i)^{power} + r)

    By supplying a positive regularization parameter one can avoid
    singularities at the locations of the data points as well as control
    the "smoothness" of the interpolation (e.g., make the weights of the
    neighbors less varied). The "smoothness" of interpolation can also
    be controlled by the power parameter (``power``).
    """

    def __init__(self, coordinates, values,
                 weights=None, leafsize=10, method='idw'):
        from scipy.spatial import cKDTree

        coordinates = np.atleast_2d(coordinates)
        if coordinates.shape[0] == 1:
            coordinates = np.transpose(coordinates)
        if coordinates.ndim != 2:
            coordinates = np.reshape(coordinates, (-1, coordinates.shape[-1]))

        values = np.asanyarray(values).ravel()

        ncoords = coordinates.shape[0]
        if ncoords < 1:
            raise ValueError('You must enter at least one data point.')

        if values.shape[0] != ncoords:
            raise ValueError('The number of values must match the number '
                             'of coordinates.')

        if weights is not None:
            weights = np.asanyarray(weights).ravel()
            if weights.shape[0] != ncoords:
                raise ValueError('The number of weights must match the '
                                 'number of coordinates.')
            if np.any(weights < 0.0):
                raise ValueError('All weight values must be non-negative '
                                 'numbers.')

        self.coordinates = coordinates
        self.ncoords = ncoords
        self.coords_ndim = coordinates.shape[1]
        self.values = values
        self.weights = weights
        self.method = method
        self.kdtree = cKDTree(coordinates, leafsize=leafsize)

    def __call__(self, positions, n_neighbors=8, max_dist=np.inf, eps=0.0,
                 power=1.0, reg=0.0, conf_dist=1e-12, fill_value=np.nan,
                 dtype=float, workers=1):
        """
        Evaluate the interpolator at the given positions.

        Parameters
        ----------
        positions : float, 1D array-like, or NxM-array-like
            Coordinates of the position(s) at which the interpolator
            should be evaluated. In general, it is expected that these
            coordinates are in a form of a NxM-like array where N is the
            number of points and M is dimension of the coordinate space.
            When M=1 (1D space), then the ``positions`` parameter may be
            input as a 1D-like array or, if only one data point is
            available, ``positions`` can be a scalar number representing
            the 1D coordinate of the data point.

            .. note::
                If the dimensionality of the ``positions`` argument is
                larger than 2, e.g., if it is of the form N1 x N2 x N3 x
                ... x Nn x M, then it will be flattened to form an array
                of size NxM where N = N1 * N2 * ... * Nn.

            .. warning::
                The dimensionality of ``positions`` must match the
                dimensionality of the ``coordinates`` used during the
                initialization of the interpolator.

        n_neighbors : int, optional
            The maximum number of nearest neighbors to use during the
            interpolation.

        max_dist : float, optional (ADDED PARAMETER)
            The maximum radius within which neighbours are considered
            during the interpolation.

        eps : float, optional
            Set to use approximate nearest neighbors; the kth neighbor
            is guaranteed to be no further than (1 + ``eps``) times the
            distance to the real *k*-th nearest neighbor. See
            `scipy.spatial.cKDTree.query` for further information.

        power : float, optional
            The power of the inverse distance used for the interpolation
            weights.  See the Notes section for more details.

        reg : float, optional
            The regularization parameter. It may be used to control the
            smoothness of the interpolator. See the Notes section for
            more details.

        conf_dist : float, optional
            The confusion distance below which the interpolator should
            use the value of the closest data point instead of
            attempting to interpolate. This is used to avoid
            singularities at the known data points, especially if
            ``reg`` is 0.0.

        fill_value : float, optional. (ADDED PARAMETER)
            Value to use when no neighbours meet the required max_dist.
            Defaults to np.nan.

        dtype : data-type, optional
            The data type of the output interpolated values. If `None`
            then the type will be inferred from the type of the
            ``values`` parameter used during the initialization of the
            interpolator.

        workers : int, optional
            How many CPU threads to use when querying the KD-tree (default 1).
            This only has effect when len(positions) is large.
        """

        n_neighbors = int(n_neighbors)
        if n_neighbors < 1:
            raise ValueError('n_neighbors must be a positive integer')

        if conf_dist is not None and conf_dist <= 0.0:
            conf_dist = None

        positions = np.asanyarray(positions)
        if positions.ndim == 0:
            # assume we have a single 1D coordinate
            if self.coords_ndim != 1:
                raise ValueError('The dimensionality of the input position '
                                 'does not match the dimensionality of the '
                                 'coordinates used to initialize the '
                                 'interpolator.')
        elif positions.ndim == 1:
            # assume we have a single point
            if (self.coords_ndim != 1 and
                    (positions.shape[-1] != self.coords_ndim)):
                raise ValueError('The input position was provided as a 1D '
                                 'array, but its length does not match the '
                                 'dimensionality of the coordinates used '
                                 'to initialize the interpolator.')
        elif positions.ndim != 2:
            raise ValueError('The input positions must be an array-like '
                             'object of dimensionality no larger than 2.')

        positions = np.reshape(positions, (-1, self.coords_ndim))
        npositions = positions.shape[0]

        distances, idx = self.kdtree.query(positions, k=n_neighbors,
                                           distance_upper_bound=max_dist, p=2,
                                           eps=eps, workers=workers)

        if dtype is None:
            dtype = self.values.dtype

        interp_values = np.empty(npositions, dtype=dtype)
        interp_values.fill(fill_value)

        if n_neighbors == 1:
            valid_idx = np.isfinite(distances)
            interp_values[valid_idx] = self.values[valid_idx]
        else:
            # TODO: can this loop be optimised / parallelised (should be
            # possible, computations are independent.)
            for k in range(npositions):
                valid_idx = np.isfinite(distances[k])
                idk = idx[k][valid_idx]
                dk = distances[k][valid_idx]

                if dk.shape[0] == 0:
                    interp_values[k] = fill_value
                    continue

                if self.method == 'idw':
                    if conf_dist is not None:
                        # check if we are close to a known data point
                        confused = (dk <= conf_dist)
                        if np.any(confused):
                            interp_values[k] = self.values[idk[confused][0]]
                            continue

                    w = 1.0 / ((dk ** power) + reg)
                    if self.weights is not None:
                        w *= self.weights[idk]

                    wtot = np.sum(w)
                    if wtot > 0.0:
                        interp_values[k] = np.dot(w, self.values[idk]) / wtot
                    else:
                        interp_values[k] = fill_value
                elif self.method == 'max':
                    interp_values[k] = np.max(self.values[idk])

        if len(interp_values) == 1:
            return interp_values[0]
        else:
            return interp_values


class FastGridInterpolator:
    """
    Class to perform fast interpolation using gridded data. The interpolator
    simply returns the values of the grid cells in which the queried points
    fall. Grid coordinates are assumed to be the centroids of each grid cell.

    Parameters
    ----------
    grid_x : list or array-like
        The x-coordinates of the gridded data (ascending).

    grid_y : list or array-like
        The y-coordinates of the gridded data (decsending).

    values : array of shape (Ny, Nx)
        The values of the gridded data.
    """

    def __init__(self, grid_x, grid_y, values):
        step_x = grid_x[1] - grid_x[0]
        step_y = grid_y[0] - grid_y[1]
        self.bin_x = grid_x - (step_x/2)
        self.bin_y = grid_y + (step_y/2)
        self.values = values

    def __call__(self, positions):
        """
        Evaluate the interpolator at the given positions.

        Parameters
        ----------
        positions : array of shape (Np, 2)
            Array of points to query. The first column contains the x-values,
            the second column contains the y-values.
        """
        x_idx = np.digitize(positions[:, 0], self.bin_x) - 1
        y_idx = np.digitize(positions[:, 1], self.bin_y, right=True) - 1
        return self.values[y_idx, x_idx]
