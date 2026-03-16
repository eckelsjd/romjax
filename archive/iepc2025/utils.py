"""Module for assorted processing utilities."""
from collections import defaultdict
from typing import Literal

import numpy as np
from scipy.interpolate import griddata
from numpy.typing import ArrayLike


def get_boundary(polys: ArrayLike):
    """Extract only the exterior edges and cells of a poly collection of quadrilaterals.
    
    :param polys: list of length N_cells, each poly is a list of 4 vertex indices
    :returns: boundary_edges - list of edges, one edge is a tuple of (p1, p2) vertex indices
              boundary_cells - list of cell ids corresponding to the boundary edges
    """
    edge_count = defaultdict(int)
    edge_to_cells = defaultdict(list)

    for cell_idx, poly in enumerate(polys):
        # Each poly is a list of 4 vertex indices
        num_vertices = len(poly)
        for i in range(num_vertices):
            p1, p2 = sorted((int(poly[i]), int(poly[(i+1)%num_vertices])))
            edge = (p1, p2)
            edge_count[edge] += 1
            edge_to_cells[edge].append(cell_idx)

    # Keep only edges that appear exactly once (boundary)
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    boundary_cells = [edge_to_cells[edge][0] for edge in boundary_edges]
    
    return boundary_edges, boundary_cells


def edge_normal(p1, p2, center=None):
    """Compute the normal vector of an edge defined by two points. Direction is determined by 
    a line connecting cell center to midpoint of the edge.
    
    :param p1: the first point (x1, y1)
    :param p2: the second point (x2, y2)
    :param center: the cell center (xc, yc) that determines correct normal direction,
                   no effect if not provided.
    """
    p1, p2 = np.array(p1), np.array(p2)
    tangent = p2 - p1
    normal = np.array([-tangent[1], tangent[0]])
    normal /= np.linalg.norm(normal)

    if center is not None:
        center = np.array(center)
        mid = (p1 + p2) / 2
        to_edge = mid - center
        to_edge /= np.linalg.norm(to_edge)

        # Flip if antiparallel
        if np.dot(normal, to_edge) < 0:
            normal = -normal

    return normal


def slice1d(centers: ArrayLike, 
            values: ArrayLike, 
            direction: Literal['horizontal', 'vertical'], 
            loc: float = 0.0,
            lims: tuple[float, float] = (None, None),
            num_points: int = 100,
            method: str = 'linear'
            ):
    """Extracts 1d interpolated slice from unstructured 2d data.
    
    :param centers: (N, 2) array of [z, r] coordinates
    :param values: (N,) array of field values at the provided coordinates
    :param direction: 'horizontal' (r=const) or 'vertical' (x=const)
    :param loc: the z or r coordinate of the slice line
    :param lims: the z or r limits, defaults to min and max of coordinates if either are None
    :param num_points: the number of interpolation points along slice
    :param method: interpolation method: 'linear', 'nearest', 'cubic'
    
    :returns coords, values: (num_points, 2), (num_points,) arrays of interpolation point 
                             coordinates and interpolated field values
    """
    axis = ['horizontal', 'vertical'].index(direction)
    amin, amax = np.min(centers[:, axis]), np.max(centers[:, axis])

    if lims is None:
        lims = (None, None)
    if lims[0] is None:
        lims = (amin, lims[1])
    if lims[1] is None:
        lims = (lims[0], amax)
    
    lims = (max(amin, lims[0]), min(amax, lims[1]))

    x0 = np.linspace(lims[0], lims[1], num_points)
    x1 = np.ones(num_points)*loc

    if direction == 'horizontal':
        coords = np.column_stack((x0, x1))
    elif direction == 'vertical':
        coords = np.column_stack((x1, x0))
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")
    
    slice_values = griddata(centers, values, coords, method=method)

    return coords, slice_values


def integrate_boundary(data: list[dict],
                       cells: ArrayLike,
                       vertices: ArrayLike,
                       connectivity: ArrayLike,
                       integrand: callable,
                       group_boundary: callable = None,
                       integrand_opts: dict = None,
                       boundary: str = 'all',
                       verbose: bool = False,
                       axisymmetric: bool = True
                       ):
    """Compute surface integral over domain boundary.
    
    :param data: List of frames of simulation data to plot. Each frame is a dict
                 with field variable names mapped to arrays of cell-center data. Only cell-center data is supported.
    :param cells: (N, 2) array of cell-center coordinates corresponding to the arrays in `data`. Order is (z, r)
    :param vertices: (M, 2) array of vertex coordinates. Order for coordinates is (z, r)
    :param connectivity: (N, 4) array specifying vertex indices for each cell
    :param integrand: Callable with edge/cell coordinates and field data
    :param group_boundary: Sorts edges by index into groups (from which boundary is selected), defaults to selecting all
    :param integrand_opts: extra kwargs to pass to the integrand function
    :param boundary: which boundaries to integrate over (defaults to all)
    :param verbose: include extra print progress statements
    :param axisymmetric: how to compute differential area (only axisymmetric about x-axis supported currently)

    :returns: the value of the surface integral at each requested time step at the selected boundaries
    """
    if integrand_opts is None:
        integrand_opts = {}

    boundary_edges, boundary_cells = get_boundary(connectivity)
    edge_indices = group_boundary(boundary_edges, vertices) if group_boundary is not None else {}  # dict of indices specifying which boundary edges belong to
    b_edges = boundary_edges if boundary == 'all' or len(edge_indices) == 0 else [boundary_edges[i] for i in edge_indices[boundary]]
    b_cells = boundary_cells if boundary == 'all' or len(edge_indices) == 0 else [boundary_cells[i] for i in edge_indices[boundary]]

    def _progress(i, n):
        if i == 0 or np.mod(i, int(0.25 * n)) == 0 or i == n - 1:
            print(f'Integrating boundary for frame {i+1}/{n}...')

    values = []
    for idx, dataframe in enumerate(data):
        if verbose:
            _progress(idx, len(data))

        res = []
        for edge, cell_idx in zip(b_edges, b_cells):
            pN, pC = vertices[edge, :], cells[cell_idx, :]
            cell_fields = {v: dataframe[v][cell_idx] for v in dataframe}           # (1,) for cell center

            if axisymmetric:
                dA = np.pi * np.sqrt((np.diff(pN[:, 0]))**2 + (np.diff(pN[:, 1]))**2) * np.sum(pN[:, 1])  # pi*L*(r0+r1) slant height formula
            else:
                raise NotImplementedError("Only axisymmetric integration about x-axis is supported")

            res.append(integrand(pN, pC, cell_fields, **integrand_opts) * dA)
        
        values.append(np.sum(res, axis=0))
    
    values = np.array(values)

    if values.shape[-1] == 1:
        values = np.squeeze(values, axis=-1)
    
    return np.array(values)  # (Nframes, shape of integrand)


def normalize(data: ArrayLike, method: str = None, consts: tuple[float, float] = None):
    """Normalize data using a provided method.
    
    :param data: any array data to normalize
    :param method: choose a provided normalization method
    :param consts: if provided, use these to normalize the data rather than compute new norm consts
    :return norm_data: the normalized data
    :return const: the normalization constants (if any)
    """
    if consts is None:
        c1, c2 = None, None
    else:
        c1, c2 = consts

    match method:
        case 'log':
            return np.log1p(data), (c1, c2)
        case 'zscore':
            if consts is None:
                c1, c2 = np.mean(data), np.std(data)
            return (data - c1) / c2, (c1, c2)
        case 'minmax':
            if consts is None:
                c1, c2 = np.min(data), np.max(data)
            return (data - c1) / (c2 - c1), (c1, c2)
        case 'sqrt':
            return np.sqrt(data), (c1, c2)
        case 'log10':
            return np.log10(data + 1), (c1, c2)
        case 'log-log':
            return np.log1p(np.log1p(data)), (c1, c2)
        case 'log-z':
            log_data = np.log1p(data)
            if consts is None:
                c1, c2 = np.mean(log_data), np.std(log_data)
            return (log_data - c1) / c2, (c1, c2)
        case 'sqrt-log':
            return np.log1p(np.sqrt(data)), (c1, c2)
        case 'log-minmax':
            log_data = np.log1p(data)
            if consts is None:
                c1, c2 = np.min(log_data), np.max(log_data)
            return (log_data - c1) / (c2 - c1), (c1, c2)
        case 'symlog':
            linthresh = 1
            linscale = 1
            base = 10
            abs_data = np.abs(data)
            res = np.empty_like(data)
            log_region = abs_data > linthresh
            res[~log_region] = linscale * data[~log_region]
            res[log_region] = np.sign(data)[log_region] * (linthresh*linscale + np.log(abs_data[log_region] / linthresh) / np.log(base))
            return res, None
        case 'symlog-minmax':
            linthresh = 1
            linscale = 1
            base = 10
            abs_data = np.abs(data)
            res = np.empty_like(data)
            log_region = abs_data > linthresh
            res[~log_region] = linscale * data[~log_region]
            res[log_region] = np.sign(data)[log_region] * (linthresh*linscale + np.log(abs_data[log_region] / linthresh) / np.log(base))
            if consts is None:
                c1, c2 = np.min(res), np.max(res)
            return (res - c1) / (c2 - c1), (c1, c2)
        case 'none' | None:
            return data, None
        case _:
            raise NotImplementedError(f"Normalization method '{method}' not recognized")


def denormalize(data: ArrayLike, method: str = None, consts: tuple[float, float] = None):
    """Denormalize data using the inverse of the specified method.
    
    :param data: the normalized data
    :param method: the original normalization method (will do the inverse here)
    :param consts: the norm constants returned by normalize for the associated method
    :return denorm_data: the denormalized data
    """
    match method:
        case 'log':
            return np.expm1(data)
        case 'zscore':
            c1, c2 = consts
            return data * c2 + c1
        case 'minmax':
            c1, c2 = consts
            return data * (c2 - c1) + c1
        case 'sqrt':
            return data ** 2
        case 'log10':
            return 10 ** data - 1
        case 'log-log':
            return np.expm1(np.expm1(data))
        case 'log-z':
            c1, c2 = consts
            return np.expm1(data * c2 + c1)
        case 'sqrt-log':
            return np.expm1(data) ** 2
        case 'log-minmax':
            c1, c2 = consts
            return np.expm1(data * (c2 - c1) + c1)
        case 'symlog':
            linthresh = 1
            linscale = 1
            base = 10
            res = np.empty_like(data)
            abs_data = np.abs(data)
            linear_region = abs_data <= linthresh * linscale
            res[linear_region] = data[linear_region] / linscale
            res[~linear_region] = np.sign(data)[~linear_region] * (linthresh * np.power(base, abs_data[~linear_region] - linthresh*linscale))
            return res
        case 'symlog-minmax':
            c1, c2 = consts
            linthresh = 1
            linscale = 1
            base = 10
            res = np.empty_like(data)
            data = data * (c2 - c1) + c1
            abs_data = np.abs(data)
            linear_region = abs_data <= linthresh * linscale
            res[linear_region] = data[linear_region] / linscale
            res[~linear_region] = np.sign(data)[~linear_region] * (linthresh * np.power(base, abs_data[~linear_region] - linthresh*linscale))
            return res
        case 'none' | None:
            return data
        case _:
            raise NotImplementedError(f"Normalization method '{method}' not recognized")


def relative_error(pred, targ, axis=None, skip_nan=False, pointwise=False, tol=1e-6):
    """Compute the relative L2 error between two vectors along the given axis.

    :param pred: the predicted values
    :param targ: the target values
    :param axis: the axis along which to compute the error
    :param skip_nan: whether to skip NaN values in the error calculation
    :param pointwise: if true, then don't sum and just return pointwise relative error
    :param tol: the tolerance in targ below which errors are set to nan (since rel error is unstable near 0)
    :returns: the relative L2 error
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        if pointwise:
            sum_func = lambda x, **kwargs: x
        else:
            sum_func = np.nansum if skip_nan else np.sum
        denom = np.sqrt(sum_func(targ**2, axis=axis))
        err = np.sqrt(sum_func((pred - targ)**2, axis=axis)) / denom

        if pointwise:
            err[denom < tol] = np.nan
    return np.nan_to_num(err, nan=np.nan, posinf=np.nan, neginf=np.nan)
