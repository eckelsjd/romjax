"""Module to compute performance metrics from Hall2De mesh data.

Includes:
  - thruster_edges - group edges into (anode, outflow, thruster) for surface integration
  - get_channel - get thruster channel dimensions from mesh
  - thrust - helper to compute total thrust
  - discharge_current - helper to compute total discharge current
  - beam_current - helper to compute ion beam current
"""
import copy
from typing import Literal
from collections import defaultdict

import numpy as np
from numpy.typing import ArrayLike

from romtools.utils import integrate_boundary, get_boundary, edge_normal, slice1d

__all__ = ['thrust', 'discharge_current', 'beam_current', 'thruster_edges', 'get_channel']

AVOGADRO_CONSTANT = 6.02214076e23     # 1/mol
FUNDAMENTAL_CHARGE = 1.602176632e-19  # C
BOLTZMANN_CONSTANT = 1.380649e-23     # J/K
ELECTRON_MASS = 9.1093837e-31         # kg

MOLECULAR_WEIGHTS = {
    'Xe': 131.293,
    'Kr': 83.798
}


def _thrust_integrand(edge_pts: ArrayLike, 
                      cell_center: ArrayLike,
                      cell_data: dict, 
                      species: Literal['Xe', 'Kr'] = 'Xe',
                      **kwargs
                      ):
    """Compute thrust contribution at a given cell and boundary edge.

    Assumes constant value of cell center values at the edge midpoint (edge data not used).
    
    :param edge_pts: (Nvertex, Ncoords) the coordinates (z,r) of each vertex of the given edge, i.e. (2, 2)
    :param cell_center: (Ncoords,) the coordinates (z,r) of the associated cell center
    :param cell_data: Dictionary giving field variable values at the cell center
    :param species: the propellant species (defaults to Xe)

    :returns: the contribution of this cell to the thrust surface integral (a 2-vector for z,r directions)
    """
    p1, p2 = edge_pts[0, :], edge_pts[1, :]
    normal = edge_normal(p1, p2, center=cell_center)
    mi = MOLECULAR_WEIGHTS[species] / AVOGADRO_CONSTANT / 1000 # kg
    
    res = np.zeros(2)

    # Neutrals
    if 'un_z (m/s)' in cell_data:
        vel = np.array([cell_data['un_z (m/s)'], cell_data['un_r (m/s)']])
        density = cell_data['nn (m^-3)']
        res += mi * density * np.dot(vel, normal) * vel
    
    # Ions
    total_density = 0.0
    visited = set()
    for v in cell_data:
        if 'ui_' in v and '(m/s)' in v:
            substr = v.split('(')[1].split(')')[0].strip()  # gives (idx_charge, idx_fluid)
            if substr not in visited:
                visited.add(substr)
                Z = int(substr.split(',')[0].strip())
                vel = np.array([cell_data[f'ui_z({substr}) (m/s)'], cell_data[f'ui_r({substr}) (m/s)']])
                density = cell_data[f'ni({substr}) (m^-3)']
                total_density += density * Z

                res += mi * density * np.dot(vel, normal) * vel
    
    # Electrons
    if 'je_z (A/m^2)' in cell_data:
        vel = np.array([cell_data['je_z (A/m^2)'], cell_data['je_r (A/m^2)']]) / (density * FUNDAMENTAL_CHARGE)
        res += ELECTRON_MASS * density * np.dot(vel, normal) * vel
    
    return res


def _current_integrand(edge_pts: ArrayLike, 
                       cell_center: ArrayLike, 
                       cell_data: dict, 
                       which: Literal['both', 'electron', 'ion'] = 'both',
                       **kwargs
                       ):
    """Compute current contribution at a given cell and boundary edge.

    Assumes constant value of cell center values at the edge midpoint (edge data not used).
    
    :param edge_pts: (Nvertex, Ncoords) the coordinates (z,r) of each vertex of the given edge, i.e. (2, 2)
    :param cell_center: (Ncoords,) the coordinates (z,r) of the associated cell center
    :param cell_data: Dictionary giving field variable values at the cell center
    :param which: the current contributions to consider (electron, ion, or both)

    :returns: the contribution of this cell to the current surface integral
    """
    p1, p2 = edge_pts[0, :], edge_pts[1, :]
    normal = edge_normal(p1, p2, center=cell_center)

    res = 0.0

    # Ions
    if which in {'both', 'ion'}:
        visited = set()
        for v in cell_data:
            if 'ui_' in v and '(m/s)' in v:
                substr = v.split('(')[1].split(')')[0].strip()  # gives (idx_charge, idx_fluid)
                if substr not in visited:
                    visited.add(substr)
                    Z = int(substr.split(',')[0].strip())
                    vel = np.array([cell_data[f'ui_z({substr}) (m/s)'], cell_data[f'ui_r({substr}) (m/s)']])
                    density = cell_data[f'ni({substr}) (m^-3)']

                    res += np.dot(FUNDAMENTAL_CHARGE * Z * density * vel, normal)
    
    # Electrons
    if which in {'both', 'electron'}:
        if 'je_z (A/m^2)' in cell_data:
            res += np.dot(np.array([cell_data['je_z (A/m^2)'], cell_data['je_r (A/m^2)']]), normal)
    
    return float(res)


def thruster_edges(edges: list[tuple[int, int]],
                   vertices: ArrayLike, 
                   tol: float = 1e-8
                   ):
    """Group edges by outflow, anode, or thruster surfaces.

                    -----------------
                    |               |
                    |               |
                ----                |
        anode-->|                   |<-- Outflow
                |                   |
                ----                |
        thruster-^  |               |
                    |               |
                    -----------------
    
    Outflow surfaces:
      Identify the bounding domain box (aligned with x-y axes), and select any
      vertical or horizontal edge that lies on this box.

    Thruster surfaces:
      Opposite of outflow surfaces (i.e. the "notch" sticking out of the box).
    
    Anode surfaces:
      All vertical edges at the left-most end of the domain (also belongs to thruster surface group).
      For robustness, will take any vertical edges that are less than 1/2 the distance from the rear
      anode to the channel exit (i.e. 1/2 the channel length).
    
    :param edges: list of tuples giving indices of vertices for all boundary edges
    :param vertices: (Nvertices, 2) the z,r coordinate values for all vertices in the mesh
    :param tol: tolerance for floating point comparisons.
    
    :returns: a dict specifying the indices of edges belonging to each category of (anode, outflow, thruster)
    """
    vertices = np.round(vertices / tol) * tol

    # Get limits of domain by longest vertical/horizontal segments
    vert_lengths = defaultdict(float)
    horz_lengths = defaultdict(float)

    for v1, v2 in edges:
        x1, y1 = vertices[v1]
        x2, y2 = vertices[v2]

        if np.abs(x1 - x2) < tol:
            vert_lengths[x1] += np.abs(y2 - y1)
        elif np.abs(y1 - y2) < tol:
            horz_lengths[y1] += np.abs(x2 - x1)

    zlims = sorted(vert_lengths.items(), key=lambda x: x[1], reverse=True)[:2]
    rlims = sorted(horz_lengths.items(), key=lambda x: x[1], reverse=True)[:2]

    # Outflow "box" limits
    zmin, zmax = sorted([x for x, _ in zlims])
    rmin, rmax = sorted([x for x, _ in rlims])
    # print(f'z: {zmin}, {zmax}')
    # print(f'r: {rmin}, {rmax}')

    # Full domain limits
    zmin_domain = np.min(vertices[:, 0])

    body_edges = set()
    outflow_edges = set()
    anode_edges = set()

    body_min = np.inf  # minimum radius of thruster body edges

    for i, (v1, v2) in enumerate(edges):
        x1, y1 = vertices[v1, :]
        x2, y2 = vertices[v2, :]

        is_vert = np.abs(x1 - x2) < tol and (np.abs(x1 - zmin) < tol or np.abs(x1 - zmax) < tol)
        is_horiz = np.abs(y1 - y2) < tol and (np.abs(y1 - rmin) < tol or np.abs(y1 - rmax) < tol)

        if is_vert or is_horiz:
            outflow_edges.add(i)
        else:
            body_edges.add(i)

            if np.min((y1, y2)) < body_min:
                body_min = np.min((y1, y2))

            if np.abs(x1 - x2) < tol and x1 < 0.5 * (zmin - zmin_domain):
                anode_edges.add(i)
    
    # Classify vertical outflow boundary below channel as thruster body
    for i in outflow_edges.copy():
        v1, v2 = edges[i]
        x1, y1 = vertices[v1, :]
        x2, y2 = vertices[v2, :]

        if np.abs(x1 - x2) < tol and np.min((y1, y2)) < body_min and x1 < (zmin + 0.5 * (zmax-zmin)):
            outflow_edges.remove(i)
            body_edges.add(i)
    
    return {'anode': anode_edges, 'thruster': body_edges, 'outflow': outflow_edges}


def get_channel(edges: list[tuple[int, int]],
                vertices: ArrayLike,
                tol: float = 1e-8
                ):
    """Extract channel dimensions from domain boundary edge information.
    
    :param edges: list of tuples giving indices of all boundary vertices
    :param vertices: (Nvertices, 2) the z,r coordinate values for all vertices in the mesh
    :param tol: tolerance for floating point comparisons
    
    :returns: dict with channel_length, inner_radius, and outer_radius for the thruster channel
    """
    groups = thruster_edges(edges, vertices, tol=tol)
    body_edges = [edges[i] for i in groups['thruster']]

    vertices = np.round(vertices / tol) * tol

    # Get longest horizontal edges (inner and outer radii of channel)
    horz_lengths = defaultdict(float)
    zmin, zmax = np.inf, -np.inf

    for v1, v2 in body_edges:
        x1, y1 = vertices[v1]
        x2, y2 = vertices[v2]

        if np.max((x1, x2)) > zmax:
            zmax = np.max((x1, x2))
        
        if np.min((x1, x2)) < zmin:
            zmin = np.min((x1, x2))

        if np.abs(y1 - y2) < tol:
            horz_lengths[y1] += np.abs(x2 - x1)

    rlims = sorted(horz_lengths.items(), key=lambda x: x[1], reverse=True)[:2]
    r_inner, r_outer = sorted([x for x, _ in rlims])

    return {'channel_length': zmax - zmin, 'inner_radius': r_inner, 'outer_radius': r_outer}


def slice_centerline(data: list[dict],
                     cells: ArrayLike,
                     vertices: ArrayLike,
                     connectivity: ArrayLike,
                     zlim: tuple[float, float] | float = (None, None),
                     verbose: bool = False,
                     num_points: int = 100,
                     method: str = 'linear'
                    ):
    """Computes 1d centerline slice.
    
    :param data: List of frames of simulation data to plot. If several frames are provided, the result will be
                 animated. If a single frame is provided, the result will be a static plot. Each frame is a dict
                 with field variable names mapped to arrays of cell-center data. Only cell-center data is supported.
    :param cells: (N, 2) array of cell-center coordinates corresponding to the arrays in `data`. Order is (z, r)
    :param vertices: (M, 2) array of vertex coordinates. Order for coordinates is (z, r)
    :param connectivity: (N, 4) array specifying vertex indices for each cell
    :param zlim: the z-axis limits to plot along centerline (defaults to entire domain if None), specify a single
                 float to specify number of channel lengths (e.g. use zlim=1.0 for a single channel length)
    :param verbose: include extra print statements if true (default False)
    :param num_points: number of points for the slice
    :param method: the scipy griddata interpolation method to use

    :returns: slice_coords, slice_data: (num_points, 2), list of frames with (num_points,) arrays for each sliced variable
    """
    boundary_edges, _ = get_boundary(connectivity)
    channel_dims = get_channel(boundary_edges, vertices)
    center_radius = (channel_dims['inner_radius'] + channel_dims['outer_radius']) / 2

    def _progress(i, n):
        if i == 0 or np.mod(i, int(0.25 * n)) == 0 or i == n - 1:
            print(f'Slicing frame {i+1}/{n}...')

    if not isinstance(zlim, tuple | list):
        zlim = (None, channel_dims['channel_length'] * zlim)
    
    # Interpolate all vars to centerline
    slice_data, slice_coords = [], None
    
    if verbose:
        print("Slicing at channel centerline...")

    for i, frame in enumerate(data):
        if verbose:
            _progress(i, len(data))

        slice_frame = {}
        for v, arr in frame.items():
            slice_coords, vals = slice1d(cells, arr, 'horizontal', loc=center_radius, lims=zlim, num_points=num_points, method=method)
            slice_frame[v] = vals
        
        slice_data.append(slice_frame)
    
    return slice_coords, slice_data


def thrust(*integrate_args, **integrate_kwargs):
    """Helper function to compute thrust from field data. Integrates over outflow boundaries.
    
    See integrate_boundary for arguments.
    
    :returns: the vector thrust (z and r)
    """
    opts = copy.deepcopy(integrate_kwargs)
    opts['boundary'] = 'outflow'

    return integrate_boundary(*integrate_args, _thrust_integrand, group_boundary=thruster_edges, **opts)


def discharge_current(*integrate_args, return_abs: bool = True, **integrate_kwargs):
    """Helper to compute discharge current. Integrates all species currents over anode boundary.
    
    See integrate_boundary for arguments.
    
    :param: return_abs: whether to return the absolute value (for convention), default True
    :returns: the discharge current (A)
    """
    opts = copy.deepcopy(integrate_kwargs)
    opts['boundary'] = 'anode'
    opts['integrand_opts'] = opts.get('integrand_opts', {})
    opts['integrand_opts']['which'] = 'both'  # sum all currents at anode

    ret = integrate_boundary(*integrate_args, _current_integrand, group_boundary=thruster_edges, **opts)

    return np.abs(ret) if return_abs else ret


def beam_current(*integrate_args, **integrate_kwargs):
    """Helper to compute ion beam current. Integrates ion currents over outflow boundary.
    
    See integrate_boundary for arguments.
    
    :returns: the ion beam current (A)
    """
    opts = copy.deepcopy(integrate_kwargs)
    opts['boundary'] = 'outflow'
    opts['integrand_opts'] = opts.get('integrand_opts', {})
    opts['integrand_opts']['which'] = 'ion'

    return integrate_boundary(*integrate_args, _current_integrand, group_boundary=thruster_edges, **opts)
