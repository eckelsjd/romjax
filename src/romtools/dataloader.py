"""Data loading. Primarily interested in PDE finite-volume numerical data (only 2d quadrilaterals currently).
General idea is to have a "Rosetta" format in Python that works for everyone, and save/load to/from this format to convert.
This internal format will evolve a lot as more external formats are supported.

With a common format, then ROM scripts, plotting, post-processing etc. will be agnostic to the source of the data.

Supported data formats:
    Dictionary
    ----------
    - Readable dictionary, easy to parse to/from other formats
    - Use as the Rosetta stone between other formats and plotting scripts
    - Extendable with other attributes as needed (connectivity, etc.)
    - Each simulation time step frame is formatted as: frame i = {'node': {varX: [X data], varY: [Y data], ...}, 'cell': {...}}
    - In general, first two variables for nodes and cells give the z and r coordinates.
    - {'frames'      : [{frame 0}, {frame 1}, {...}],                         # simulation data frames (1 per timestep)
       'time'        : [time 0, time 1, ...],                                 # simulation time steps (seconds)
       'connectivity': [[quad 1], [quad 2], ...]                              # mesh quadrilateral connectivity
       'performance' : {'thrust': (Nt,), 'discharge_current': (Nt,), ...}     # performance integrated quantities
       'centerline'  : {'varX': (Nt, Nslice), 'varY': (Nt, Nslice), ...}      # channel-centerline 1d slice time history
       'species'     : 'Xe'                                                   # plasma species
       'name'        : 'SPT-100 case description'                             # description of the simulation
      }
    - If passed through the filter_data function, the format is mostly the same except node/cell coordinates are moved
      to the top level as the 'cells' and 'nodes' keys, and the node/cell data in each frame are move to the top-level
      of each frame, i.e. each frame only contains all node or all cell data. This filtered dictionary must be used
      for saving/loading to the Numpy format.
                
    Tecplot
    -------
    - Human-readable ASCII text data stored in .dat files
    - TITLE, VARIABLES, and ZONE header information posted at each simulation frame (i.e. time step) followed by a 
        BLOCK of the field data and cell connectivity data
    - VARIABLES is a list of all the variable names (ordered)
    - ZONE use FEQUADRILATERAL data in BLOCK format, which means all data for var #1 is listed, then var #2, and so on.
        This script will only work if each variable's block data ends at the end of a line, so that the next variable starts
        on the next line etc.
    - VARLOCATION info tells which variables are cell-centered (1-indexed) versus node-value (default). Node and cell data
        are still listed sequentially regardless in the data BLOCK following the VARIABLES ordering.
    - The T attribute in the ZONE header is used to tell the simulation time for that simulation frame.
    - Cell connectivity gives the list of node numbers for each each cell (4 total for quadrilaterals)
    - Coordiante locations (z,r) are assumed to be the first two node/cell variables
    - Coordinate locations and connectivity are only given in the first ZONE frame and shared via
      VARSHARELIST and CONNECTIVITYSHAREZONE

    HDF5
    ----
    - Binary stored in hierarchical tree structure (.h5)
    - Same structure as Dictionary, but can include more metadata/attributes
    - Preferred for long-term storage and quicker loading

    Numpy
    -----
    - Numpy ndarray of shape (N_time, N_cells (N_nodes), N_vars)
    - Can be compressed/loaded from .npz
    - Preferred as condensed numerical format for running algorithms
    - Coupled with arrays for [time], [coords], [variables] for indexing the simulation data array

Includes:
  - load_tecplot - extract nodal and cell data from 2d tecplot output .dat files
  - save_tecplot - save data to tecplot .dat file
  - load_h5 - extract data from .h5 output files
  - save_h5 - save data to .h5 file
  - filter_data - filter Dictionary data 
  - to_numpy - convert dictionary to numpy array
  - from_numpy - convert numpy array to dictionary
  - load_numpy - load numpy data from .npz
  - save_numpy - save numpy data to .npz
"""
# ruff: noqa
import copy
import re
from pathlib import Path
from typing import Iterable, Literal

import h5py
import numpy as np
from numpy.typing import ArrayLike

__all__ = ['load_tecplot', 'save_tecplot', 'load_h5', 'save_h5', 'load_numpy', 'save_numpy', 'filter_data', 'to_numpy', 'from_numpy', 'FRAME_KEY']

FRAME_KEY = 'frames'
CHAR_REPLACE = {
    "/": "%"
}

# TODO: remove these specific attributes
# Special attributes to describe simulation data
ATTRS = {
    'name': 'My cool simulation',
    'species': 'Xe'
}  


def _replace_chars(s: str, inverse: bool = False):
    """Replace all disallowed characters in string to be filename compatible.

    For example:
        '/' -> '%' for group names in h5 files
    
    :param s: string to replace characters
    :param inverse: do the replacement in reverse
    """
    new_s = s
    for c, r in CHAR_REPLACE.items():
        new_s = new_s.replace(r, c) if inverse else new_s.replace(c, r)
    return new_s


def load_tecplot(filename: str | Path, t_start: float = 0.0, t_end: float = -1.0):
    """Extract Tecplot data into the Dictionary format.

    Assumptions on file format:
      - TITLE and VARIABLES listed at top of file before first ZONE
      - "_Species_" substring should be provided in the title
      - Everything to the left of first hypen in the title is saved as a description of the simulation
      - T="float" attribute in every ZONE gives the time step (in seconds)
      - N= and E= attributes in every ZONE give number of nodes and elements (same for all zones)
      - Only BLOCK data and FEQUADRILATERAL type supported
      - A single []=CELLCENTERED group may be included in the VARLOCATION attribute of the ZONE header
        to indicate cell variables (assumed the same for every zone)
      - Shared variables are located in the first ZONE, i.e. VARSHARELIST([]=1) in all zones > 1
      - Connectivity list is located only in the first ZONE, i.e. CONNECTIVITYSHAREZONE=1 in all zones > 1
      - Generally, first 2 node variables are (z,r) coordinates, and first 2 cell variables are (zC, rC) coordinates
    
    :param filename: the .dat tecplot file to load from
    :param t_start: the simulation time to start loading data from (default 0.0)
    :param t_end: the simulation time after which to stop loading data from (default -1.0 which goes to the end)
    :returns: {data         - list of node/cell data at each time step
               time         - list of simulation times (s)
               connectivity - list of nodes for each cell
               species      - plasma species
               name         - description of simulation from the title}
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if t_end < 0:
        t_end = np.inf

    attrs = copy.deepcopy(ATTRS)

    # Parse variable names
    var_names = []
    collecting_vars = False
    vars_buffer = ""

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("TITLE"):
            l = stripped.split('-')
            attrs['name'] = l[0].split('"')[1].strip()
            l = stripped.split('_')
            for i, ele in enumerate(l):
                if "species" in ele.lower():
                    attrs['species'] = l[i+1]
                    break
        elif stripped.upper().startswith("VARIABLES"):
            collecting_vars = True
            vars_buffer += stripped
        elif collecting_vars and not stripped.upper().startswith("ZONE"):
            vars_buffer += " " + stripped
        elif stripped.upper().startswith("ZONE"):
            break

    var_names = re.findall(r'"([^"]+)"', vars_buffer)
    num_vars = len(var_names)

    zones = []      # list of dicts containing cell/node data at each time step
    sim_times = []  # list of simulation times (s)
    conn = []       # list of cell connectivity

    cell_centered_vars = set()
    parsed_zone_metadata = False
    parsed_connectivity = False

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("ZONE"):
            # Parse zone header
            zone_header = ''
            while True:
                if any(key in lines[i].strip() for key in ['ZONE', 'ZONETYPE', 'VARLOCATION', 'VARSHARELIST']):
                    zone_header += ' ' + lines[i].strip()
                    i += 1
                else:
                    break

            # Extract metadata (Time, Nodes, Elements)
            t_match = re.search(r'T\s*=\s*(?:"([^"]+)"|([^\s,]+))', zone_header)
            if t_match:
                time_str = t_match.group(1) or t_match.group(2)
                try:
                    sim_time = float(time_str)
                except ValueError:
                    sim_time = np.nan  # Non-numeric zone title
            else:
                sim_time = np.nan
            
            if sim_time < t_start:
                i += 1
                parsed_connectivity = True  # If we skip first ZONE, there will be no connectivity
                continue
            
            if sim_time > t_end:
                break

            sim_times.append(sim_time)

            # Parse VARSHARELIST and CONNECTIVITYSHAREZONE (assumes only 1 zone gets shared)
            # conn_share_zone = None
            # var_share_zone = None
            shared_vars = set()
            varshare_match = re.search(r'VARSHARELIST\s*=\s*\(\s*(.*?)\s*\)', zone_header, re.IGNORECASE)
            if varshare_match:
                varshare_str = varshare_match.group(1)
                # var_share_zone = int(varshare_str.split('=')[1].strip())

                entries = re.findall(r'\[([^\]]+)\]\s*=', varshare_str, flags=re.IGNORECASE)
                for entry in entries:
                    tokens = [token.strip() for token in entry.split(',')]
                    for token in tokens:
                        if '-' in token:
                            start, end = map(int, token.split('-'))
                            shared_vars.update(range(start - 1, end))  # 0-based
                        else:
                            shared_vars.add(int(token) - 1)
            
            # connshare_match = re.search(r'CONNECTIVITYSHAREZONE\s*=\s*\(\s*(.*?)\s*\)', zone_header, re.IGNORECASE)
            # if connshare_match:
            #     connshare_str = connshare_match.group(1)
            #     conn_share_zone = int(connshare_str.split('=')[1].strip())

            # Only need to parse once (all zones have same metadata except T)
            if not parsed_zone_metadata:
                n_match = re.search(r'N\s*=\s*(\d+)', zone_header)
                e_match = re.search(r'E\s*=\s*(\d+)', zone_header)
                n_nodes = int(n_match.group(1))
                n_cells = int(e_match.group(1))

                # Check ZONETYPE and DATAPACKING
                if "ZONETYPE=FEQUADRILATERAL" not in zone_header.upper():
                    raise ValueError("Only ZONETYPE=FEQUADRILATERAL is supported.")
                if "DATAPACKING=BLOCK" not in zone_header.upper():
                    raise ValueError("Only DATAPACKING=BLOCK is supported.")

                # Parse VARLOCATION
                varloc_match = re.search(r'VARLOCATION\s*=\s*\(\s*(.*?)\s*\)', zone_header, re.IGNORECASE)
                if varloc_match:
                    varloc_str = varloc_match.group(1)
        
                    entries = re.findall(r'\[([^\]]+)\]\s*=\s*CELLCENTERED', varloc_str, flags=re.IGNORECASE)
                    for entry in entries:
                        tokens = [token.strip() for token in entry.split(',')]
                        for token in tokens:
                            if '-' in token:
                                start, end = map(int, token.split('-'))
                                cell_centered_vars.update(range(start - 1, end))  # 0-based
                            else:
                                cell_centered_vars.add(int(token) - 1)

                parsed_zone_metadata = True

            # Read BLOCK data (will only work if blocks are evenly divided on lines)
            data = []  # (N_vars, N_cells/N_nodes)
            var_no = 0
            while len(data) < num_vars - len(shared_vars):
                if var_no in shared_vars:  # Don't read a block for shared variables
                    var_no += 1
                    continue

                block = []
                while len(block) < (n_cells if var_no in cell_centered_vars else n_nodes):
                    line_data = lines[i].strip()
                    if line_data:
                        for x in line_data.split():
                            if x.startswith("#"):
                                break  # Ignore everything after a comment
                            block.append(float(x))
                    i += 1
                data.append(block)
                var_no += 1

            # Separate node/cell variables
            idx = 0
            node_data = {}
            cell_data = {}
            for j, var_name in enumerate(var_names):
                if j in shared_vars:
                    continue
                elif j in cell_centered_vars:
                    cell_data[var_name] = data[idx]
                    idx += 1
                else:
                    node_data[var_name] = data[idx]
                    idx += 1

            # Read connectivity (same for every zone)
            if not parsed_connectivity:
                while len(conn) < n_cells:
                    line_data = lines[i].strip()
                    if line_data:
                        nums = []
                        for x in line_data.split():
                            if x.startswith("#"):
                                break
                            nums.append(int(x) - 1)

                        if len(nums) != 4:
                            raise ValueError("Expected 4 nodes per quadrilateral element.")
                        conn.append(nums)
                    i += 1

                parsed_connectivity = True

            zone = {'node': node_data, 'cell': cell_data}
            zones.append(zone)
        else:
            i += 1
    
    ret_dict = {FRAME_KEY: zones, 'time': np.array(sim_times), **attrs}

    if len(conn) > 0:
        ret_dict['connectivity'] = np.array(conn)

    return ret_dict


def save_tecplot(data_dict: dict, filename: str | Path, overwrite: bool = False):
    """Save data from the common Dictionary format to a tecplot .dat file.
    
    :param data_dict: the data to save (see Dictionary format)
    :param filename: the .dat file to save to
    :param overwrite: whether to overwrite or append (default)
    """
    fframe = data_dict[FRAME_KEY][0]
    node_vars = list(fframe['node'].keys())
    cell_vars = list(fframe['cell'].keys())
    all_vars = node_vars + cell_vars
    title = f"{data_dict.get('name', ATTRS['name'])} - _Species_{data_dict.get('species', ATTRS['species'])}_ - plasma primitives (postprocess)"

    num_nodes = len(next(iter(fframe['node'].values())))
    num_cells = len(next(iter(fframe['cell'].values())))

    if overwrite or not Path(filename).exists():
        mode = 'w'
        initial_write = True
    else:
        mode = 'a'
        initial_write = False

    def _write_block(f, vec, num_per_line=10):
        """Helper to write block of data to file."""
        for i in range(0, len(vec), num_per_line):
            line = ' '.join(f"{x:14.6E}" for x in vec[i:i+num_per_line])
            f.write(line + '\n')

    with open(filename, mode) as fd:
        if initial_write:
            fd.write(f'TITLE="{title}"\n')
            fd.write('VARIABLES=' + ','.join(f'"{v}"' for v in all_vars) + '\n')
        
        for zone_id, (frame, t) in enumerate(zip(data_dict[FRAME_KEY], data_dict['time'])):
            is_first_zone = initial_write and (zone_id == 0)

            zone_header = f'ZONE T="{t:10.3E}", N={num_nodes}, E={num_cells}, DATAPACKING=BLOCK, ZONETYPE=FEQUADRILATERAL, ' 
            zone_header += f'VARLOCATION=([{len(node_vars)+1}-{len(all_vars)}]=CELLCENTERED)'

            # Add variable and connectivity sharing
            if not is_first_zone:
                zone_header += f'\nVARSHARELIST=([1-2,{len(node_vars)+1}-{len(node_vars)+2}]=1), CONNECTIVITYSHAREZONE=1'
            
            fd.write(zone_header + '\n')

            # Data BLOCK
            if is_first_zone:
                for v in all_vars:
                    vec = frame['node'].get(v, frame['cell'].get(v))
                    _write_block(fd, vec)
                
                for quad in data_dict['connectivity']:
                    fd.write(' '.join(f"{int(i+1):8d}" for i in quad) + '\n')
            else:
                for i, v in enumerate(node_vars):
                    if i < 2: continue
                    _write_block(fd, frame['node'].get(v))
                
                for i, v in enumerate(cell_vars):
                    if i < 2: continue
                    _write_block(fd, frame['cell'].get(v))


def save_h5(data_dict: dict, filename: str | Path, overwrite: bool = False):
    """Save data from the common Dictionary format to an .h5 file.
    
    :param data_dict: the data to save (see the Dictionary format described above)
    :param filename: the .h5 file to save to
    :param overwrite: whether to overwrite or append (default)
    """
    def _recursively_save(h5group, obj):
        """Helper to recursively save dictionary items to h5 file"""
        for key, val in obj.items():
            k = _replace_chars(key)
            if isinstance(val, dict):
                if k in h5group:
                    if overwrite:
                        del h5group[k]
                        subgroup = h5group.create_group(k, track_order=True)
                    else:
                        subgroup = h5group[k]
                else:
                    subgroup = h5group.create_group(k, track_order=True)
                _recursively_save(subgroup, val)
            else:
                if k in h5group:
                    if overwrite:
                        del h5group[k]
                        h5group.create_dataset(k, data=np.array(val))
                else:
                    h5group.create_dataset(k, data=np.array(val))

    with h5py.File(filename, 'a', track_order=True) as f:
        for key in data_dict:
            # Simulation spatial data
            if key == FRAME_KEY:
                if key in f:
                    if overwrite:
                        del f[key]
                        data_group = f.create_group(key, track_order=True)
                    else:
                        data_group = f[key]
                else:
                    data_group = f.create_group(key, track_order=True)
                
                num_nodes, num_cells = None, None

                for (t, frame) in zip(data_dict['time'], data_dict[FRAME_KEY]):
                    t_str = f"t={t:.6E}"
                    frame_group = data_group[t_str] if t_str in data_group else data_group.create_group(t_str, track_order=True)
                    frame_group.attrs['time'] = t

                    for loc in ['node', 'cell']:
                        if loc in frame:
                            loc_group = frame_group[loc] if loc in frame_group else frame_group.create_group(loc, track_order=True)

                            for var, array in frame[loc].items():
                                v = _replace_chars(var)
                                if v in loc_group:
                                    if overwrite:
                                        del loc_group[v]
                                        loc_group.create_dataset(v, data=np.array(array))
                                else:
                                    loc_group.create_dataset(v, data=np.array(array))
                                
                                if loc == 'node' and num_nodes is None:
                                    num_nodes = len(array)
                                if loc == 'cell' and num_cells is None:
                                    num_cells = len(array)
                if num_nodes is not None:
                    data_group.attrs['nodes'] = num_nodes
                if num_cells is not None:
                    data_group.attrs['cells'] = num_cells

            # Simulation time
            elif key == 'time':
                if key in f:
                    if overwrite:
                        del f[key]
                        dset = f.create_dataset(key, data=np.array(data_dict['time']), maxshape=(None,))
                    else:
                        dset = f[key]
                        dset.resize(dset.shape[0] + len(data_dict['time']), axis=0)
                        dset[dset.shape[0]:] = data_dict['time']
                else:
                    dset = f.create_dataset(key, data=np.array(data_dict['time']), maxshape=(None,))
                f['time'].attrs['units'] = 's'

            # Recurse on dictionaries
            elif isinstance(data_dict[key], dict):
                k = _replace_chars(key)
                if k in f:
                    if overwrite:
                        del f[k]
                        group = f.create_group(k, track_order=True)
                    else:
                        group = f[k]
                else:
                    group = f.create_group(k, track_order=True)
                _recursively_save(group, data_dict[key])
            
            # Special attributes
            elif key in ATTRS:
                f.attrs[key] = data_dict[key]
            
            # Anything else gets saved as an array
            else:
                k = _replace_chars(key)
                if k in f:
                    if overwrite:
                        del f[k]
                        f.create_dataset(k, data=np.array(data_dict[key]))
                else:
                    f.create_dataset(k, data=np.array(data_dict[key]))


def load_h5(filename: str | Path, t_start: float = 0.0, groups: list[str] = None):
    """Load data from .h5 file into the Dictionary format.
    
    :param filename: the .h5 file to load data from (structured as data/, time/, etc.)
    :param t_start: the simulation time to start loading data from (default 0.0)
    :param groups: the top-level groups to load (defaults to all groups if None)
    """
    def _recursively_load(h5group):
        """Helper to load an arbitrary h5 group."""
        result = {}
        for key in h5group:
            k = _replace_chars(key, inverse=True)
            item = h5group[key]
            if isinstance(item, h5py.Dataset):
                result[k] = item[()]
            elif isinstance(item, h5py.Group):
                result[k] = _recursively_load(item)
        return result

    with h5py.File(filename, 'r', track_order=True) as f:
        result = {}
        for a in f.attrs:
            result[a] = f.attrs.get(a)
            
        for key in f:
            if groups is not None and key not in groups:
                continue

            if key == FRAME_KEY:
                data_list = []
                data_group = f[FRAME_KEY]

                for frame_key in sorted(data_group.keys(), key=lambda x: float(x.split('=')[1])):
                    if float(frame_key.split('=')[1]) < t_start:
                        continue

                    frame_group = data_group[frame_key]
                    frame = {}

                    for loc in ['node', 'cell']:
                        if loc in frame_group:
                            loc_data = {}

                            for var in frame_group[loc]:
                                v = _replace_chars(var, inverse=True)
                                loc_data[v] = frame_group[loc][var][:]
                            frame[loc] = loc_data
                    data_list.append(frame)

                result[FRAME_KEY] = data_list
            elif key == 'time':
                times = f[key][:]
                idx_start = np.argmin(np.abs(times - t_start))
                result['time'] = times[idx_start:]
            elif isinstance(f[key], h5py.Group):
                k = _replace_chars(key, inverse=True)
                result[k] = _recursively_load(f[key])
            else:
                k = _replace_chars(key, inverse=True)
                result[k] = f[key][()]

    return result


def filter_data(d: dict,
                variables: list[str] = None,
                loc: Literal['cell', 'node'] = 'cell',
                select_frames: Iterable[int] = None,
                tlim: tuple[int, int] = (None, None),
                time_avg: bool = False,
                cell_coord_names: list[str] = None,
                node_coord_names: list[str] = None,
                ):
    """Filter output from the data Dictionary format.
    
    :param d: All field data, must be in the Dictionary format
    :param variables: the names of the variables to keep (defaults to all)
    :param loc: whether to keep cell-centered or node variables
    :param select_frames: time step indices to keep (defaults to all)
    :param tlim: only keep frames between these time limits (defaults to all)
    :param time_avg: whether to return time average values
    :param node_coord_names: Names of 2d z,r coordinate variables for nodes (defaults to first 2 node vars)
    :param cell_coord_names: Names of 2d z,r coordinate variables for cell centers (defaults to first 2 cell vars)
    
    :returns d_filtered: dictionary with same format as input dict, except filtered according to specified
                         arguments, also includes 'coords' for node/cell coordinates (N, 2)
    """
    d_filtered = {FRAME_KEY: [], 'time': []}
    fframe = d[FRAME_KEY][0]
    sim_time = d['time']
    assert 'node' in fframe and 'cell' in fframe, "Frames must have both 'node' and 'cell' data."
    node_vars = list(fframe['node'].keys())
    cell_vars = list(fframe['cell'].keys())

    if variables is None:
        variables = list(fframe[loc].keys())

    if node_coord_names is None:
        if len(node_vars) < 2:
            raise ValueError("Node coordinate names are inferred from the first two 'node' variables in the first frame.")
        node_coord_names = node_vars[:2]
    if cell_coord_names is None:
        if len(cell_vars) < 2:
            raise ValueError("Cell coordinate names are inferred from the first two 'cell' variables in the first frame.")
        cell_coord_names = cell_vars[:2]
    
    node_coords = np.vstack((fframe['node'][node_coord_names[0]], fframe['node'][node_coord_names[1]])).T  # (Nnode, 2)
    cell_coords = np.vstack((fframe['cell'][cell_coord_names[0]], fframe['cell'][cell_coord_names[1]])).T  # (Ncell, 2)

    d_filtered.update({'coords': node_coords if loc == 'node' else cell_coords})

    # Decide indices of frames to keep
    if tlim[0] is None:
        tlim = (0.5 * sim_time[0], tlim[1])
    if tlim[1] is None or tlim[1] < 0:
        tlim = (tlim[0], 2 * sim_time[-1])

    if select_frames is None:
        select_frames = range(len(d[FRAME_KEY]))

    for i in select_frames:
        if tlim[0] <= sim_time[i] <= tlim[1]:
            d_filtered['time'].append(sim_time[i])
            d_filtered[FRAME_KEY].append({v: d[FRAME_KEY][i][loc][v] for v in variables})
    
    if time_avg:
        N = len(d_filtered[FRAME_KEY])
        avg = {}
        for v in variables:
            arr = np.zeros(len(d_filtered[FRAME_KEY][0][v]))
            for i in range(N):
                arr += d_filtered[FRAME_KEY][i][v]
            avg[v] = arr / N
        
        d_filtered[FRAME_KEY] = [avg]
    
    d_filtered['time'] = np.array(d_filtered['time'])
    
    return d_filtered


def to_numpy(d: dict):
    """Convert filtered Dictionary data to numpy array. The dictionary should be in a filtered format (see filter_data).
    
    d = {'frames': [{var1: [...], var2: [...]}, {...}, ...],
         'time'  : [t1, t2, ...],
         'coords': (Ncells or Nnodes, 2)
         }
    
    :param d: data to save, must be in the filtered Dictionary format, only data in d['frames'] will be saved,
              along with d['time'] and d['coords']
    :returns time, coords, variables, data: the simulation times (Nt,), the (z,r) coordinates (Nx, 2), 
                                            the variable names (Nvar,), and the simulation data (Nt, Nx, Nvar)
    """
    variables = list(d['frames'][0].keys())

    Nt = len(d['frames'])
    Nx = len(d['coords'])
    Nvar = len(variables)

    data = np.zeros((Nt, Nx, Nvar))

    for i, frame in enumerate(d['frames']):
        for j, v in enumerate(variables):
            data[i, :, j] = frame[v]

    return d['time'], d['coords'], variables, data


def from_numpy(time: ArrayLike,
               coords: ArrayLike,
               variables: list[str],
               data: ArrayLike
               ):
    """Convert time, coords, variable names, and array data to a filtered Dictionary.
    
    :param time: (Nt,) array of simulation time (seconds)
    :param coords: (Nx, 2) array of (z,r) coordinates
    :param variables: (Nvar,) list of variable names
    :param data: (Nt, Nx, Nvar) array of simulation data
    :returns d: filtered Dictionary data
    """
    d = {'time': time, 'coords': coords, 'frames': []}

    for i in range(len(data)):
        d['frames'].append({v: data[i, :, j] for j, v in enumerate(variables)})
    
    return d


def save_numpy(time: ArrayLike,
               coords: ArrayLike,
               variables: list[str],
               data: ArrayLike,
               toloc: str | Path | dict):
    """Save Numpy array data to .npz or dict.

    :param time: (Nt,) array of simulation time (seconds)
    :param coords: (Nx, 2) array of (z,r) coordinates
    :param variables: (Nvar,) list of variable names
    :param data: (Nt, Nx, Nvar) array of simulation data
    :param toloc: .npz path to save array data. If a dict is passed, then save to the dict 
                   and rather than write to .npz file
    """
    d = dict(time=time, coords=coords, **{v: data[:, :, j] for j, v in enumerate(variables)})
    
    if isinstance(toloc, dict):
        toloc.update(d)
    else:
        np.savez(toloc, **d)


def load_numpy(fromloc: str | Path | dict):
    """Load Numpy array data from .npz or dict.
    
    :param fromloc: the .npz path to load. If a dict is passed, then load from the dict
    :returns time, coords, variables, data: the simulation times (Nt,), the (z,r) coordinates (Nx, 2), 
                                            the variable names (Nvar,), and the simulation data (Nt, Nx, Nvar)
    """
    special = {'time', 'coords'}

    if isinstance(fromloc, dict):
        res = (
            fromloc['time'],
            fromloc['coords'],
            [v for v in fromloc if v not in special],
            np.stack([arr for v, arr in fromloc.items() if v not in special], axis=2)
        )
    else:
        with np.load(fromloc) as fd:
            res = (
                fd['time'],
                fd['coords'],
                [v for v in fd if v not in special],
                np.stack([arr for v, arr in fd.items() if v not in special], axis=2)
            )
    
    return res
