import os
from tempfile import NamedTemporaryFile

import pytest
import numpy as np
import xarray as xr

from enlilviz.enlil import Enlil
from enlilviz import read_evo
from enlilviz.io import _unit_conversion

# Data scaling
# Range of variables
scale_factors = {'den': (0.1, 60),
                 'temp': (500, 20000),
                 'cme': (-0.1, 0.1),
                 'pol': (-1, 1),
                 'vel': (300, 1200),
                 'mag': (-20, 20)}


def scale_variable(x, var):
    """Scale the variable to a more sensible range."""
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = scale_factors[var]

    return (x - xmin) / (xmax - xmin) * (ymax - ymin) + ymin


@pytest.fixture
def enlil_run():
    """Enlil 2d xarray Dataset."""
    # Satellite times
    earth_t = np.arange(np.datetime64("2010-01-01"),
                        np.datetime64("2010-01-10"))
    n_earth_t = len(earth_t)
    # Slice times (make it shorter on purpose)
    t = np.arange(np.datetime64("2010-01-01"),
                  np.datetime64("2010-01-05"))
    nt = len(t)
    # Positions (angles in degrees)
    positions = ['r', 'lat', 'lon']
    r = np.linspace(0.1, 1.7, 5)
    lat = np.linspace(-59, 59, 10)
    lon = np.linspace(-179, 179, 15)
    step = np.arange(-5, 5)
    nstep = len(step)

    # satellites
    satellites = ['Earth', 'STEREO_A', 'STEREO_B']
    nsat = len(satellites)

    coord_dict = {'earth_t': earth_t, 't': t,
                  'r': r, 'lat': lat, 'lon': lon,
                  'satellite': satellites, 'step': step}

    # Slices
    """Generate fake data.
    dim1: linear
    dim2: sinusoidal
    dim3: quadratic
    """
    x = np.arange(len(r))
    y = np.sin(np.linspace(0, np.pi, len(lat)))
    z = np.arange(len(lon))**2

    data_cube = (x[:, np.newaxis, np.newaxis] +
                 y[np.newaxis, :, np.newaxis] +
                 z[np.newaxis, np.newaxis, :])

    slice_vars = ['den', 'vel', 'pol', 'cme']
    data_dict = {}
    data_labels = {'r': ['t', 'lon', 'lat'],
                   'lat': ['t', 'lon', 'r'],
                   'lon': ['t', 'lat', 'r']}
    for var in slice_vars:
        for pos in positions:
            name = 'slice_' + var + '_' + pos
            if pos == 'r':
                data = data_cube[0, :, :].T
            elif pos == 'lat':
                data = data_cube[:, 0, :].T
            elif pos == 'lon':
                data = data_cube[:, :, 0].T
            # Add the time dimension
            data = np.tile(data, (nt, 1, 1))
            data = scale_variable(data, var)
            data_dict[name] = xr.DataArray(data, dims=data_labels[pos])

    # Satellite time series
    ts_scalar_vars = ['den', 'temp', 'cme', 'pol']
    ts_vector_vars = ['vel', 'mag']

    for var in ts_scalar_vars:
        data = scale_variable(np.arange(n_earth_t), var)
        # Make sure we make it shape nsat x n_earth_t
        data = np.broadcast_to(data, (nsat, n_earth_t))
        data_dict[var] = xr.DataArray(data, dims=['satellite', 'earth_t'])
        fld_data = scale_variable(np.arange(nstep), var)
        # Expand dimensions to the proper size
        fld_data = np.tile(fld_data, (nsat, nt, 1))
        data_dict['fld_' + var] = xr.DataArray(fld_data,
                                               dims=['satellite', 't', 'step'])

    for coord in ['r', 'lat', 'lon']:
        for var in ts_vector_vars:
            name = var + '_' + coord
            data = scale_variable(np.arange(n_earth_t), var)
            data = np.broadcast_to(data, (nsat, n_earth_t))
            data_dict[name] = xr.DataArray(data, dims=['satellite', 'earth_t'])
            fld_name = 'fld_' + name
            fld_data = scale_variable(np.arange(nstep), var)
            fld_data = np.tile(fld_data, (nsat, nt, 1))
            data_dict[fld_name] = xr.DataArray(fld_data,
                                               dims=['satellite', 't', 'step'])

        # Do position separately
        name = 'pos_' + coord
        fld_name = 'fld_' + name
        if coord == 'r':
            data = np.linspace(0.1, 1.7, n_earth_t)
            fld_data = np.linspace(0.1, 1.7, nstep)
        elif coord == 'lat':
            data = np.linspace(-45, 45, n_earth_t)
            fld_data = np.linspace(-45, 45, nstep)
        elif coord == 'lon':
            data = np.linspace(-80, 120, n_earth_t)
            fld_data = np.linspace(-80, 120, nstep)
        data = np.broadcast_to(data, (nsat, n_earth_t))
        data_dict[name] = xr.DataArray(data, dims=['satellite', 'earth_t'])
        fld_data = np.tile(fld_data, (nsat, nt, 1))
        data_dict[fld_name] = xr.DataArray(fld_data,
                                           dims=['satellite', 't', 'step'])

    ds = xr.Dataset(data_dict, coords=coord_dict)
    # Extra attributes used in the plots
    ds.attrs = {'enlil_version': '2.6',
                'wsa_version': 'WSA_V2.2',
                'model_run_id': '25677'}
    return Enlil(ds)


@pytest.fixture
def evo():
    """Creates a test evolution xarray file."""
    nevo = 20

    gen_data = {1: np.arange(nevo),
                2: np.sin(np.linspace(0, 2*np.pi, nevo)),
                3: np.arange(nevo)**2}

    data = {'X1': np.linspace(0.1, 1.7, nevo)*_unit_conversion['AU'],
            'X2': np.deg2rad(np.linspace(60, 120, nevo)),
            'X3': np.deg2rad(np.linspace(30, 80, nevo)),
            'TIME': np.arange(nevo)*60*60*24,
            'DT': np.arange(nevo),
            'NSTEP': np.arange(nevo),
            'D': scale_variable(gen_data[3], 'den')/_unit_conversion['den'],
            'T': scale_variable(gen_data[2], 'temp'),
            'V1': scale_variable(gen_data[1], 'vel')/_unit_conversion['vel'],
            'V2': scale_variable(gen_data[2], 'vel')/_unit_conversion['vel'],
            'V3': scale_variable(gen_data[3], 'vel')/_unit_conversion['vel'],
            'B1': scale_variable(gen_data[1], 'mag'),
            'B2': scale_variable(gen_data[2], 'mag'),
            'B3': scale_variable(gen_data[3], 'mag'),
            'DP': np.linspace(0, 0.1, nevo),
            'BP': np.linspace(-1, 1, nevo)}
    # Need to make data Arrays for all of the variables with the single dim
    for x in data:
        data[x] = xr.DataArray(data[x], dims=['nevo'])

    ds = xr.Dataset(data, coords={'nevo': np.arange(nevo)})

    ds.attrs = {'label': 'earth',
                'rundate_cal': "2010-01-01T00"}

    with NamedTemporaryFile(suffix='.nc', delete=False) as f:
        ds.to_netcdf(f.name)

        evo = read_evo(f.name)
        f.close()
        os.unlink(f.name)

    return evo
