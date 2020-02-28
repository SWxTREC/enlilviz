import pytest
import numpy as np
import xarray as xr

from enlilviz.enlil import Enlil, Evolution


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

    # Slices
    def gen_slice_data(shape):
        """Generate fake data.
        dim1: linear
        dim2: sinusoidal
        dim3: quadratic
        """
        x = np.arange(shape[0])
        y = np.sin(np.linspace(0, 2*np.pi, shape[1]))
        z = np.arange(shape[2])**2

        data = (x[:, np.newaxis, np.newaxis] *
                y[np.newaxis, :, np.newaxis] *
                z[np.newaxis, np.newaxis, :])
        return data

    slice_vars = ['den', 'vel', 'pol', 'cme']
    data_dict = {}
    data_dims = {'r': (nt, len(lon), len(lat)),
                 'lat': (nt, len(lon), len(r)),
                 'lon': (nt, len(lat), len(r))}
    data_labels = {'r': ['t', 'lon', 'lat'],
                   'lat': ['t', 'lon', 'r'],
                   'lon': ['t', 'lat', 'r']}
    for var in slice_vars:
        for pos in positions:
            name = 'slice_' + var + '_' + pos
            data = gen_slice_data(data_dims[pos])
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
def evolution():
    """Enlil Evolution xarray Dataset."""
    ds = None
    return Evolution(ds)
