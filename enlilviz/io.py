"""Input routines for reading Enlil netcdf files."""
import numpy as np
import xarray as xr

from enlilviz.enlil import Enlil, Evolution

# Mass of hydrogen (kg)
_m_hydrogen = 1.6735575e-27
_m3_to_cm3 = (1./100)**3
_unit_conversion = {"AU": 1.496e+11,  # Earth to Sun distance (m)
                    "den": 1/_m_hydrogen * _m3_to_cm3,  # kg/m3 -> N/cm3
                    "vel": 1/1000}  # m/s -> km/s

# Default variables and satellites found in Enlil 2D output files
# Here we redefine variable names for more descriptive use
# TODO: Need to remove fixed number of satellites and variables
#       Probably need to find a way to parse the dataset variable names
#       right when the dataset is read in below
_variables = {'X1': 'pos_r', 'X2': 'pos_lat', 'X3': 'pos_lon',
              'V1': 'vel_r', 'V2': 'vel_lat', 'V3': 'vel_lon',
              'B1': 'mag_r', 'B2': 'mag_lat', 'B3': 'mag_lon',
              'Density': 'den', 'Temperature': 'temp',
              'DP_CME': 'cme', 'BP_POLARITY': 'pol'}
# Of course evo files are labeled differently
_variables_evo = {'X1': 'pos_r', 'X2': 'pos_lat', 'X3': 'pos_lon',
                  'V1': 'vel_r', 'V2': 'vel_lat', 'V3': 'vel_lon',
                  'B1': 'mag_r', 'B2': 'mag_lat', 'B3': 'mag_lon',
                  'D': 'den', 'T': 'temp', 'DP': 'cme', 'BP': 'pol'}
_satellites = ['Earth', 'STEREO_A', 'STEREO_B']


def load_example():
    """
    Loads example data that can be used for demonstration and development.

    Returns
    -------
    enlil.Enlil
        An Enlil class representing the example dataset.
    """
    coordinates = ('r', 'lat', 'lon')
    vars = ('den', 'vel', 'pol', 'cme')
    satellites = ['Earth', 'STEREO_A', 'STEREO_B']
    nsat = len(satellites)

    nr = 512
    r = np.linspace(0.101, 1.698, nr)
    nlat = 60
    lat = np.linspace(59, -59, nlat)
    nlon = 180
    lon = np.linspace(-179, 179, nlon)
    # Test functions (r: linear, lat: sin, lon: cos)
    functions = {'r': lambda x: (x - np.min(r))/(np.max(r) - np.min(r)),
                 'lat': lambda x: np.abs(np.sin(np.deg2rad(x))),
                 'lon': lambda x: np.abs(np.cos(np.deg2rad(x)))}

    # Transform to min/max of the specific variable
    var_range = {'den': (1, 40), 'vel': (300, 800),
                 'pol': (-1, 1), 'cme': (-5e-24, 5e-24)}

    t1 = np.datetime64('2020-01-01T00:00')
    t2 = np.datetime64('2020-01-05T00:00')
    # hourly time series for 4 days
    t = np.arange(t1, t2, dtype='datetime64[60m]')
    nt = len(t)
    # Satellite data is finer resolution
    earth_t = np.arange(t1, t2, dtype='datetime64[30m]')
    nt_earth = len(earth_t)

    da_dict = {}
    for coord in coordinates:

        # Slice variables
        for var in vars:
            if coord == 'lon':
                # t, lat, r
                dims = ['t', 'lat', 'r']
                coords = [t, lat, r]
                data = (functions['r'](r[np.newaxis, np.newaxis, :]) *
                        functions['lat'](lat[np.newaxis, :, np.newaxis]) *
                        np.ones(shape=(nt, nlat, nr)))
            elif coord == 'lat':
                dims = ['t', 'lon', 'r']
                coords = [t, lon, r]
                data = (functions['r'](r[np.newaxis, np.newaxis, :]) *
                        functions['lon'](lon[np.newaxis, :, np.newaxis]) *
                        np.ones(shape=(nt, nlon, nr)))
            else:  # r
                dims = ['t', 'lon', 'lat']
                coords = [t, lon, lat]
                data = (functions['lat'](lat[np.newaxis, np.newaxis, :]) *
                        functions['lon'](lon[np.newaxis, :, np.newaxis]) *
                        np.ones(shape=(nt, nlon, nlat)))

            # Transform the data to min/max of the variable range
            data = ((data - np.min(data)) / (np.max(data) - np.min(data)) *
                    (var_range[var][1] - var_range[var][0]) +
                    var_range[var][0])

            da = xr.DataArray(data=data, dims=dims, coords=coords,
                              name=f'slice_{var}_{coord}')
            da_dict[da.name] = da

        # Time series at satellites
        dims = ['satellite', 'earth_t']
        coords = [satellites, earth_t]

        # Position
        data = np.ones(shape=(nsat, nt_earth))
        if coord == 'lat':
            data[0, :] = 3
            data[1, :] = -3
            data[2, :] = 8
        elif coord == 'lon':
            # Make the STEREO satellites be perpendicular to Earth
            data[1, :] = -90
            data[2, :] = 90

        da_dict[f'pos_{coord}'] = xr.DataArray(data=data, dims=dims,
                                               coords=coords)

        # Velocity
        data = np.ones(shape=(nsat, nt_earth))
        if coord == 'r':
            data *= np.linspace(*var_range['vel'], nt_earth)[np.newaxis, :]

        da_dict[f'vel_{coord}'] = xr.DataArray(data=data, dims=dims,
                                               coords=coords)

    # non-coordinate data

    # Density
    data = np.ones(shape=(nsat, nt_earth))
    data *= np.linspace(*var_range['den'], nt_earth)[np.newaxis, :]
    da_dict['den'] = xr.DataArray(data=data, dims=dims,
                                  coords=coords)

    # temp
    data = np.ones(shape=(nsat, nt_earth))
    data *= np.linspace(8000, 20000, nt_earth)[np.newaxis, :]
    da_dict['temp'] = xr.DataArray(data=data, dims=dims,
                                   coords=coords)

    # cme
    data = np.ones(shape=(nsat, nt_earth))
    data *= np.linspace(*var_range['cme'], nt_earth)[np.newaxis, :]
    da_dict['cme'] = xr.DataArray(data=data, dims=dims,
                                  coords=coords)

    # pol
    data = np.ones(shape=(nsat, nt_earth))
    data *= np.linspace(*var_range['pol'], nt_earth)[np.newaxis, :]
    da_dict['pol'] = xr.DataArray(data=data, dims=dims,
                                  coords=coords)

    ds = xr.Dataset(da_dict)
    ds.attrs = {'enlil_version': 'Example',
                'wsa_version': 'Example',
                'model_run_id': 1}
    return Enlil(ds)


def read_enlil2d(filename):
    """Load a 2D post-processed Enlil file into an Enlil object.

    Parameters
    ----------
    filename : str
        netcdf Enlil post-processed output file
        Example: wsa_enlil.latest.suball.nc

    Returns
    -------
    enlil.Enlil
        An Enlil class representing the loaded file.
    """
    ds = xr.load_dataset(filename)

    # Now transform the dimensions and coordinates
    ds = _transform_dimensions(ds)

    # Are fieldline variables present? Store as a list [False] default
    # [False, True] if they are present, which helps with the loops
    # below. Could probably use to be refactored at some point.
    fieldline_vars = [False]
    if 'fld_step' in ds:
        # Make a new coordinate, step, for the collapsed +/- fieldline
        ds = ds.assign_coords({'step': np.concatenate([-ds['fld_step'][::-1],
                                                       ds['fld_step']+1])})
        fieldline_vars += [True]

    # Transform and calibrate all of the slices
    slice_variables = ['dd12_3d', 'vv12_3d', 'pp12_3d', 'cc12_3d',
                       'dd13_3d', 'vv13_3d', 'pp13_3d', 'cc13_3d',
                       'dd23_3d', 'vv23_3d', 'pp23_3d', 'cc23_3d']
    for var in slice_variables:
        try:
            ds = _calibrate_variable(ds, var)
        except KeyError:
            # Ignore if the variable wasn't found in the dataset
            pass

    # Now work on the field line and satellite data
    for fieldline in fieldline_vars:
        for var in _variables:
            name = ''

            # Combine all of the satellites together
            if fieldline:
                # These variables aren't on the fieldlines
                if _variables[var] in ['cme', 'pol']:
                    continue
                da = xr.concat([_unstack_fieldline(ds[sat + '_FLD_' + var])
                                for sat in _satellites],
                               dim='satellite')
                ds = ds.drop_vars([sat + '_FLD_' + var for sat in _satellites])
                name += 'fld_'
            else:
                try:
                    da = xr.concat([ds[sat + '_' + var] for sat in _satellites],
                                dim='satellite')
                    ds = ds.drop_vars([sat + '_' + var for sat in _satellites])
                except KeyError:
                    # If the variable isn't here, continue the loop
                    continue

            name += _variables[var]
            da.name = name

            da = _transform_variable(da)
            ds[name] = da

    return Enlil(ds)


def read_evo(filename):
    """Load an `evo` post-processed Enlil file into an Evolution object.

    Parameters
    ----------
    filename : str
        netcdf Enlil post-processed output file
        Example: evo.earth.nc

    Returns
    -------
    enlil.Evolution
        An Evolution class representing the loaded file.
    """
    ds = xr.load_dataset(filename)

    # Change the dimension to time
    # Depending on which version, the key could be rundate or refdate
    try:
        t0 = np.datetime64(ds.attrs['rundate_cal'], 's')
    except KeyError:
        t0 = np.datetime64(ds.attrs['refdate_cal'], 's')
    time = t0 + np.array(ds['TIME'], np.timedelta64)
    ds = ds.rename({'nevo': 'earth_t'}).assign_coords({'earth_t': time})
    ds = ds.drop_vars(['TIME', 'DT', 'NSTEP'])

    for var in _variables_evo:
        if var not in ds:
            continue

        da = ds[var]
        # Update the name to be consistent across data sets
        name = _variables_evo[var]
        da.name = name

        # Unit conversions
        da = _transform_variable(da)
        ds[name] = da
        ds = ds.drop_vars(var)
        ds.attrs['name'] = ds.label

    return Evolution(ds)


def _calibrate_variable(ds, var):
    """Calibrates a variable from the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable to calibrate.
    var : str
        Variable to calibrate.

    Raises
    ------
    KeyError
        If the variable is not in the dataset.

    NOTES
    -----
    All variables whose names begin with 'uncalibrated' are calibrated
    with the following formula (linear transform):
    var_calibrated=((var_uncalibrated-cal_min) *
                    (var_max-param_min)/cal_range) + var_min
    where param_max and param_min are attributes associated with each
    uncalibrated parameter.
    """
    if 'cal_min' in ds.attrs:
        cal_min = ds.attrs['cal_min']
        cal_range = ds.attrs['cal_range']
    else:
        # Calibration information not in the file
        cal_min = -32768
        cal_range = 65535

    if (var[:4] + '_min' not in ds[var].attrs or
            var[:4] + '_max' not in ds[var].attrs):
        raise KeyError('Variable: ' + var + ' has no calibration ' +
                       'information, it may have already been calibrated.')

    var_min = ds[var].attrs.pop(var[:4] + '_min')
    var_max = ds[var].attrs.pop(var[:4] + '_max')

    # Linear fit
    da = ((ds[var].astype(np.float64) - cal_min) *
          (var_max - var_min) / cal_range + var_min)

    # Change to a more descriptive name
    name = 'slice_'

    # Convert variables and update units attributes
    if var[:2] == 'vv':
        da *= _unit_conversion['vel']
        da.attrs['units'] = 'km/s'
        da.attrs['long_name'] = 'velocity'
        name += 'vel'
    elif var[:2] == 'dd':
        da *= _unit_conversion['den']
        if '1' in var:
            # Only multiply the lat/lon slices by r**2
            da *= ds['r']**2
        da.attrs['units'] = 'r2 N/cm3'
        da.attrs['long_name'] = 'density'
        name += 'den'
    elif var[:2] == 'pp':
        da.attrs['long_name'] = 'magnetic polarity'
        name += 'pol'
    elif var[:2] == 'cc':
        da.attrs['long_name'] = 'cloud parameter'
        name += 'cme'

    # Change name attributes for the slices
    if '12' in var:
        da.attrs['long_name'] += ' on 0 longitude slice'
        name += '_lon'
    elif '13' in var:
        da.attrs['long_name'] += ' on Earth latitude slice'
        name += '_lat'
    elif '23' in var:
        da.attrs['long_name'] += ' on 1 AU radial slice'
        name += '_r'

    # Update the name of the data array
    da.name = name
    ds[name] = da
    ds = ds.drop_vars(var)
    return ds


def _transform_dimensions(ds):
    """Changes dimension names and adds coordinate values.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to change dimensions and coordinates of.
    """
    t0 = np.datetime64(ds.attrs['REFDATE_CAL'], 's')
    t = t0 + np.array(ds['time'], np.timedelta64)
    earth_t = t0 + np.array(ds['Earth_TIME'], np.timedelta64)
    ds = ds.assign_coords({'x': ds['x_coord'],
                           'y': ds['y_coord'],
                           'z': ds['z_coord'],
                           't': t,
                           'earth_t': earth_t,
                           'satellite': _satellites,
                           }).drop_vars(['x_coord', 'y_coord', 'z_coord',
                                         'time', 'Earth_TIME'])

    ds['x'] = ds['x'] / _unit_conversion['AU']
    ds['x'].attrs = {'long_name': 'radial cell positions',
                     'units': 'AU'}
    ds['y'] = np.rad2deg(np.pi/2 - ds['y'])
    ds['y'].attrs = {'long_name': 'latitude cell positions',
                     'units': 'degrees_north'}
    ds['z'] = np.rad2deg(ds['z'] - np.pi)
    ds['z'].attrs = {'long_name': 'longitude cell positions',
                     'units': 'degrees_east'}
    ds = ds.rename({'x': 'r', 'y': 'lat', 'z': 'lon'})
    return ds


def _transform_variable(da):
    """Transform the variable and add proper attributes and units.

    Parameters
    ----------
    da : xarray.DataArray
        The DataArray to update the units and attributes of.
    """

    # Position validation is coordinate dependent so do those
    # conversions first
    if da.name == 'pos_r':
        da /= _unit_conversion['AU']
        da.attrs['units'] = 'AU'
    elif da.name == 'pos_lat':
        da.values = np.rad2deg(np.pi/2 - da)
        da.attrs['units'] = 'degrees_north'
    elif da.name == 'pos_lon':
        da.values = np.rad2deg(da - np.pi)
        da.attrs['units'] = 'degrees_east'

    if 'r' in da.name:
        da.attrs['long_name'] = 'radial '
    elif 'lat' in da.name:
        da.attrs['long_name'] = 'latitudinal '
    elif 'lon' in da.name:
        da.attrs['long_name'] = 'longitudinal '

    if 'pos' in da.name:
        da.attrs['long_name'] += 'position'
    elif 'vel' in da.name:
        da *= _unit_conversion['vel']
        da.attrs['long_name'] += 'velocity'
        da.attrs['units'] = 'km/s'
    elif 'mag' in da.name:
        da *= 1e9  # T to nT
        da.attrs['long_name'] += 'magnetic field'
        da.attrs['units'] = 'nT'
    elif 'den' in da.name:
        da *= _unit_conversion['den']
        da.attrs['long_name'] = 'density'
        da.attrs['units'] = '#/cm3'
    elif 'temp' in da.name:
        da.attrs['long_name'] = 'temperature'
        da.attrs['units'] = 'K'
    elif da.name == 'cme':
        da.attrs['long_name'] = 'cloud parameter'
    elif da.name == 'pol':
        da.attrs['long_name'] = 'magnetic polarity'

    return da


def _unstack_fieldline(da):
    """Unstacks the forward/backward field line trace variables.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with `fld_step` dimension that will be flattened.
    """
    return xr.concat([da[:, ::-1, 0], da[:, :, 1]],
                     dim='fld_step').rename({'fld_step': 'step'})
