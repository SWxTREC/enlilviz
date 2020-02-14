import numpy as np
import xarray as xr

from enlilviz.enlil import Enlil

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
_satellites = ['Earth', 'STEREO_A', 'STEREO_B']


def read_enlil2d(filename):
    """
    Load a 2D post-processed Enlil file into an *Enlil* class

    filename : netcdf Enlil post-processed output file
               wsa_enlil.latest.suball.nc
    """
    ds = xr.load_dataset(filename)

    # Now transform the dimensions and coordinates
    ds = _transform_dimensions(ds)

    # Make a new coordinate, step, for the collapsed +/- fieldline
    ds = ds.assign_coords({'step': np.concatenate([-ds['fld_step'][::-1],
                                                   ds['fld_step']+1])})

    # Transform and calibrate all of the slices
    slice_variables = ['dd12_3d', 'vv12_3d', 'pp12_3d', 'cc12_3d',
                       'dd13_3d', 'vv13_3d', 'pp13_3d', 'cc13_3d',
                       'dd23_3d', 'vv23_3d', 'pp23_3d', 'cc23_3d']
    for var in slice_variables:
        ds = _calibrate_variable(ds, var)

    # Now work on the field line and satellite data
    for fieldline in [False, True]:
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
                ds = ds.drop([sat + '_FLD_' + var for sat in _satellites])
                name += 'fld_'
            else:
                da = xr.concat([ds[sat + '_' + var] for sat in _satellites],
                               dim='satellite')
                ds = ds.drop([sat + '_' + var for sat in _satellites])

            name += _variables[var]
            da.name = name

            da = _transform_variable(da)
            ds[name] = da

    return Enlil(ds)


def _calibrate_variable(ds, var):
    """Calibrates the variable *var* from the dataset *ds*.
    All variables whose names begin with 'uncalibrated' are calibrated
    with the following formula (linear transform):
    var_calibrated=((var_uncalibrated-cal_min) *
                    (var_max-param_min)/cal_range) + var_min
    where param_max and param_min are attributes associated with each
    uncalibrated parameter.
    """
    cal_min = ds.attrs['cal_min']
    cal_range = ds.attrs['cal_range']
    if (var[:4] + '_min' not in ds[var].attrs or
            var[:4] + '_max' not in ds[var].attrs):
        raise KeyError('Variable: ' + var + ' has no calibration ' +
                       'information, it may have already been calibrated.')

    var_min = ds[var].attrs.pop(var[:4] + '_min')
    var_max = ds[var].attrs.pop(var[:4] + '_max')

    # Linear fit
    da = ((ds[var] - cal_min) *
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
        da.attrs['units'] = '#/cm3'
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
    ds = ds.drop(var)
    return ds


def _transform_dimensions(ds):
    """Changes dimension names and adds coordinate values."""
    t0 = np.datetime64(ds.attrs['REFDATE_CAL'])
    ds = ds.assign_coords({'x': ds['x_coord'],
                           'y': ds['y_coord'],
                           'z': ds['z_coord'],
                           't': t0 + ds['time'],
                           'earth_t': t0 + ds['Earth_TIME'],
                           'satellite': _satellites,
                           }).drop(['x_coord', 'y_coord', 'z_coord',
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
    """Transform the variable and add proper attributes and units."""

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
    """Unstacks the forward/backward field line trace variables"""
    return xr.concat([da[:, ::-1, 0], da[:, :, 1]],
                     dim='fld_step').rename({'fld_step': 'step'})
