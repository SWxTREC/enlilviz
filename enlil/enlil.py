"""Main enlil data module."""
import numpy as np
import xarray as xr

# Earth to Sun distance (m)
AU = 1.496e+11
# Mass of hydrogen (kg)
m_hydrogen = 1.6735575e-27
# m3 to cm3
m3_to_cm3 = (1./100)**3
# kg/m3 -> N/cm3
density_conversion = 1./m_hydrogen * m3_to_cm3
# m/s -> km/s
velocity_conversion = 1./1000.


class Enlil:
    """
    An Enlil model run read in from *filename*.
    """

    def __init__(self, filename):
        self.filename = filename
        # Open the netcdf file through xarray
        # 'wsa_enlil.latest.suball.nc'
        self.dataset = xr.open_dataset(filename)

        self.t0 = np.datetime64(self.dataset.attrs['REFDATE_CAL'])
        self.times = self.t0 + self.dataset['time'].values
        self.earth_times = self.t0 + self.dataset['Earth_TIME'].values

        # Radial position (AU)
        self.r = self.dataset['x_coord'].values/AU
        # latitude (radians) [Changing from colat -> lat]
        self.lat = np.pi/2-self.dataset['y_coord'].values
        # Longitude (radians) [Need to recenter the 0 point]
        self.lon = self.dataset['z_coord'].values-np.pi

    @property
    def ntimes(self):
        return len(self.times)

    @property
    def nr(self):
        return len(self.r)

    @property
    def nlat(self):
        return len(self.lat)

    @property
    def nlon(self):
        return len(self.lon)

    def _calibrate_parameter(self, param):
        """Calibrates the parameter *param* from the dataset.
        All parameters whose names begin with 'uncalibrated' are calibrated
        with the following formula (linear transform):
        param_calibrated=((param_uncalibrated-cal_min) *
                          (param_max-param_min)/cal_range) +
                          param_min
        where param_max and param_min are attributes associated with each
        uncalibrated parameter.
        """
        cal_min = self.dataset.attrs['cal_min']
        cal_range = self.dataset.attrs['cal_range']
        param_min = self.dataset[param].attrs[param[:4] + '_min']
        param_max = self.dataset[param].attrs[param[:4] + '_max']

        # Linear fit
        return ((self.dataset[param] - cal_min) *
                (param_max - param_min) / cal_range + param_min)

    def get_satellite_position(self, satellite, time):
        """
        Returns the position of the *satellite* (r, lat, lon)
        nearest to the requested *time*.
        """
        # Validate requested satellite
        if satellite not in ("Earth", "STEREO_A", "STEREO_B"):
            raise ValueError("Satellite not in dataset, must be one of: "
                             "EARTH, STEREO_A, STEREO_B")

        t = np.nonzero((time <= self.earth_times))[0][0]
        # radial coordinate (AU), colat -> lat, rotate longitude 180 degrees
        return (self.dataset[satellite + '_X1'][t]/AU,
                np.pi/2 - self.dataset[satellite + '_X2'][t],
                self.dataset[satellite + '_X3'][t] - np.pi)

    def get_satellite_data(self, satellite, param):
        """
        Returns the time-series of data from the requested *satellite*.
        *param* must be one of: velocity, density
        """
        # Validate requested satellite
        if satellite not in ("Earth", "STEREO_A", "STEREO_B"):
            raise ValueError("Satellite not in dataset, must be one of: "
                             "EARTH, STEREO_A, STEREO_B")

        # Validate parameter
        if param == 'velocity':
            data = self.dataset[satellite + '_V1']/1000.
        elif param == 'density':
            data = self.dataset[satellite + '_Density']*density_conversion
        else:
            raise ValueError("Parameter must be one of: velocity, density")

        return data

    def get_data(self, param, dim1, dim2, time=None):
        """
        Returns the data along the given dimensions.

        *param* must be one of: velocity, density, polarity
        *dim1* and *dim2* must be one of: r, lat, lon
        if *time* is None: data across all time will be returned in a
                           3D array of shape (ntime, dim1, dim2)
        if *time* is datetime compatible: then a 2D slice of data
                                          for that time will be returned
        """
        # Transform to coordinate numbers for dataset access
        dimension_trans = {"r": "1", "lat": "2", "lon": "3"}
        parameter_trans = {"density": "dd", "velocity": "vv", "polarity": "pp"}
        parameter_scaling = {"density": density_conversion*self.r**2,
                             "velocity": 1./1000,
                             "polarity": 1.}
        # Validate the dimensions
        if (dim1 not in dimension_trans or dim2 not in dimension_trans):
            raise ValueError("Dimensions must be one of ({})".format(
                set(dimension_trans.keys())))
        if param not in parameter_trans:
            raise ValueError("Parameter must be one of ({})".format(
                set(parameter_trans.keys())))

        # Found in the dataset as "dd12_3d" or similar
        full_param = (parameter_trans[param] +
                      dimension_trans[dim1] +
                      dimension_trans[dim2] +
                      "_3d")
        data = self._calibrate_parameter(full_param)*parameter_scaling[param]

        # Subset based on time if requested
        # TODO: Do we want to allow integers, or datetimes in here?
        if time is not None:
            if isinstance(time, np.datetime64):
                tindex = np.nonzero(time <= self.times)[0][0]
                data = data[tindex, :, :]
            else:
                raise ValueError("time must be a numpy datetime64 object")
        return data
