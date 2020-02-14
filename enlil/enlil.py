"""Main enlil data module."""
import functools


def _validate_satellite(func):
    """A decorator that checks the satellite in the dataset."""
    @functools.wraps(func)
    def wrapper(self, satellite, *args, **kwargs):
        # First argument has to be satellite
        if satellite not in self.ds['satellite']:
            raise IndexError("Satellite not in dataset, must be one of: "
                             "{}".format(self.ds['satellite'].values))

        return func(self, satellite, *args, **kwargs)
    return wrapper


class Enlil:
    """
    An Enlil model run initialized from an :xarray.Dataset: *ds*.

    It adds helper methods to make getting data easier
    """

    def __init__(self, ds):
        self.ds = ds

    @property
    def times(self):
        """Data times for slice plots."""
        return self.ds['t'].values

    @property
    def earth_times(self):
        """Satellite times for time series plots."""
        return self.ds['earth_t'].values

    @property
    def r(self):
        """Radial coordinate (AU)."""
        return self.ds['r'].values

    @property
    def lon(self):
        """Longitudinal coordinate (deg)."""
        return self.ds['lon'].values

    @property
    def lat(self):
        """Latitudinal coordinate (deg)."""
        return self.ds['lat'].values

    @_validate_satellite
    def get_satellite_position(self, satellite, time):
        """
        Returns the position of the *satellite* (r, lat, lon)
        nearest to the requested *time*.
        """

        ds = self.ds.sel({'earth_t': time}, method='nearest')
        ds = ds.loc[{'satellite': satellite}]
        return (ds['pos_r'], ds['pos_lat'], ds['pos_lon'])

    @_validate_satellite
    def get_satellite_data(self, satellite, var, coord=None):
        """
        Returns the time-series of data from the requested *satellite*.
        *var* : A variable in the dataset
        *coord* : The coordinate direction of the variable (if it is a vector)
        """

        varname = var
        if coord is not None:
            varname += '_' + coord
        return self.ds.loc[{'satellite': satellite}][varname]

    def get_slice(self, var, plane, time=None):
        """
        Returns a spatial slice of the data.

        *var* : Requested : velocity, density, polarity
        *plane* : Slicing plane (r, lat, lon)
        *time* : The time to slice by, if None a 3d array
                 will be returned (time, dim1, dim2)
        """
        varname = 'slice_{}_{}'.format(var, plane)
        da = self.ds[varname]
        if time is not None:
            da = da.sel({'t': time}, method='nearest')

        return da
