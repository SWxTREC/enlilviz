"""Main enlil data module."""
import functools


def _validate_satellite(func):
    """A decorator that checks to see if the satellite is in the dataset."""
    @functools.wraps(func)
    def wrapper(self, satellite, *args, **kwargs):
        # First argument has to be satellite
        if satellite not in self.ds['satellite']:
            raise IndexError("Satellite not in dataset, must be one of: "
                             "{}".format(self.ds['satellite'].values))

        return func(self, satellite, *args, **kwargs)
    return wrapper


class Enlil:
    """An Enlil model run.

    This is a class for storing the 2D slices and satellite data
    contained within the post-processed files. There are extra methods
    added to make working with the data easier.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset read in from Enlil netcdf output files.
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
        """Returns the position of the satellite.

        Parameters
        ----------
        satellite : str
            Satellite of interest.
        time : datetime-like
            Time of interest. Looks for nearest time.

        Returns
        -------
        r : float
            Radial position of satellite (AU)
        lat : float
            Latitudinal position of the satellite (deg).
        lon : float
            Longitudinal position of the satellite (deg).
        """

        ds = self.ds.sel({'earth_t': time}, method='nearest')
        ds = ds.loc[{'satellite': satellite}]
        return (ds['pos_r'], ds['pos_lat'], ds['pos_lon'])

    @_validate_satellite
    def get_satellite_data(self, satellite, var, coord=None):
        """Get the time series data from the requested satellite.

        Parameters
        ----------
        satellite : str
            Satellite of interest.
        var : str
            Variable of interest.
        coord : str, optional
            Coordinate of interest if the variable is a vector quantity.
            (r, lon, lat)

        Returns
        -------
        xarray.DataArray
            Time series of data.
        """

        varname = var
        if coord is not None:
            varname += '_' + coord
        return self.ds.loc[{'satellite': satellite}][varname]

    def get_slice(self, var, slice_plane, time=None):
        """Get a 2D slice of data.

        Parameters
        ----------
        var : str
            Variable of interest.
        slice_plane : str
            Slicing plane of the data (r, lat, lon).
            Note that a slice in lon, will return data with r/lat coordinates.
        time : datetime-like, optional
            Time of interest. If left out, all times will be returned.

        Returns
        -------
        xarray.DataArray
            Data sliced along a plane.
        """
        varname = 'slice_{}_{}'.format(var, slice_plane)
        da = self.ds[varname]
        if time is not None:
            da = da.sel({'t': time}, method='nearest')

        return da


class Evolution:
    """A temporal `Evolution` Enlil output file.

    This is a class for storing and accessing the temporal data from
    specific satellite/object locations requested during the Enlil model run.
    There are extra methods added to make working with the data easier.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset read in from Enlil netcdf output files.
    """

    def __init__(self, ds):
        self.ds = ds

    @property
    def times(self):
        """Times within the dataset.

        There are no slices in Evolution files, so this
        method simply returns all satellite times.
        """
        return self.earth_times

    @property
    def earth_times(self):
        """Satellite times for time series plots."""
        return self.ds['earth_t'].values

    def get_position(self, time):
        """Returns the position of the satellite.

        Parameters
        ----------
        time : datetime-like
            Time of interest. Looks for nearest time.

        Returns
        -------
        r : float
            Radial position of satellite (AU)
        lat : float
            Latitudinal position of the satellite (deg).
        lon : float
            Longitudinal position of the satellite (deg).
        """

        ds = self.ds.sel({'earth_t': time}, method='nearest')
        return (ds['pos_r'], ds['pos_lat'], ds['pos_lon'])

    def get_satellite_data(self, satellite, var, coord=None):
        """Get the time series data from the requested satellite.

        Parameters
        ----------
        satellite : str
            Satellite of interest.
        var : str
            Variable of interest.
        coord : str, optional
            Coordinate of interest if the variable is a vector quantity.
            (r, lon, lat)

        Returns
        -------
        xarray.DataArray
            Time series of data.
        """
        # TODO: Remove satellite from this? Right now, it makes it simpler
        #       to keep the plot calls consistent with TimeSeries
        varname = var
        if coord is not None:
            varname += '_' + coord
        return self.ds[varname]
