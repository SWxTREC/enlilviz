"""Tests for `enlilviz` package."""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_properties(enlil_run):
    """Testing the properties of the Enlil object."""
    expected = np.arange(np.datetime64("2010-01-01"),
                         np.datetime64("2010-01-05"))
    assert_array_equal(enlil_run.times, expected)

    expected = np.arange(np.datetime64("2010-01-01"),
                         np.datetime64("2010-01-10"))
    assert_array_equal(enlil_run.earth_times, expected)


def test_get_position(enlil_run):
    """Testing the get position function."""
    pos1 = enlil_run.get_satellite_position('Earth', enlil_run.times[0])
    assert_array_almost_equal(pos1, (0.1, -45, -80))
    # Test in the middle
    pos2 = enlil_run.get_satellite_position('Earth', enlil_run.earth_times[2])
    assert_array_almost_equal(pos2, (0.5, -22.5, -30))
    # Test the end
    pos3 = enlil_run.get_satellite_position('Earth', enlil_run.earth_times[-1])
    assert_array_almost_equal(pos3, (1.7, 45, 120))


def test_get_data_scalar(enlil_run):
    """Testing the get_satellite_data scalar function."""
    data = enlil_run.get_satellite_data('Earth', 'den')
    ndata = len(enlil_run.earth_times)
    # Density range: 0.1 - 60, functional form: x
    expected = np.arange(ndata)
    expected = expected/np.max(expected) * (60-0.1) + 0.1

    assert_array_almost_equal(data, expected)


def test_get_data_vector(enlil_run):
    """Testing the get_satellite_data vector function.

    Testing with two different coordinates.
    """
    data = enlil_run.get_satellite_data('Earth', 'vel', coord='r')
    ndata = len(enlil_run.earth_times)
    # Velocity range: 300, 1200, functional form: x
    expected = np.arange(ndata)
    expected = expected/np.max(expected) * (1200 - 300) + 300
    assert_array_almost_equal(data, expected)

    data = enlil_run.get_satellite_data('Earth', 'vel', coord='lon')
    assert_array_almost_equal(data, expected)

    data = enlil_run.get_satellite_data('Earth', 'vel', coord='lat')
    assert_array_almost_equal(data, expected)


def test_get_slice(enlil_run):
    """Testing the get_slice function."""
    # Shape of data should be (5, 10, 15) and 4 times
    r = np.linspace(0.1, 1.7, 5)
    lat = np.linspace(-59, 59, 10)
    lon = np.linspace(-179, 179, 15)
    x = np.arange(len(r))
    y = np.sin(np.linspace(0, np.pi, len(lat)))
    z = np.arange(len(lon))**2

    def scale_variable(x):
        """Scale the variable to the proper velocity range."""
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = 300, 1200

        return (x - xmin) / (xmax - xmin) * (ymax - ymin) + ymin

    # Get the data for the first point in time
    data = enlil_run.get_slice('vel', 'r', time=enlil_run.times[0])
    # Should return lat, lon
    expected = scale_variable(z[:, np.newaxis] + y[np.newaxis, :])
    assert_array_almost_equal(data, expected)

    data = enlil_run.get_slice('vel', 'lat', time=enlil_run.times[0])
    # Should return lat, lon
    expected = scale_variable(z[:, np.newaxis] + x[np.newaxis, :])
    assert_array_almost_equal(data, expected)

    data = enlil_run.get_slice('vel', 'lon', time=enlil_run.times[0])
    # Should return lat, lon
    expected = scale_variable(y[:, np.newaxis] + x[np.newaxis, :])
    assert_array_almost_equal(data, expected)

    # Test without any time variable to get a 3D slice
    data = enlil_run.get_slice('vel', 'r')
    # Should return lat, lon
    expected = scale_variable(z[:, np.newaxis] + y[np.newaxis, :])
    # Expand along the time dimension
    expected = np.tile(expected, (4, 1, 1))
    assert_array_almost_equal(data, expected)
