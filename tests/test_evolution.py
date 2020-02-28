"""Tests for evolution class."""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_properties(evo):
    """Testing the properties of the Enlil object."""
    assert len(evo.times) == 20

    expected = np.arange(np.datetime64("2010-01-01"),
                         np.datetime64("2010-01-21"))
    assert_array_equal(evo.times, expected)
    assert_array_equal(evo.earth_times, expected)


def test_get_position(evo):
    """Testing the get position function."""
    pos1 = evo.get_position(evo.times[0])
    assert_array_almost_equal(pos1, (0.1, 30, -150))
    # Test in the middle
    pos2 = evo.get_position(evo.times[10])
    assert_array_almost_equal(pos2, (0.942105,   -1.578947, -123.684211))
    # Test the end
    pos1 = evo.get_position(evo.times[-1])
    assert_array_almost_equal(pos1, (1.7, -30, -100))


def test_get_data_scalar(evo):
    """Testing the get_satellite_data scalar function."""
    data = evo.get_satellite_data('earth', 'den')
    ndata = len(evo.times)
    # Density range: 0.1 - 60, functional form: x**2
    expected = np.arange(ndata)**2
    expected = expected/np.max(expected) * (60-0.1) + 0.1

    assert_array_almost_equal(data, expected)


def test_get_data_vector(evo):
    """Testing the get_satellite_data vector function.

    Testing with two different coordinates.
    """
    data = evo.get_satellite_data('earth', 'vel', coord='r')
    ndata = len(evo.times)
    # Velocity range: 300, 1200, functional form: x
    expected = np.arange(ndata)
    expected = expected/np.max(expected) * (1200 - 300) + 300
    assert_array_almost_equal(data, expected)

    data = evo.get_satellite_data('earth', 'vel', coord='lon')
    # Velocity range: 300, 1200, functional form: x**2
    expected = np.arange(ndata)**2
    expected = expected/np.max(expected) * (1200 - 300) + 300
    assert_array_almost_equal(data, expected)
