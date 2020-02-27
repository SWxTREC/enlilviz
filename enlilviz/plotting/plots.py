"""Generic plot wrappers for Enlil objects."""
from datetime import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

mpl.style.use(['dark_background'])

fontsize = 18
mpl.rcParams.update({'font.size': fontsize,
                     'xtick.labelsize': fontsize,
                     'ytick.labelsize': fontsize,
                     'axes.labelsize': fontsize,
                     'axes.titlesize': fontsize})

# sample the colormaps. Use 128 from each so we get 256 colors in total
_cm1 = mpl.cm.get_cmap('gist_ncar')(np.linspace(0.05, 1, 128))
# It would be better to use a nicer colormap if possible...
# cm1 = mpl.cm.get_cmap('plasma')(np.linspace(0., 1, 128))
_cm2 = mpl.cm.get_cmap('Greys')(np.linspace(0., 1, 128))

# combine them and build a new colormap
ENLIL_CMAP = LinearSegmentedColormap.from_list('enlil_cmap',
                                               np.vstack((_cm1, _cm2)))
DEN_CMAP = ENLIL_CMAP
VEL_CMAP = ENLIL_CMAP
# Colormap normalizations for density and velocity
DEN_NORM = mpl.colors.Normalize(vmin=0, vmax=60)
VEL_NORM = mpl.colors.Normalize(vmin=200, vmax=1600)

CMAP_LOOKUP = {'den': (DEN_CMAP, DEN_NORM),
               'vel': (VEL_CMAP, VEL_NORM)}

SAT_COLORS = {'Earth': 'tab:green',
              'STEREO_A': 'tab:red',
              'STEREO_B': 'tab:blue'}

RMIN, RMAX = 0, 1.8
LATMIN, LATMAX = -60, 60


class _BasePlot:
    """A base plotting class for access to generic properties.

    Parameters
    ----------
    enlil_run : enlil.Enlil
        An Enlil model run
    ax : axis, optional
        An axis to create the plot in. If left out, a proper default
        axis will be created for the plot.

    Attributes
    ----------
    enlil_run : enlil.Enlil
        The Enlil model run associated with this plot.
    ax : axis
        The axis for this plot.
    plot_data : dict
        All of the plot objects that can be updated, such as
        lines, markers, and mesh collections.
    """

    def __init__(self, enlil_run, ax=None):
        self.enlil_run = enlil_run
        # Store the current time index
        self._index = 0

        # Dictionary to store plot variables
        self.plot_data = {}

        self.ax = plt.subplot() if ax is None else ax

        # Initialize all of the plots
        self._init_plot()

    @property
    def time(self):
        """Current time of the plot."""
        return self.enlil_run.times[self._index]

    def set_time(self, time):
        """Sets the current time of the plot and updates the data.

        Parameters
        ----------
        time : datetime-like
            The time to set the current plot to.
        """
        da = self.enlil_run.times
        # Find the nearest time in the dataset, then find the
        # index where that occurs
        index = (da == da.sel({'t': time}, method='nearest')).values.argmax()
        self.set_index(index)

    def set_index(self, index):
        """Sets the current index of the plot and updates the data.

        Parameters
        ----------
        index : int
            The index to set the plot to.

        Raises
        ------
        IndexError
            If the index is outside of the valid range of the
            length of times available in the Enlil object.
        """
        if index > len(self.enlil_run.times) or index < 0:
            raise IndexError("The index is outside of the valid range.")
        # Only update if the index is changed
        if index != self._index:
            self._index = index
            self.update()

    def __next__(self):
        """Step ahead to the next time in the plot."""
        self._index += 1
        if self._index < len(self.enlil_run.times):
            self.update()
            return self
        # We have reached the end of the available times
        raise StopIteration

    def __iter__(self):
        """Iterate over all of the times available."""
        self._index = 0
        return self

    def _init_plot(self):
        """Initialize the axis with all of the plot data."""
        raise NotImplementedError

    def update(self):
        """Update all variable quantities within the plot."""
        raise NotImplementedError

    def _get_polarity_data(self, slice_plane):
        """
        Helper function to get the inner and outer boundary
        polarity data for the given slicing plane.

        Parameters
        ----------
        slice_plane : str
            The slicing plane one of: lon, lat

        Returns
        -------
        masked_array
            The inner boundary of polarity with negative values masked.
        masked_array
            The outer boundary of polarity with negative values masked.
        """
        run = self.enlil_run
        r = run.r
        # Get the polarity data
        inner_pol = run.get_slice('pol', slice_plane, self.time).sel(
            {'r': r[0]})
        inner_pol = np.ma.masked_where(inner_pol <= 0, np.ones_like(inner_pol))
        inner_pol *= r[0] - 0.025

        outer_pol = run.get_slice('pol', slice_plane, self.time).sel(
            {'r': r[-1]})
        outer_pol = np.ma.masked_where(outer_pol <= 0, np.ones_like(outer_pol))
        outer_pol *= r[-1] + 0.04

        return (inner_pol, outer_pol)


class LatitudeSlice(_BasePlot):
    """
    A latitude polar slice plot, which is in longitude-radial coordinates.

    Parameters
    ----------
    var : str
        Variable to plot (den, vel)
    """

    def __init__(self, enlil_run, var, ax=None):
        self.var = var
        self.ax = plt.subplot(projection='polar') if ax is None else ax
        super().__init__(enlil_run, self.ax)

    def _init_plot(self):
        """Initialize the axis with all of the plot data."""
        run = self.enlil_run
        r = run.r
        lon = np.deg2rad(run.lon)

        # Make the longitude-radial mesh
        r_lon, lon_r = _mesh_grid(r, lon)

        # Get the polarity data
        inner_pol, outer_pol = self._get_polarity_data('lat')

        ax = self.ax
        ax.axis('off')

        cmap, norm = CMAP_LOOKUP[self.var]
        data = run.get_slice(self.var, 'lat', self.time)
        mesh = ax.pcolormesh(lon_r, r_lon, data,
                             cmap=cmap, norm=norm,
                             shading='flat')
        self.plot_data['mesh'] = mesh

        # Polarity lines
        arc_min, = ax.plot(lon, inner_pol, c='tab:orange', lw=2)
        arc_max, = ax.plot(lon, outer_pol, c='tab:orange', lw=2)
        self.plot_data['pol_min'] = arc_min
        self.plot_data['pol_max'] = arc_max

        # Planets and satellites
        ax.plot(0, 0, 'o', color='gold', markersize=10, zorder=2)
        # Coodinates to polar plot are: lon, r
        for sat in ['Earth', 'STEREO_A', 'STEREO_B']:
            pos = run.get_satellite_position(sat, self.time)
            marker, = ax.plot(np.deg2rad(pos[2]), pos[0], 'o',
                              color=SAT_COLORS[sat],
                              markeredgecolor='k', markersize=10, zorder=2)
            self.plot_data[sat] = marker

        # Circle on Earth (1AU)
        # ax.axvline(0., c='k', zorder=1)
        circle_points = np.linspace(0, 2*np.pi)
        ax.plot(circle_points, np.ones(len(circle_points)), c='k',
                zorder=1, lw=1)
        theta_ticks = np.linspace(-np.pi/2, 0, 10)
        rs = (0.975, 1.025)
        for theta in theta_ticks:
            ax.plot([theta, theta], rs, c='k', zorder=1, lw=1)
        # Solid line along theta = 0
        ax.plot([0, 0], [0, 2], c='k', zorder=1, lw=1)
        # A bottom tick (marker=3) every 0.1 out to 1.7 AU
        rticks = np.arange(17)/10
        for r in rticks:
            ax.plot(0, r, marker=3, c='k', zorder=1)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(RMIN, RMAX)

    def update(self):
        """Update all variable quantities within the plot."""
        run = self.enlil_run
        lon = np.deg2rad(run.lon)
        data = run.get_slice(self.var, 'lat', self.time)
        self.plot_data['mesh'].set_array(data.values.flatten())

        inner_pol, outer_pol = self._get_polarity_data('lat')
        self.plot_data['pol_min'].set_data(lon, inner_pol)
        self.plot_data['pol_max'].set_data(lon, outer_pol)

        for sat in ['Earth', 'STEREO_A', 'STEREO_B']:
            pos = run.get_satellite_position(sat, self.time)
            self.plot_data[sat].set_data(np.deg2rad(pos[2]), pos[0])


class LongitudeSlice(_BasePlot):
    """A longitude polar slice plot, which is in latitude-radial coordinates.

    Parameters
    ----------
    var : str
        Variable to plot (den, vel)
    """

    def __init__(self, enlil_run, var, ax=None):
        self.var = var
        self.ax = plt.subplot(projection='polar') if ax is None else ax
        super().__init__(enlil_run, self.ax)

    def _init_plot(self):
        """Initialize the axis with all of the plot data."""
        run = self.enlil_run
        r = run.r
        lat = np.deg2rad(run.lat)

        # Make the longitude-radial mesh
        r_lat, lat_r = _mesh_grid(r, lat)

        # Get the polarity data
        inner_pol, outer_pol = self._get_polarity_data('lon')

        ax = self.ax
        ax.axis('off')

        cmap, norm = CMAP_LOOKUP[self.var]
        data = run.get_slice(self.var, 'lon', self.time)
        mesh = ax.pcolormesh(lat_r, r_lat, data,
                             cmap=cmap, norm=norm, shading='flat')
        self.plot_data['mesh'] = mesh

        # Polarity lines
        arc_min, = ax.plot(lat, inner_pol, c='tab:orange', lw=2)
        arc_max, = ax.plot(lat, outer_pol, c='tab:orange', lw=2)
        self.plot_data['pol_min'] = arc_min
        self.plot_data['pol_max'] = arc_max

        # Planets and satellites
        ax.plot(0, 0, 'o', color='gold', markersize=10, zorder=2)
        # Coodinates to polar plot are: lat, r
        for sat in ['Earth']:
            pos = run.get_satellite_position(sat, self.time)
            marker, = ax.plot(np.deg2rad(pos[1]), pos[0], 'o',
                              color=SAT_COLORS[sat],
                              markeredgecolor='k', markersize=10, zorder=2)
            self.plot_data[sat] = marker

        # x == theta, y == r
        circle_points = np.linspace(0, 2*np.pi)
        ax.plot(circle_points, np.ones(len(circle_points)), c='k',
                zorder=1, lw=1)
        theta_ticks = np.linspace(-np.pi/2, 0, 10)
        rs = (0.975, 1.025)
        for theta in theta_ticks:
            ax.plot([theta, theta], rs, c='k', zorder=1, lw=1)
        # Solid line along theta = 0
        ax.plot([0, 0], [0, 2], c='k', zorder=1, lw=1)
        # A bottom tick (marker=3) every 0.1 out to 1.7 AU
        rticks = np.arange(17)/10
        for r in rticks:
            ax.plot(0, r, marker=3, c='k', zorder=1)
        ax.set_xlim(np.deg2rad([LATMIN, LATMAX]))
        ax.set_ylim(RMIN, RMAX)
        ax.set_yticks([])

    def update(self):
        """Update all variable quantities within the plot."""
        run = self.enlil_run
        lat = np.deg2rad(run.lat)
        data = run.get_slice(self.var, 'lon', self.time)
        self.plot_data['mesh'].set_array(data.values.flatten())

        inner_pol, outer_pol = self._get_polarity_data('lon')
        self.plot_data['pol_min'].set_data(lat, inner_pol)
        self.plot_data['pol_max'].set_data(lat, outer_pol)

        for sat in ['Earth']:
            pos = run.get_satellite_position(sat, self.time)
            self.plot_data[sat].set_data(np.deg2rad(pos[1]), pos[0])


class RadialSlice(_BasePlot):
    """
    A radial slice plot, which is in longitude-latitude coordinates.

    Parameters
    ----------
    var : str
        Variable to plot (den, vel)
    """

    def __init__(self, enlil_run, var, ax=None):
        self.var = var
        self.ax = plt.subplot(projection='mollweide') if ax is None else ax
        super().__init__(enlil_run, self.ax)

    def _init_plot(self):
        """Initialize the axis with all of the plot data."""
        run = self.enlil_run
        lon = np.deg2rad(run.lon)
        lat = np.deg2rad(run.lat)

        # Make the longitude-latitude mesh
        lon_lat, lat_lon = _mesh_grid(lon, lat)

        ax = self.ax
        ax.axis('off')

        cmap, norm = CMAP_LOOKUP[self.var]
        # Need to transpose the data for plotting
        data = run.get_slice(self.var, 'r', self.time).T
        mesh = ax.pcolormesh(lon_lat, lat_lon, data,
                             cmap=cmap, norm=norm,
                             shading='flat')
        self.plot_data['mesh'] = mesh

        # Planets and satellites
        # Coodinates to polar plot are: lon, lat
        for sat in ['Earth', 'STEREO_A', 'STEREO_B']:
            pos = run.get_satellite_position(sat, self.time)
            marker, = ax.plot(np.deg2rad(pos[2]), np.deg2rad(pos[1]), 'o',
                              color=SAT_COLORS[sat],
                              markeredgecolor='k', markersize=10, zorder=2)
            self.plot_data[sat] = marker

        # Specified in degree spacing
        ax.set_longitude_grid(30)
        ax.set_latitude_grid(15)
        # Remove longitude labels
        ax.xaxis.set_ticklabels([])
        ax.grid(True)
        ax.axis('on')

    def update(self):
        """Update all variable quantities within the plot."""
        run = self.enlil_run
        # transpose the data to match the coordinates
        data = run.get_slice(self.var, 'r', self.time).T
        self.plot_data['mesh'].set_array(data.values.flatten())

        for sat in ['Earth', 'STEREO_A', 'STEREO_B']:
            pos = run.get_satellite_position(sat, self.time)
            self.plot_data[sat].set_data(np.deg2rad(pos[2]),
                                         np.deg2rad(pos[1]))


class TimeSeries(_BasePlot):
    """A time series plot of satellite data.

    Parameters
    ----------
    sat : str
        Satellite of interest (Earth, STEREO_A, STEREO_B)
    var : str
        Variable to plot (den, vel)
    """
    def __init__(self, enlil_run, sat, var, coord=None, ax=None):
        self.sat = sat
        self.var = var
        self.coord = coord
        self.ax = plt.subplot() if ax is None else ax
        super().__init__(enlil_run, self.ax)

    def _init_plot(self):
        """Initialize the axis with all of the plot data."""
        run = self.enlil_run
        times = run.earth_times

        data = run.get_satellite_data(self.sat, self.var, self.coord).values
        ax = self.ax

        # Label the plot in axes coordinates
        ax.text(0.025, 0.75, self.sat.replace('_', ' '),
                transform=ax.transAxes, zorder=1)

        ax.plot(times, data, c=SAT_COLORS[self.sat], lw=2)
        time = self.time.astype(datetime)
        self.plot_data['timeline'] = ax.axvline(time, c='y',
                                                linewidth=3, zorder=5)

        if self.sat == 'Earth':
            if self.var == 'den':
                ax.set_title('Plasma Density (/cm$^3$)')
            elif self.var == 'vel':
                ax.set_title('Radial Velocity (km/s)')

        ax.set_xlim(run.times[0], run.times[-1])
        ax.xaxis.set_major_locator(mpl.dates.DayLocator())
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d"))
        ax.tick_params(axis='both', which='major', direction='in',
                       length=6, width=1,
                       top=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in',
                       length=3, width=1,
                       top=True)
        ax.xaxis.set_minor_locator(mpl.dates.HourLocator(interval=2))

        if self.sat != 'STEREO_B':
            plt.setp(ax.get_xticklabels(), visible=False)

    def update(self):
        """Update all variable quantities within the plot."""
        line = self.plot_data['timeline']
        line.set_xdata(self.time)


class Title(_BasePlot):
    """A title axis with the current time."""

    def __init__(self, enlil_run, ax=None):
        self.ax = plt.subplot() if ax is None else ax
        super().__init__(enlil_run, self.ax)

    def _init_plot(self):
        """Initialize the axis with all of the plot data."""
        ax = self.ax
        # Turn all borders and spines off
        ax.axis('off')
        self.plot_data['title'] = ax.text(0.5, 0.5,
                                          self._title_string(),
                                          fontsize=36, color='gold',
                                          horizontalalignment='center',
                                          verticalalignment='center')

    def update(self):
        """Update all variable quantities within the plot."""
        self.plot_data['title'].set_text(self._title_string())

    def _title_string(self):
        """String representation of the time for titles."""
        t = str(self.time)[:16].replace('T', ' ')
        t = t[:-2] + "00:00"
        return t


class Colorbar:
    """Colorbar axis."""
    def __init__(self, var, ax=None):
        self.var = var
        self.ax = plt.subplot() if ax is None else ax

        cmap, norm = CMAP_LOOKUP[var]
        label_dict = {'den': 'Plasma Density (r$^2$N/cm$^3$)',
                      'vel': 'Radial Velocity (km/s)'}
        tick_dict = {'den': (0, 15, 30, 45, 60),
                     'vel': (200, 550, 900, 1250, 1600)}
        self.cb = mpl.colorbar.ColorbarBase(self.ax, cmap=cmap, norm=norm,
                                            orientation='vertical',
                                            label=label_dict[var],
                                            ticks=tick_dict[var],)
        self.ax.yaxis.set_label_position('left')


def _mesh_grid(x, y):
    """A helper function to extrapolate/center the meshgrid coordiantes.

    Matplotlib's pcolormesh currently needs data specified at edges
    and drops the last column of the data, unfortunately. This function
    borrows from matplotlib PR #16258, which will automatically extend
    the grids in the future (Likely MPL 3.3+).
    """
    def _interp_grid(X):
        # helper for below
        if np.shape(X)[1] > 1:
            dX = np.diff(X, axis=1)/2.
            X = np.hstack((X[:, [0]] - dX[:, [0]],
                           X[:, :-1] + dX,
                           X[:, [-1]] + dX[:, [-1]]))
        else:
            # This is just degenerate, but we can't reliably guess
            # a dX if there is just one value.
            X = np.hstack((X, X))
        return X

    X, Y = np.meshgrid(x, y)
    # extend rows
    X = _interp_grid(X)
    Y = _interp_grid(Y)
    # extend columns
    X = _interp_grid(X.T).T
    Y = _interp_grid(Y.T).T
    return X, Y
