"""Default forecast center figures."""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Lots of extra imports for polar plotting to look nice
from mpl_toolkits.axisartist import (Subplot, SubplotHost,
                                     ParasiteAxesAuxTrans, floating_axes,
                                     angle_helper)
from mpl_toolkits.axisartist.grid_finder import FixedLocator, MaxNLocator, DictFormatter
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D

import numpy as np

mpl.style.use(['dark_background'])
enlil_style = {
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "font.size": 18,
    "figure.figsize": (20, 12.5),
    "figure.dpi": 72,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "lines.linewidth": 1,
    "lines.markersize": 6,
    "legend.fontsize": 18,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"]}

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

# PolarAxes.PolarTransform takes radians.
# However, we want our coordinate system in degrees
tr_polar_deg = Affine2D().scale(np.pi/180., 1.) + PolarAxes.PolarTransform()
# Here is just a radians transform
tr_polar_rad = PolarAxes.PolarTransform()

LAT_MIN, LAT_MAX = -60, 60  # degrees
R_MIN, R_MAX = 0.1, 1.7  # AU

grid_helper = floating_axes.GridHelperCurveLinear(
    tr_polar_rad, extremes=(np.deg2rad(LAT_MAX), np.deg2rad(LAT_MIN),
                            R_MAX, R_MIN),
    grid_locator1=FixedLocator([]),  # Remove ticks from polar plot
    grid_locator2=FixedLocator([]),
    tick_formatter1=None,
    tick_formatter2=None)


class ForecasterPlot:
    """
    Figure class for the main Forecaster plot.

    *enlil_run* : A :class:`enlil.Enlil` model run.
    """

    def __init__(self, enlil_run):
        self.enlil_run = enlil_run
        # Store the current time index
        self.index = 0

        # Figure and axis creation
        self.fig = plt.figure(figsize=(20, 12.5), dpi=72)
        # Dictionary to store axes
        self.axes = {}
        # Dictionary to store plot variables
        self.plot_data = {}

        self.init_axes()
        self.init_plots()

    def init_axes(self):
        """
        Initializing the axes and setting up the grid. This routine
        makes heavy use of gridspecs for the layout.
        """
        gs0 = gridspec.GridSpec(3, 1, height_ratios=(2, 10, 10), hspace=0.3)
        self.axes["title"] = plt.subplot(gs0[0])

        # Density row
        # -----------
        gs00 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs0[1],
                                                width_ratios=(1, 10, 5, 15))
        self.axes["den_colorbar"] = plt.subplot(gs00[0])
        self.axes["den_longitude"] = plt.subplot(gs00[1], projection='polar')
        # Using a normal polar subplot leaves too much space around the edges,
        # so for now we will adjust the white space with a floating subplot.
        # ax_den2 = plt.subplot(gs00[2], projection='polar')
        ax_den2box = floating_axes.FloatingSubplot(self.fig, gs00[2],
                                                   grid_helper=grid_helper)
        self.fig.add_subplot(ax_den2box)
        ax_den2box.axis('off')

        # create a parasite axes with transData in R/lat
        self.axes["den_latitude"] = ax_den2box.get_aux_axes(tr_polar_rad)
        # Updating the clip path of the parasite axis
        self.axes["den_latitude"].patch = ax_den2box.patch
        # Make it a smaller zorder
        self.axes["den_latitude"].patch.zorder = 0.9

        # Nested time series
        gs000 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs00[3])
        ax_den_earth = plt.subplot(gs000[0])
        self.axes["den_time_earth"] = ax_den_earth
        self.axes["den_time_stereo_a"] = plt.subplot(gs000[1],
                                                     sharex=ax_den_earth,
                                                     sharey=ax_den_earth)
        self.axes["den_time_stereo_b"] = plt.subplot(gs000[2],
                                                     sharex=ax_den_earth,
                                                     sharey=ax_den_earth)

        # Velocity row
        # ------------
        gs01 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs0[2],
                                                width_ratios=(1, 10, 5, 15))
        self.axes["vel_colorbar"] = plt.subplot(gs01[0])
        self.axes["vel_longitude"] = plt.subplot(gs01[1], projection='polar')
        # Using a normal polar subplot leaves too much space around the edges,
        # so for now we will adjust the white space with a floating subplot.
        # ax_vel2 = plt.subplot(gs01[2], projection='polar')
        ax_vel2box = floating_axes.FloatingSubplot(self.fig, gs01[2],
                                                   grid_helper=grid_helper)
        self.fig.add_subplot(ax_vel2box)

        # create a parasite axes with transData in R/lat
        self.axes["vel_latitude"] = ax_vel2box.get_aux_axes(tr_polar_rad)
        # Updating the clip path of the parasite axis
        self.axes["vel_latitude"].patch = ax_vel2box.patch
        # Make it a smaller zorder
        self.axes["vel_latitude"].patch.zorder = 0.9

        # Nested time series
        gs010 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs01[3])
        # Additionally share the x-axis with the density time series
        ax_vel_earth = plt.subplot(gs010[0], sharex=ax_den_earth)
        self.axes["vel_time_earth"] = ax_vel_earth
        self.axes["vel_time_stereo_a"] = plt.subplot(gs010[1],
                                                     sharex=ax_vel_earth,
                                                     sharey=ax_vel_earth)
        self.axes["vel_time_stereo_b"] = plt.subplot(gs010[2],
                                                     sharex=ax_vel_earth,
                                                     sharey=ax_vel_earth)

        # Add the colorbars right away since those won't ever be updated.
        self._add_colorbars()

    def _add_colorbars(self):
        """Add the colorbars to the figure."""
        den_mappable = mpl.cm.ScalarMappable(norm=DEN_NORM, cmap=DEN_CMAP)
        plt.colorbar(den_mappable, cax=self.axes["den_colorbar"],
                     orientation='vertical',
                     label='Plasma Density (r$^2$N/cm$^3$)')

        vel_mappable = mpl.cm.ScalarMappable(norm=VEL_NORM, cmap=VEL_CMAP)
        plt.colorbar(vel_mappable, cax=self.axes["vel_colorbar"],
                     orientation='vertical', label='Radial Velocity (km/s)')

        self.axes["den_colorbar"].yaxis.set_label_position('left')
        self.axes["vel_colorbar"].yaxis.set_label_position('left')

    def _init_lon_r_plots(self):
        """Add longitude-radial meshes to the axes."""
        run = self.enlil_run
        r = run.r
        lon = np.deg2rad(run.lon)
        t0 = run.times[0]

        # Make the longitude-radial mesh
        r_lon, lon_r = _mesh_grid(r, lon)

        # Get satellite positions
        earth_pos = run.get_satellite_position('Earth', t0)
        stereo_a_pos = run.get_satellite_position('STEREO_A', t0)
        stereo_b_pos = run.get_satellite_position('STEREO_B', t0)

        # Get the polarity data
        outer_pol = run.get_slice('pol', 'lat', t0).sel({'r': r[-1]})
        outer_pol = np.ma.masked_where(outer_pol <= 0, np.ones_like(outer_pol))
        outer_pol *= r[-1]
        inner_pol = run.get_slice('pol', 'lat', t0).sel({'r': r[0]})
        inner_pol = np.ma.masked_where(inner_pol <= 0, np.ones_like(inner_pol))
        inner_pol *= r[0] - 0.025

        # Density plot first
        # ------------------
        ax = self.axes["den_longitude"]
        ax.axis('off')

        data = run.get_slice('den', 'lat', t0) * r**2
        mesh = ax.pcolormesh(lon_r, r_lon, data,
                             cmap=DEN_CMAP, norm=DEN_NORM,
                             shading='flat')
        self.plot_data['den_lon_mesh'] = mesh

        # Polarity lines
        arc_min, = ax.plot(lon, inner_pol,
                           c='tab:orange', linewidth=3)
        arc_max, = ax.plot(lon, outer_pol,
                           c='tab:orange', linewidth=5)
        self.plot_data['den_lon_arc_min'] = arc_min
        self.plot_data['den_lon_arc_max'] = arc_max

        # Planets and satellites
        sun, = ax.plot(0, 0, 'o', color='gold', markersize=10, zorder=2)
        # Coodinates to polar plot are: lon, r
        earth, = ax.plot(np.deg2rad(earth_pos[2]), earth_pos[0], 'o', color='tab:green',
                         markeredgecolor='k', markersize=10, zorder=2)
        stereo_a, = ax.plot(np.deg2rad(stereo_a_pos[2]), stereo_a_pos[0], 'o',
                            color='tab:red', markeredgecolor='k',
                            markersize=10, zorder=2)
        stereo_b, = ax.plot(np.deg2rad(stereo_b_pos[2]), stereo_b_pos[0], 'o',
                            color='tab:blue', markeredgecolor='k',
                            markersize=10, zorder=2)
        self.plot_data['den_lon_sun'] = sun
        self.plot_data['den_lon_earth'] = earth
        self.plot_data['den_lon_stereo_a'] = stereo_a
        self.plot_data['den_lon_stereo_b'] = stereo_b

        # Circle on Earth (1AU)
        # ax.axvline(0., c='k', zorder=1)
        circle_points = np.linspace(0, 2*np.pi)
        ax.plot(circle_points, np.ones(len(circle_points)), c='k',
                zorder=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, np.max(r))

        # Velocity
        # --------
        ax = self.axes["vel_longitude"]
        ax.axis('off')
        data = run.get_slice('vel', 'lat', t0)
        mesh = ax.pcolormesh(lon_r, r_lon, data,
                             cmap=VEL_CMAP, norm=VEL_NORM, shading='flat')
        self.plot_data['vel_lon_mesh'] = mesh

        # Polarity lines
        arc_min, = ax.plot(lon, inner_pol, c='tab:orange',
                           linewidth=3, zorder=3)
        arc_max, = ax.plot(lon, outer_pol, c='tab:orange',
                           linewidth=5, zorder=3)
        self.plot_data['vel_lon_arc_min'] = arc_min
        self.plot_data['vel_lon_arc_max'] = arc_max

        # Planets and satellites
        sun, = ax.plot(0, 0, 'o', color='gold', markersize=10, zorder=2)
        # Coodinates to polar plot are: lon, r
        earth, = ax.plot(np.deg2rad(earth_pos[2]), earth_pos[0], 'o',
                         color='tab:green',
                         markeredgecolor='k', markersize=10, zorder=2)
        stereo_a, = ax.plot(np.deg2rad(stereo_a_pos[2]), stereo_a_pos[0], 'o',
                            color='tab:red', markeredgecolor='k',
                            markersize=10, zorder=2)
        stereo_b, = ax.plot(np.deg2rad(stereo_b_pos[2]), stereo_b_pos[0], 'o',
                            color='tab:blue', markeredgecolor='k',
                            markersize=10, zorder=2)
        self.plot_data['vel_lon_sun'] = sun
        self.plot_data['vel_lon_earth'] = earth
        self.plot_data['vel_lon_stereo_a'] = stereo_a
        self.plot_data['vel_lon_stereo_b'] = stereo_b

        # Circle on Earth (1AU)
        # ax.axvline(0., c='k', zorder=1)
        circle_points = np.linspace(0, 2*np.pi)
        ax.plot(circle_points, np.ones(len(circle_points)), c='k', zorder=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, np.max(r))

    def _init_lat_r_plots(self):
        """Add latitude-radial meshes to the axes."""
        run = self.enlil_run
        r = run.r
        lat = np.deg2rad(run.lat)
        t0 = run.times[0]

        # Make the longitude-radial mesh
        r_lat, lat_r = _mesh_grid(r, lat)

        # Get satellite positions
        # We don't need STEREO spacecraft for this slicing plane
        earth_pos = run.get_satellite_position('Earth', t0)

        # Get the polarity data
        outer_pol = run.get_slice('pol', 'lon', t0).sel({'r': r[-1]})
        outer_pol = np.ma.masked_where(outer_pol <= 0, np.ones_like(outer_pol))
        outer_pol *= r[-1]
        inner_pol = run.get_slice('pol', 'lon', t0).sel({'r': r[0]})
        inner_pol = np.ma.masked_where(inner_pol <= 0, np.ones_like(inner_pol))
        inner_pol *= r[0] - 0.025

        # Density latitude mesh
        # ---------------------
        ax = self.axes['den_latitude']
        ax.axis('off')

        data = run.get_slice('den', 'lon', t0) * r**2
        mesh = ax.pcolormesh(lat_r, r_lat, data,
                             cmap=DEN_CMAP, norm=DEN_NORM, shading='flat')
        self.plot_data['den_lat_mesh'] = mesh

        # Polarity lines
        arc_min, = ax.plot(lat, inner_pol, c='tab:orange',
                           linewidth=3, zorder=3)
        arc_max, = ax.plot(lat, outer_pol, c='tab:orange',
                           linewidth=5, zorder=3)
        self.plot_data['den_lat_arc_min'] = arc_min
        self.plot_data['den_lat_arc_max'] = arc_max

        # Planets and satellites
        # sun, = ax.plot(0, 0, 'o', color='gold', markersize=10, zorder=2)
        # Coodinates to polar plot are: lat, r
        earth, = ax.plot(np.deg2rad(earth_pos[1]), earth_pos[0], 'o',
                         color='tab:green',
                         markeredgecolor='k', markersize=10, zorder=2)
        # self.plot_data['den_lat_sun'] = sun
        self.plot_data['den_lat_earth'] = earth

        # x == theta, y == r
        ax.set_xlim(np.min(lat), np.max(lat))
        ax.set_ylim(0, np.max(r))
        ax.set_yticks([])

        # Velocity latitude mesh
        # ----------------------
        ax = self.axes['vel_latitude']
        ax.axis('off')
        data = run.get_slice('vel', 'lon', t0)
        mesh = ax.pcolormesh(lat_r, r_lat, data,
                             cmap=VEL_CMAP, norm=VEL_NORM, shading='flat')
        self.plot_data['vel_lat_mesh'] = mesh

        # Polarity lines
        arc_min, = ax.plot(lat, inner_pol, c='tab:orange', linewidth=3)
        arc_max, = ax.plot(lat, outer_pol, c='tab:orange', linewidth=5)
        self.plot_data['vel_lat_arc_min'] = arc_min
        self.plot_data['vel_lat_arc_max'] = arc_max

        # Planets and satellites
        # sun, = ax.plot(0, 0, 'o', color='gold', markersize=10, zorder=2)
        # Coodinates to polar plot are: lat, r
        earth, = ax.plot(np.deg2rad(earth_pos[1]), earth_pos[0], 'o',
                         color='tab:green',
                         markeredgecolor='k', markersize=10, zorder=2)
        # self.plot_data['vel_lat_sun'] = sun
        self.plot_data['vel_lat_earth'] = earth

        # x == theta, y == r
        ax.set_xlim(np.min(lat), np.max(lat))
        ax.set_ylim(0, np.max(r))
        ax.set_yticks([])

    def init_plots(self):

        self._init_lon_r_plots()
        self._init_lat_r_plots()

        return


        # ----------------------
        # Velocity latitude mesh
        # ----------------------
        ax = self.axes['vel_latitude']
        ax.axis('off')
        mesh = ax.pcolormesh(lat_r, r_lat, np.ones(lat_r.shape),
                             cmap=VEL_CMAP, norm=VEL_NORM, shading='gouraud')
        self.plot_data['vel_lat_mesh'] = mesh

        # Polarity lines
        arc_min, = ax.plot(lat, r[0]*np.ones(len(lat)), c='tab:orange', linewidth=3)
        arc_max, = ax.plot(lat, r[-1]*np.ones(len(lat)), c='tab:orange', linewidth=5)
        self.plot_data['vel_lat_arc_min'] = arc_min
        self.plot_data['vel_lat_arc_max'] = arc_max

        # Planets and satellites
        sun = ax.scatter(0, 0, color='white', s=100, zorder=2)
        earth = ax.scatter(0, 0, color='tab:green', edgecolor='k', s=100, zorder=2)
        # For the xy-plane this would only make sense if longitude of Earth == longitude of STEREO
        # ax.scatter(0, 0, color='tab:red', edgecolor='k', s=100, zorder=2)
        # ax.scatter(0, 0, color='tab:blue', edgecolor='k', s=100, zorder=2)
        self.plot_data['vel_lat_sun'] = sun
        self.plot_data['vel_lat_earth'] = earth

        # x == theta, y == r
        ax.set_xlim(np.min(lat), np.max(lat))
        ax.set_ylim(0, np.max(r))
        ax.set_yticks([])

        # -----------------
        # Time series plots
        # -----------------
        # Called in a separate function to plot the static time series with data
        self.init_time_series()

        # Now make the vertical lines indicating the current time
        t0 = self.time
        ax_earth = self.axes['den_time_earth']
        ax_stereo_a = self.axes['den_time_stereo_a']
        ax_stereo_b = self.axes['den_time_stereo_b']
        self.plot_data['den_time_earth'] = ax_earth.axvline(t0, c='y', linewidth=2)
        self.plot_data['den_time_stereo_a'] = ax_stereo_a.axvline(t0, c='y', linewidth=2)
        self.plot_data['den_time_stereo_b'] = ax_stereo_b.axvline(t0, c='y', linewidth=2)

        ax_earth = self.axes['vel_time_earth']
        ax_stereo_a = self.axes['vel_time_stereo_a']
        ax_stereo_b = self.axes['vel_time_stereo_b']
        self.plot_data['vel_time_earth'] = ax_earth.axvline(t0, c='y', linewidth=2)
        self.plot_data['vel_time_stereo_a'] = ax_stereo_a.axvline(t0, c='y', linewidth=2)
        self.plot_data['vel_time_stereo_b'] = ax_stereo_b.axvline(t0, c='y', linewidth=2)

        # Title
        ax = self.axes['title']
        # Turn all borders and spines off
        ax.axis('off')
        self.plot_data['title'] = ax.text(0.5, 0.5, str(t0).replace('T', ' ')[:16],
                                          fontsize=30, color='w',
                                          horizontalalignment='center',
                                          verticalalignment='center')

        # Update the plot with all of the proper data from the first time-step
        self.fig.canvas.draw()
        self.fig.set_constrained_layout(False)
        return self.update_plot(self.time)

    def init_time_series(self):
        times = self.enlil_run.earth_times
        earth_den = self.enlil_run.get_satellite_data('Earth', 'density')
        stereo_a_den = self.enlil_run.get_satellite_data('Earth', 'density')
        stereo_b_den = self.enlil_run.get_satellite_data('Earth', 'density')
        ax_earth = self.axes['den_time_earth']
        ax_stereo_a = self.axes['den_time_stereo_a']
        ax_stereo_b = self.axes['den_time_stereo_b']

        ax_earth.plot(times, earth_den, c='tab:green')
        ax_stereo_a.plot(times, stereo_a_den, c='tab:red')
        ax_stereo_b.plot(times, stereo_b_den, c='tab:blue')
        plt.setp(ax_earth.get_xticklabels(), visible=False)
        plt.setp(ax_stereo_a.get_xticklabels(), visible=False)
        ax_earth.set_xlim(self.enlil_run.times[0], self.enlil_run.times[-1])
        ax_earth.xaxis.set_major_locator(mpl.dates.DayLocator())
        ax_earth.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d"))
        ax_earth.set_title('Plasma Density (/cm$^3$)')

        earth_vel = self.enlil_run.get_satellite_data('Earth', 'velocity')
        stereo_a_vel = self.enlil_run.get_satellite_data('STEREO_A', 'velocity')
        stereo_b_vel = self.enlil_run.get_satellite_data('STEREO_B', 'velocity')
        ax_earth = self.axes['vel_time_earth']
        ax_stereo_a = self.axes['vel_time_stereo_a']
        ax_stereo_b = self.axes['vel_time_stereo_b']

        ax_earth.plot(times, earth_vel, c='tab:green')
        ax_stereo_a.plot(times, stereo_a_vel, c='tab:red')
        ax_stereo_b.plot(times, stereo_b_vel, c='tab:blue')
        plt.setp(ax_earth.get_xticklabels(), visible=False)
        plt.setp(ax_stereo_a.get_xticklabels(), visible=False)
        ax_earth.set_xlim(self.enlil_run.times[0], self.enlil_run.times[-1])
        ax_earth.xaxis.set_major_locator(mpl.dates.DayLocator())
        ax_earth.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d"))
        ax_earth.set_title('Radial Velocity (km/s)')

    @property
    def time(self):
        """Current time for the plot."""
        return self.enlil_run.times[self.index]

    def update_plot(self, time):
        """Updates the plot with data from the requested time."""
        ds = self.enlil_run
        r = self.enlil_run.r
        lat = self.enlil_run.lat

        
        self.index = np.nonzero(time <= self.enlil_run.times)[0][0]

        # ---------------
        # Longitude plots
        # ---------------
        den_data = enlil.get_data('density', 'r', 'lon', time=time)
        den_data = _cyclic_point(den_data.T).T
        self.plot_data['den_lon_mesh'].set_array(den_data.flatten())

        vel_data = enlil.get_data('velocity', 'r', 'lon', time=time)
        vel_data = _cyclic_point(vel_data.T).T
        self.plot_data['vel_lon_mesh'].set_array(vel_data.flatten())

        pol_data = enlil.get_data('polarity', 'r', 'lon', time=time)
        pol_data = _cyclic_point(pol_data.T).T
        neg_polarity = pol_data < 0
        arc_min_lons = np.ma.masked_where(neg_polarity[:, 0], lon)
        arc_min_rs = np.ma.masked_array(np.ones(shape=lon.shape)*r[0], mask=arc_min_lons.mask)
        arc_max_lons = np.ma.masked_where(neg_polarity[:, -1], lon)
        arc_max_rs = np.ma.masked_array(np.ones(shape=lon.shape)*r[-1], mask=arc_max_lons.mask)
        self.plot_data['den_lon_arc_min'].set_data(arc_min_lons, arc_min_rs)
        self.plot_data['den_lon_arc_max'].set_data(arc_max_lons, arc_max_rs)
        self.plot_data['vel_lon_arc_min'].set_data(arc_min_lons, arc_min_rs)
        self.plot_data['vel_lon_arc_max'].set_data(arc_max_lons, arc_max_rs)

        # --------------
        # Latitude plots
        # --------------
        den_data = enlil.get_data('density', 'r', 'lat', time=time)
        self.plot_data['den_lat_mesh'].set_array(den_data.values.flatten())

        vel_data = enlil.get_data('velocity', 'r', 'lat', time=time)
        self.plot_data['vel_lat_mesh'].set_array(vel_data.values.flatten())

        pol_data = enlil.get_data('polarity', 'r', 'lat', time=time)
        neg_polarity = pol_data < 0
        arc_min_lats = np.ma.masked_where(neg_polarity[:, 0], lat)
        arc_min_rs = np.ma.masked_array(np.ones(shape=lat.shape)*r[0], mask=arc_min_lats.mask)
        arc_max_lats = np.ma.masked_where(neg_polarity[:, -1], lat)
        arc_max_rs = np.ma.masked_array(np.ones(shape=lat.shape)*r[-1], mask=arc_min_lats.mask)
        self.plot_data['den_lat_arc_min'].set_data(arc_min_lats, arc_min_rs)
        self.plot_data['den_lat_arc_max'].set_data(arc_max_lats, arc_max_rs)
        self.plot_data['vel_lat_arc_min'].set_data(arc_min_lats, arc_min_rs)
        self.plot_data['vel_lat_arc_max'].set_data(arc_max_lats, arc_max_rs)

        # -------------------
        # Satellite positions
        # -------------------
        earth_pos = enlil.get_satellite_position('Earth', time)
        stereo_a_pos = enlil.get_satellite_position('STEREO_A', time)
        stereo_b_pos = enlil.get_satellite_position('STEREO_B', time)
        # set_offsets for scatter marker positions (polar plot is: lon, r)
        self.plot_data['den_lon_earth'].set_offsets([[earth_pos[2], earth_pos[0]]])
        self.plot_data['den_lon_stereo_a'].set_offsets([[stereo_a_pos[2], stereo_a_pos[0]]])
        self.plot_data['den_lon_stereo_b'].set_offsets([[stereo_b_pos[2], stereo_b_pos[0]]])
        # Update latitude plot too
        self.plot_data['den_lat_earth'].set_offsets([[earth_pos[1], earth_pos[0]]])
        # Velocity plots
        self.plot_data['vel_lon_earth'].set_offsets([[earth_pos[2], earth_pos[0]]])
        self.plot_data['vel_lon_stereo_a'].set_offsets([[stereo_a_pos[2], stereo_a_pos[0]]])
        self.plot_data['vel_lon_stereo_b'].set_offsets([[stereo_b_pos[2], stereo_b_pos[0]]])
        # Update latitude plot too
        self.plot_data['vel_lat_earth'].set_offsets([[earth_pos[1], earth_pos[0]]])

        # -----------------
        # Time-series lines
        # -----------------
        for line in [self.plot_data['den_time_earth'], self.plot_data['den_time_stereo_a'],
                     self.plot_data['den_time_stereo_b'], self.plot_data['vel_time_earth'],
                     self.plot_data['vel_time_stereo_a'], self.plot_data['vel_time_stereo_b']]:
            line.set_xdata(time)

        # Title
        self.plot_data['title'].set_text(str(self.time).replace('T', ' ')[:16])

        self._artists = [self.plot_data['den_lon_mesh'], self.plot_data['den_lat_mesh'],
                         self.plot_data['vel_lon_mesh'], self.plot_data['vel_lat_mesh'],
                         self.plot_data['den_time_earth'], self.plot_data['den_time_stereo_a'],
                         self.plot_data['den_time_stereo_b'], self.plot_data['vel_time_earth'],
                         self.plot_data['vel_time_stereo_a'], self.plot_data['vel_time_stereo_b'],
                         self.plot_data['title']]
        # Removing the mesh plots from the blit artists
        # self._artists = [self.plot_data['den_time_earth'], self.plot_data['den_time_stereo_a'],
        #                  self.plot_data['den_time_stereo_b'], self.plot_data['vel_time_earth'],
        #                  self.plot_data['vel_time_stereo_a'], self.plot_data['vel_time_stereo_b']]
        self._artists = [self.plot_data[x] for x in self.plot_data]
        for artist in self._artists:
            artist.set_animated(True)
        return self._artists

    def save(self, filename):
        """Saves the current figure with the given filename."""
        self.fig.savefig(filename)


def _mesh_grid(x, y):
    """
    matplotlib.pcolormesh currently needs data specified at edges
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
