"""Default forecast center figure."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from enlilviz.plotting.plots import (LatitudeSlice, LongitudeSlice,
                                     TimeSeries, Title, Colorbar)

mpl.style.use(['dark_background'])


class ForecasterPlot:
    """Figure class for the main Forecaster plot.

    Parameters
    ----------
    enlil_run : enlilviz.enlil.Enlil
        An Enlil model run.
    watermark : str, optional
        An optional watermark to add to the plots.
    """

    def __init__(self, enlil_run, watermark=None):
        self.enlil_run = enlil_run
        self.watermark = watermark
        # Store the current time index
        self._index = 0

        # Figure and axis creation
        self.fig = plt.figure(figsize=(20, 12.5), dpi=72)

        # Dictionary to store axes
        self.plots = {}

        self._init_plots()

    def _init_plots(self):
        """Sets up the axes for the plots and initializes the data."""
        fig = self.fig
        run = self.enlil_run

        # Top title, span the entire distance
        title = fig.add_axes([0, 0.9, 1, 0.1])
        self.plots["title"] = Title(run, ax=title)

        # Bottom extra matter, span the entire distance
        bottom = fig.add_axes([0, 0, 1, 0.05])
        bottom.axis('off')
        # Put model info in the bottom right corner
        bottom.text(0.995, 0.005, self._model_info(), alpha=0.5,
                    horizontalalignment='right', verticalalignment='bottom')
        # Watermark text in the bottom left
        if self.watermark:
            bottom.text(0.005, 0.005, self.watermark, alpha=0.5,
                        horizontalalignment='left', verticalalignment='bottom')

        # Center heights for velocity/density rows
        center_vel = 0.25
        center_den = 0.7

        # This was rigidly defined for this plot
        aspect_ratio = 20/12.5

        # how much height for the rows to take up
        h = 0.4
        # Make sure the width polar axes are the same aspect ratio
        w = 0.4/aspect_ratio

        # Colorbars
        cbar_h = h * 2/3
        cbar_w = 0.01
        cbar_den = fig.add_axes([0.025, center_den - cbar_h/2, cbar_w, cbar_h])
        Colorbar('den', cbar_den)
        cbar_vel = fig.add_axes([0.025, center_vel - cbar_h/2, cbar_w, cbar_h])
        Colorbar('vel', cbar_vel)
        cbar_w = 0.075

        # Polarity text
        ax_pol = fig.add_axes([0.0025, (center_den + center_vel)/2 - 0.05,
                               cbar_w, cbar_w])
        ax_pol.axis('off')
        ax_pol.text(0.5, 0.5, "Polarity", fontsize=18,
                    horizontalalignment='center', verticalalignment='center')
        ax_pol2 = fig.add_axes([0.0025, (center_den + center_vel)/2 - 0.07,
                               cbar_w, cbar_w])
        ax_pol2.axis('off')
        ax_pol2.text(0.5, 0.5, "+", fontsize=18, color='tab:orange',
                     horizontalalignment='center', verticalalignment='center')

        # First polar plots (latitude slices)
        lat_den = fig.add_axes([cbar_w, center_den - h/2, w, h],
                               projection='polar')
        self.plots['lat_den'] = LatitudeSlice(run, 'den', ax=lat_den)
        lat_vel = fig.add_axes([cbar_w, center_vel - h/2, w, h],
                               projection='polar')
        self.plots['lat_vel'] = LatitudeSlice(run, 'vel', ax=lat_vel)

        # Second polar plots (longitude slices)
        # They partially overlap in the full circle, but when axes extents
        # are set this will be a wedge
        lon_den = fig.add_axes([cbar_w + w*0.9, center_den - h/2, w, h],
                               projection='polar')
        self.plots['lon_den'] = LongitudeSlice(run, 'den', ax=lon_den)
        lon_vel = fig.add_axes([cbar_w + w*0.9, center_vel - h/2, w, h],
                               projection='polar')
        self.plots['lon_vel'] = LongitudeSlice(run, 'vel', ax=lon_vel)

        # Time series plots
        ts_h = 0.1
        ts_w = 0.435
        gap = 0.025
        left = 0.55
        ts1_den = fig.add_axes([left, center_den + ts_h/2 + gap, ts_w, ts_h])
        ts2_den = fig.add_axes([left, center_den - ts_h/2, ts_w, ts_h],
                               sharex=ts1_den, sharey=ts1_den)
        ts3_den = fig.add_axes([left, center_den - ts_h/2 - gap - ts_h,
                                ts_w, ts_h], sharex=ts1_den, sharey=ts1_den)
        self.plots['ts_den_earth'] = TimeSeries(run, 'Earth', 'den',
                                                ax=ts1_den)
        self.plots['ts_den_stereo_a'] = TimeSeries(run, 'STEREO_A', 'den',
                                                   ax=ts2_den)
        self.plots['ts_den_stereo_b'] = TimeSeries(run, 'STEREO_B', 'den',
                                                   ax=ts3_den)

        ts1_vel = fig.add_axes([left, center_vel + 0.075, ts_w, ts_h])
        ts2_vel = fig.add_axes([left, center_vel - ts_h/2, ts_w, ts_h],
                               sharex=ts1_vel, sharey=ts1_vel)
        ts3_vel = fig.add_axes([left, center_vel - ts_h/2 - gap - ts_h,
                                ts_w, ts_h], sharex=ts1_vel, sharey=ts1_vel)
        self.plots['ts_vel_earth'] = TimeSeries(run, 'Earth', 'vel',
                                                coord='r', ax=ts1_vel)
        self.plots['ts_vel_stereo_a'] = TimeSeries(run, 'STEREO_A', 'vel',
                                                   coord='r', ax=ts2_vel)
        self.plots['ts_vel_stereo_b'] = TimeSeries(run, 'STEREO_B', 'vel',
                                                   coord='r', ax=ts3_vel)

    @property
    def time(self):
        """Current time of the plot."""
        return self.enlil_run.times[self._index]

    @property
    def _time_string(self):
        """String representation of the current time."""
        t = str(self.time)[:16]
        # Remove bad characters from the filename
        for char in " :-":
            t = t.replace(char, '')
        # Floor the minutes to match previous work
        t = t[:-2] + "00"
        return t

    def _model_info(self):
        """String representation of the model run."""
        attrs = self.enlil_run.ds.attrs
        s = "Enlil v{0} | {1} | Model Run ID: {2}".format(
                attrs['enlil_version'],
                attrs['wsa_version'].replace('_', ' ').replace('V', 'v'),
                attrs['model_run_id'])
        return s

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

    def update(self):
        """Updates the plot with all of the proper data."""
        for plot in self.plots:
            p = self.plots[plot]
            p.set_index(self._index)

    def save(self, filename=None):
        """
        Saves the current figure with the given filename.
        If filename is None, the filename will be made from
        the current time: 'enlil_{curr_time}.png'."""
        if filename is None:
            filename = "enlil_{}.png".format(self._time_string)
        self.fig.savefig(filename)
