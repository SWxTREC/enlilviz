========
enlilviz
========


.. image:: https://img.shields.io/pypi/v/enlilviz.svg
        :target: https://pypi.python.org/pypi/enlilviz

.. image:: https://img.shields.io/travis/SWxTREC/enlilviz.svg
        :target: https://travis-ci.org/SWxTREC/enlilviz

.. image:: https://readthedocs.org/projects/enlilviz/badge/?version=latest
        :target: https://enlilviz.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




A Python package for visualizing the output from the Enlil solar wind code.

* Documentation: https://enlilviz.readthedocs.io.


Features
--------

Read in Enlil data files into an xarray dataset for analysis::

  import enlilviz as ev

  run = ev.read_enlil2d('wsa_enlil.latest.suball.nc')

  evo = ev.read_evo('evo.earth.nc')

Plot time series and slices with the data::

  import enlilviz.plotting as evplot

  evplot.TimeSeries(run, 'Earth', 'den')

  evplot.LongitudeSlice(run, 'den')

  evplot.LatitudeSlice(run, 'vel')

You can also generate common figures that are pre-arranged::

  forecaster = evplot.ForecasterPlot(run)
  forecaster.save()

Or iterate through the entire time series to make a movie::

  for plot in iter(forecaster):
      plot.save()
