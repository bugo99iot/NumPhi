import math
import numpy as np
import random

from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, Slider, CustomJS
from bokeh.models.sources import ColumnDataSource
from bokeh.transform import linear_cmap
from bokeh.layouts import column, widgetbox
from bokeh.palettes import Plasma256 as palette  # this import will will be highlighted by PyCharm, ignore it
from numpy.core._multiarray_umath import ndarray

from typing import List
import logging
import os

# define logging level
logger = logging.getLogger(__name__)

if os.getenv('ENV') in ['staging', 'production']:
    logging.basicConfig(level=logging.WARNING)
else:
    logging.basicConfig(level=logging.INFO)


def plot_bokeh_board_toy():

    data_dict_1 = {"t": np.array([1.0, 0.0, 0.0]),
                   "r": np.array([0, 0, 0]),
                   "q": np.array([1, 2, 3])}

    data_dict_2 = {"t": np.array([0.0, 1.0, 0.0]),
                   "r": np.array([0, 0, 0]),
                   "q": np.array([1, 2, 3])}

    data_dict_3 = {"t": np.array([0.0, 0.0, 1.0]),
                   "r": np.array([0, 0, 0]),
                   "q": np.array([1, 2, 3])}

    sources = [ColumnDataSource(data_dict_1), ColumnDataSource(data_dict_1), ColumnDataSource(data_dict_2), ColumnDataSource(data_dict_3)]

    # select source to begin with
    source = sources[0]

    size = 0.5
    orientation = "pointytop"

    p = figure(title="Test Bokeh", match_aspect=True, tools="wheel_zoom,reset", plot_width=500, plot_height=500)

    p.hex_tile(q="q", r="r", size=size,
               fill_color=linear_cmap('t', palette, 0.0, 1.0),
               line_color=None, source=source, orientation=orientation,
               hover_color="pink", hover_alpha=0.8)

    # Add the slider
    code = """       
        var step = cb_obj.value + 1;    
        var new_data = sources[step].data;
        source.data = new_data;
        source.change.emit();   
        """

    callback = CustomJS(args=dict(source=source, sources=sources), code=code)

    slider = Slider(start=0, end=len(sources)-2, step=1, value=0, title="Epoch", callback=callback)

    hover = HoverTool(tooltips=[("tolerance", "@t{0.00}"),
                                ("r, q", "@r, @q")], callback=callback)

    p.add_tools(hover)

    layout = column(p, widgetbox(slider))

    show(layout)

plot_bokeh_board_toy()