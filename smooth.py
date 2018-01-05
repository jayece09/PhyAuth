import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np
import pandas as pd
import scipy

from scipy import signal


x = np.linspace(0, 10, 100)
y = np.sin(x)
y_noise = [y_item + np.random.choice([-1, 1])*np.random.random() for y_item in y]

trace1 = go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(
        size=2,
        color='rgb(0, 0, 0)',
    ),
    name='Sine'
)

trace2 = go.Scatter(
    x=x,
    y=y_noise,
    mode='markers',
    marker=dict(
        size=6,
        color='#5E88FC',
        symbol='circle-open'
    ),
    name='Noisy Sine'
)

trace3 = go.Scatter(
    x=x,
    y=signal.savgol_filter(y, 53, 3),
    mode='markers',
    marker=dict(
        size=6,
        color='#C190F0',
        symbol='triangle'
    ),
    name='Savitzky-Golay'
)

layout = go.Layout(
    showlegend=True
)

data = [trace1, trace2, trace3]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='smoothing-savitzky-golay-filter')