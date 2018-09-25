# -*- coding: utf-8 -*-
"""
Functions and classes that provide visualisation functionalities
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os


class Visualization:
    """
    Creates a new visualization in a new window. Any added subcharts will be added to this window.

    Visualisation(dims=(10, 10))

    Parameters
    ----------
    dims: list of int, optional, default=(10,10)
        size of the newly created window

    Attributes
    ----------
    Visualisation.figure_number: int, begin=0
        number of figures that have been created
    Visualisation.horizontal_padding: float, default=0.4
        horizontal padding of plot
    Visualisation.font_size_label: int, default=12
        font size of title
    Visualisation.font_size_label: int, default=12
        font size of label
    Visualisation.graph_lind_width: int, default 2
        line width of graph
    fig: mpl.figure
        handle of figure created by matplotlib.pyplot
    """

    figure_number = 0
    horizontal_padding = 0.4
    font_size_label = 12
    font_size_title = 12
    graph_line_width = 2

    def __init__(self, dims=(10, 10)):
        self.fig = plt.figure(Visualization.figure_number, figsize=(dims[0], dims[0]), facecolor=[1, 1, 1])
        Visualization.figure_number += 1
        # add some horizontal spacing to avoid overlap with labels
        plt.subplots_adjust(hspace=Visualization.horizontal_padding)

    def create_new_chart(self, layout_id=None):
        """
        Add a new subplot to the current visualization, so that multiple graphs can be overlaid onto one chart
        (e.g. scatterplot over heatmap).

        create_new_chart(layout_id=None)

        Parameters
        ----------
        layout_id: (3-digit) int, optional, default=None
            denoting the position of the graph in figure (xyn : 'x'=width, 'y'=height of grid, 'n'=position within grid)
        """
        self.fig.add_subplot(layout_id)

    def add_line_plot(self, title, labels, data, x_lim=None, y_lim=None):
        """
        Draw a 1D line graph into the current figure.

        add_line_plot(title, labels, data, x_lim=None, y_lim=None)

        Parameters
        ----------
        title: str
            title of the graph
        labels: {str:str} dict
            {'x': name of x-axis, 'y': name of y-axis}
        x_lim: [2] list of float, optional, default=None
            x limits for the function argument or value
        y_lim: [2] list of float, optional, default=None
            y limits for the function argument or value
        """
        self.create_sub_plot(title, labels, x_lim=x_lim, y_lim=y_lim)

        for i in range(len(data['pointSets'])):
            plt.plot(data['pointSets'][i]['x'], data['pointSets'][i]['y'],
                     linestyle=data['linestyle'][i],
                     color=data['color'][i],
                     linewidth=Visualization.graph_line_width)

        plt.legend(data['names'], loc="upper left")
        plt.grid()

    def add_heat_map(self, title, labels, grid_points, data_points, v_lim=(None, None),
                     x_lim=None, y_lim=None, colormap=None):
        """
        Draw a 2D heatmap into the current figure.

        add_heat_map(title, labels, grid_points, data_points, v_lim=(None, None), x_lim=None, y_lim=None, colormap=None)

        Parameters
        ----------
        title: str
            title of the graph
        labels: {str:str} dict
            {'x': name of x-axis, 'y': name of y-axis}
        grid_points: [2] list of np.ndarray
            arrays of the x and y positions of the grid points
        data_points: np.ndarray of the data points that are placed into the grid
        x_lim: [2] list of float, optional, default=None
            x limits for the function argument or value
        y_lim: [2] list of float, optional, default=None
            y limits for the function argument or value
        v_lim: [2] list of float, optional, default=(None,None)
            limits of the color scale
        colormap: str, optional, default=None
            the colormap to use
        """
        self.create_sub_plot(title, labels, x_lim=x_lim, y_lim=y_lim)

        plt.pcolormesh(grid_points[0], grid_points[1], data_points, vmin=v_lim[0], vmax=v_lim[1], cmap=colormap)

        plt.colorbar()

    @staticmethod
    def add_scatter_plot(shape, plot_size, color_sequence, colormap=None, v_lim=(None, None)):
        """
        Draw a scatter plot onto the current chart.

        add_scatter_plot(shape, plot_size, color_sequence, colormap=None, v_lim=(None, None))

        Parameters
        ----------
        shape: {str: np.ndarray} dict
            {'x': positions on x-axis, 'y': positions on y-axis}
        plot_size: np.ndarray
            the marker size in the squared number of points
        color_sequence: str or list
            marker colors
        colormap: str, optional, default=None
            the colormap to use
        v_lim: [2] list of float, optional, default=(None,None)
            limits of the color scale
        """
        plt.scatter(shape['x'], shape['y'], s=plot_size, c=color_sequence, vmin=v_lim[0], vmax=v_lim[1], cmap=colormap)

    @staticmethod
    def create_sub_plot(title, labels, x_lim, y_lim):
        """
        Set the title, labels and the axis limits of a plot.

        create_sub_plot(title, labels, x_lim, y_lim)

        Parameters
        ----------
        title: str
            title of the plot
        labels: {str:str} dict
            {'x': name of x-axis, 'y': name of y-axis}
        x_lim: [2] list of float
            x limits for the function argument or value
        y_lim: [2] list of float
            y limits for the function argument or value
        """
        plt.title(title, fontsize=Visualization.font_size_title)
        plt.ylabel(labels['y'], fontsize=Visualization.font_size_label)
        plt.xlabel(labels['x'], fontsize=Visualization.font_size_label)

        ax = plt.gca()
        if x_lim is not None:
            ax.set_xlim(x_lim[0], x_lim[1])
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])

    @staticmethod
    def show():
        """
        Show plots.
        """
        plt.show()


def plot_sobol_indices(sobol_rel_order_mean, sobol_rel_1st_order_mean, fn_plot, random_vars):
    """
    Plot the Sobol indices into different sub-plots.

    plot_sobol_indices(sobol_rel_order_mean, sobol_rel_1st_order_mean, fn_plot, random_vars)

    Parameters
    ----------
    sobol_rel_order_mean: np.ndarray
        average proportion of the Sobol indices of the different order to the total variance (1st, 2nd, etc..,)
        over all output quantities
    sobol_rel_1st_order_mean: np.ndarray
        average proportion of the random variables of the 1st order Sobol indices to the total variance over all
        output quantities
    fn_plot: str
        filename of plot
    random_vars: list of str [dim]
        string labels of the random variables
    """

    # set the global colors
    mpl.rcParams['text.color'] = '000000'
    mpl.rcParams['figure.facecolor'] = '111111'

    # set a global style
    plt.style.use('seaborn-talk')

    cmap = plt.cm.rainbow

    # make bar plot of order ratios
    labels = ['order=' + str(i) for i in range(1, len(sobol_rel_order_mean) + 1)]
    mask = np.where(sobol_rel_order_mean >= 0.05)[0]
    mask_not = np.where(sobol_rel_order_mean < 0.05)[0]
    labels = [labels[idx] for idx in mask]
    if mask_not.any():
        labels.append('misc.')
        values = np.hstack((sobol_rel_order_mean[mask], np.sum(sobol_rel_order_mean[mask_not])))
    else:
        values = sobol_rel_order_mean

    colors = cmap(np.linspace(0.1, 0.9, len(labels)))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_title('Sobol indices (order)')
    ax.pie(values, labels=labels, colors=colors,
           autopct='%1.2f%%', shadow=True, explode=[0.1] * len(labels))
    plt.savefig(os.path.splitext(fn_plot)[0] + '_order.png', facecolor='#ffffff')

    # make bar plot of 1st order parameter ratios
    mask = np.where(sobol_rel_1st_order_mean >= 0.05)[0]
    mask_not = np.where(sobol_rel_1st_order_mean < 0.05)[0]
    labels = [random_vars[idx] for idx in mask]
    if mask_not.any():
        labels.append('misc.')
        values = np.hstack((sobol_rel_1st_order_mean[mask], np.sum(sobol_rel_1st_order_mean[mask_not])))
    else:
        values = sobol_rel_1st_order_mean

    colors = cmap(np.linspace(0., 1., len(labels)))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_title('Sobol indices 1st order (parameters)')
    ax.pie(values, labels=labels, colors=colors,
           autopct='%1.2f%%', shadow=True, explode=[0.1] * len(labels))
    plt.savefig(os.path.splitext(fn_plot)[0] + '_parameters.png', facecolor='#ffffff')
