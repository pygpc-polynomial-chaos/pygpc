import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os


class Visualization:
    """
    Creates a new visualization in a new window. Any added subcharts will be added to this window.

    Parameters:
    -----------
    dims: list of int
        size of the newly created window
    """

    # class variables
    figNo = 0
    horizontalPadding = 0.4
    fontSizeLabel = 12
    fontSizeTitle = 12
    graphLineWidth = 2

    def __init__(self, dims=(10, 10)):
        self.fig = plt.figure(Visualization.figNo, figsize=(dims[0], dims[0]), facecolor=[1, 1, 1])
        Visualization.figNo += 1
        # add some horizontal spacing to avoid overlap with labels
        plt.subplots_adjust(hspace=Visualization.horizontalPadding)

    def create_new_chart(self, layout_id=None):
        """
        Add a new subplot to the current visualization, so that multiple graphs can be overlaid onto one chart
        (e.g. scatterplot over heatmap).

        Parameters:
        -----------
        layout_id: int (3-digit)
            denoting the position of the graph in figure (xyn : 'x'=width, 'y'=height of grid, 'n'=position within grid)
        """
        self.fig.add_subplot(layout_id)

    def add_line_plot(self, title, labels, data, x_lim=None, y_lim=None):
        """
        Draw a 1D line graph into the current figure.

        Parameters:
        -----------
        title: str
            title of the graph
        labels: {string:string} dict
            {'x': name of x-axis, 'y': name of y-axis}
        x_lim: [2x1] list of float
            x limits for the function argument or value
        y_lim: [2x1] list of float
            y limits for the function argument or value
        """
        self.create_sub_plot(title, labels, x_lim=x_lim, y_lim=y_lim)

        for i in range(len(data['pointSets'])):
            plt.plot(data['pointSets'][i]['x'], data['pointSets'][i]['y'],
                     linestyle=data['linestyle'][i],
                     color=data['color'][i],
                     linewidth=Visualization.graphLineWidth)

        plt.legend(data['names'], loc="upper left")
        plt.grid()

    def add_heat_map(self, title, labels, grid_points, data_points, v_lim=(None, None),
                     x_lim=None, y_lim=None, colormap=None):
        """
        Draw a 2D heatmap into the current figure.

        Parameters:
        -----------
        title. str
            title of the graph
        labels: {string:string} dict
            {'x': name of x-axis, 'y': name of y-axis}
        grid_points:  - 2 x n array: containing an the arrays of x|y positions of the grid points
        data_points: - 1D array_ of the data-points that should be positoned into the grid
        x_lim: [2x1] list of float
            x limits for the function argument or value
        y_lim: [2x1] list of float
            y limits for the function argument or value
        v_lim: [2x1] list of float
            limits of the color scale
        colormap: str
            the colormap to use
        """
        self.create_sub_plot(title, labels, x_lim=x_lim, y_lim=y_lim)

        plt.pcolormesh(grid_points[0], grid_points[1], data_points, vmin=v_lim[0], vmax=v_lim[1], cmap=colormap)

        plt.colorbar()

    @staticmethod
    def add_scatter_plot(shape, plot_size, color_sequence, colormap=None, v_lim=(None, None)):
        """
        draws a scatterplot onto the current chart

        Parameters:
        -----------
        shape : {string: np.ndarray} dict
            {'x': positions on x-axis, 'y': positions on y-axis}
        plot_size: ??
        color_sequence: ??
        colormap: str
            the colormap to use
        v_lim: [2x1] list of float
            limits of the color scale
        """
        plt.scatter(shape['x'], shape['y'], s=plot_size, c=color_sequence, vmin=v_lim[0], vmax=v_lim[1], cmap=colormap)

    @staticmethod
    def create_sub_plot(title, labels, x_lim, y_lim):
        """
        Helper function that sets the title, labels and the axis limits of a plot.

        Parameters:
        -----------
        title: str
            title of the plot
        labels: {string:string} dict
            {'x': name of x-axis, 'y': name of y-axis}
        x_lim: [2x1] list of float
            x limits for the function argument or value
        y_lim: [2x1] list of float
            y limits for the function argument or value
        """
        plt.title(title, fontsize=Visualization.fontSizeTitle)
        plt.ylabel(labels['y'], fontsize=Visualization.fontSizeLabel)
        plt.xlabel(labels['x'], fontsize=Visualization.fontSizeLabel)

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
