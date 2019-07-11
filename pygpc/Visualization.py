# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import scipy.stats
import warnings

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    warnings.warn("If you want to use plot functionality from pygpc, "
                  "please install matplotlib (pip install matplotlib).")
    pass


class Visualization:
    """
    Creates a new visualization in a new window. Any added sub-charts will be added to this window.

    Visualisation(dims=(10, 10))

    Attributes
    ----------
    Visualisation.figure_number: int, begin=0
        Number of figures that have been created
    Visualisation.horizontal_padding: float, default=0.4
        Horizontal padding of plot
    Visualisation.font_size_label: int, default=12
        Font size of title
    Visualisation.font_size_label: int, default=12
        Font size of label
    Visualisation.graph_lind_width: int, default 2
        Line width of graph
    fig: mpl.figure
        Handle of figure created by matplotlib.pyplot

    Parameters
    ----------
    dims: list of int, optional, default=(10,10)
        Size of the newly created window
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
        mpl.rcParams['text.usetex'] = True

    def create_new_chart(self, layout_id=None):
        """
        Add a new subplot to the current visualization, so that multiple graphs can be overlaid onto one chart
        (e.g. scatterplot over heatmap).

        create_new_chart(layout_id=None)

        Parameters
        ----------
        layout_id: (3-digit) int, optional, default=None
            Denoting the position of the graph in figure (xyn : 'x'=width, 'y'=height of grid, 'n'=position within grid)
        """
        self.fig.add_subplot(layout_id)

    def add_line_plot(self, title, labels, data, x_lim=None, y_lim=None):
        """
        Draw a 1D line graph into the current figure.

        add_line_plot(title, labels, data, x_lim=None, y_lim=None)

        Parameters
        ----------
        title: str
            Title of the graph
        labels: {str:str} dict
            {'x': name of x-axis, 'y': name of y-axis}
        x_lim: list of float [2], optional, default=None
            x-limits for the function argument or value
        y_lim: list of float [2], optional, default=None
            y-limits for the function argument or value
        data: ndarray of float
            Data that should be plotted
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
            Title of the graph
        labels: {str:str} dict
            {'x': name of x-axis, 'y': name of y-axis}
        grid_points: list of ndarray of float [2]
            Arrays of the x and y positions of the grid points e.g.: [np.array(x_points), np.array(y_points)]
        data_points: np.ndarray of the data points that are placed into the grid
        x_lim: list of float [2], optional, default=None
            x-limits for the function argument or value
        y_lim: list of float [2], optional, default=None
            y-limits for the function argument or value
        v_lim: list of float [2], optional, default=(None,None)
            Limits of the color scale
        colormap: str, optional, default=None
            The colormap to use
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
        plot_size: ndarray of float
            The marker size in the squared number of points
        color_sequence: str or list of str
            Marker colors
        colormap: str, optional, default=None
            The colormap to use
        v_lim: list of float [2], optional, default=(None,None)
            Limits of the color scale
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
            Title of the plot
        labels: {str:str} dict
            {'x': name of x-axis, 'y': name of y-axis}
        x_lim: list of float [2]
            x-limits for the function argument or value
        y_lim: list of float [2]
            y-limits for the function argument or value
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


def b2rcw(cmin_input, cmax_input):
    """ Blue, white, and red color map.
    This function is designed to generate a blue to red colormap. The color of the colorbar is from blue to white and
    then to red, corresponding to the data values from negative to zero to positive, respectively.
    The color white always corresponds to value zero. The brightness of blue and red will change according to your
    setting, so that the brightness of the color corresponded to the color of his opposite number.

    Parameters
    ----------
    cmin_input: float
        Minimum value of data
    cmax_input: float
        Maximum value of data

    Returns
    -------
    newmap: ndarray of float [N_RGB x 3]
        Colormap

    Examples
    --------
    >>> b2rcw_cmap_1 = make_cmap(b2rcw(-3, 6)) # is from light blue to deep red
    >>> b2rcw_cmap_2 = make_cmap(b2rcw(-3, 3)) # is from deep blue to deep red
    """

    # check the input
    if cmin_input >= cmax_input:
        raise ValueError('input error, the color range must be from a smaller one to a larger one')

    # color configuration : from blue to light blue to white until to red
    red_top = np.array([1, 0, 0])
    white_middle = np.array([1, 1, 1])
    blue_bottom = np.array([0, 0, 1])

    # color interpolation
    color_num = 250
    color_input = np.vstack((blue_bottom, white_middle, red_top))
    oldsteps = np.array([-1, 0, 1])
    newsteps = np.linspace(-1, 1, color_num)

    newmap_all = np.zeros((color_num, 3))*np.nan

    for j in range(3):
        newmap_all[:, j] = np.min(np.vstack((np.max(
            np.vstack((np.interp(newsteps, oldsteps, color_input[:, j]), np.zeros(color_num))), axis=0),
                                             np.ones(color_num))), axis=0)

    if (cmin_input < 0) & (cmax_input > 0):

        if np.abs(cmin_input) < cmax_input:
            #    |--------|---------|--------------------|
            # -cmax      cmin       0                  cmax         [cmin,cmax]

            start_point = int(np.ceil((cmin_input+cmax_input)/2.0/cmax_input*color_num)-1)
            newmap = newmap_all[start_point:color_num, :]

        elif np.abs(cmin_input) >= cmax_input:
            #    |------------------|------|--------------|
            #   cmin                0     cmax          -cmin         [cmin,cmax]

            end_point = int(np.round((cmax_input-cmin_input)/2.0/np.abs(cmin_input)*color_num)-1)
            newmap = newmap_all[1:end_point, :]

    elif cmin_input >= 0:

        #   |-----------------|-------|-------------|
        # -cmax               0      cmin          cmax         [cmin,cmax]

        start_point = int(np.round((cmin_input+cmax_input)/2.0/cmax_input*color_num)-1)
        newmap = newmap_all[start_point:color_num, :]

    elif cmax_input <= 0:
        #   |------------|------|--------------------|
        #  cmin         cmax    0                  -cmin         [cmin,cmax]

        end_point = int(np.round((cmax_input-cmin_input)/2.0/np.abs(cmin_input)*color_num)-1)
        newmap = newmap_all[1:end_point, :]

    else:
        newmap = None

    return newmap


def make_cmap(colors, position=None, bit=False):
    """
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.


    Parameters
    ----------
    colors: list of 3-tuples [n_rgb]
        RGB values. The RGB values may either be in 8-bit [0 to 255] (in which bit must be set to True when called)
        or arithmetic [0 to 1] (default).
    position: ndarray of float [n_rgb], optional, default=None
        Contains values from 0 to 1 to dictate the location of each color.
    bit: boolean, optional, default=False
        Defines if colors are in 8-bit [0 to 255] (True) or arithmetic [0 to 1] (False)

    Returns
    -------
    cmap: mpl.colors instance
        Colormap
    """

    bit_rgb = np.linspace(0, 1, 256)
    if position is None:
        position = np.linspace(0, 1, len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap


def plot_sobol_indices(sobol_rel_order_mean, sobol_rel_1st_order_mean, fn_plot, random_vars):
    """
    Plot the Sobol indices into different sub-plots.

    plot_sobol_indices(sobol_rel_order_mean, sobol_rel_1st_order_mean, fn_plot, random_vars)

    Parameters
    ----------
    sobol_rel_order_mean: ndarray of float [n_sobol]
        Average proportion of the Sobol indices of the different order to the total variance (1st, 2nd, etc..,)
        over all output quantities
    sobol_rel_1st_order_mean: ndarray of float [dim]
        Average proportion of the random variables of the 1st order Sobol indices to the total variance over all
        output quantities
    fn_plot: str
        Filename of plot
    random_vars: [dim] list of str
        String labels of the random variables
    """

    # combine parameters < "perc_limit_show" in %
    perc_limit_show = 0.03

    # set the global colors
    mpl.rcParams['text.color'] = '000000'
    mpl.rcParams['figure.facecolor'] = '111111'

    # set a global style
    plt.style.use('seaborn-talk')

    cmap = plt.cm.rainbow

    # make pie plot of order ratios
    labels = ['order=' + str(i) for i in range(1, len(sobol_rel_order_mean) + 1)]
    mask = np.where(sobol_rel_order_mean >= perc_limit_show)[0]
    mask_not = np.where(sobol_rel_order_mean < perc_limit_show)[0]
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

    # make pie plot of 1st order parameter ratios
    mask = np.where(sobol_rel_1st_order_mean >= perc_limit_show)[0]
    mask_not = np.where(sobol_rel_1st_order_mean < perc_limit_show)[0]
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


def plot_2d_grid(coords, weights=None, fn_plot=None):
    """
    Plot 2D grid and save it as fn_plot.pdf

    Parameters
    ----------
    coords: ndarray of float [n_grid, 2]
        Grid points
    weights: ndarray of float [n_grid], optional, default=None
        Integration weights
    fn_plot: str
        Filename of plot so save (.pdf)

    Returns
    -------
    <file> .pdf file
        Plot of grid-points
    """

    if weights is not None:
        weights = np.abs(weights)

    mpl.rc('text', usetex=True)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, squeeze=True, figsize=(5.5, 5))
    ax1.scatter(coords[:, 0], coords[:, 1], s=weights)
    ax1.grid()
    ax1.set_xlabel('$x_1$', fontsize=16)
    ax1.set_ylabel('$x_2$', fontsize=16)

    fn = os.path.splitext(fn_plot)[0]
    plt.savefig(fn, facecolor='#ffffff', format="pdf")


def plot_beta_pdf_fit(data, a_beta, b_beta, p_beta, q_beta, a_uni=None, b_uni=None,
                      interactive=True, fn_plot=None, xlabel="$x$", ylabel="$p(x)$"):
    """
    Plot data, fitted beta pdf (and corresponding uniform) distribution

    Parameters
    ----------
    data: ndarray of float
        Data to fit beta distribution on
    a_beta: float
        Lower limit of beta distribution
    b_beta: float
        Upper limit of beta distribution
    p_beta: float
        First shape parameter of beta distribution
    q_beta: float
        Second shape parameter of beta distribution
    a_uni: float (optional)
        Lower limit of uniform distribution
    b_uni: float (optional)
        Upper limit of uniform distribution
    interactive: bool, default = True
        Show plot (True/False)
    fn_plot:
        Filename of plot so save (as .png and .pdf)
    xlabel: str (optional)
        Label of x-axis
    ylabel: str (optional)
        Label of y-axis

    Returns
    -------
    <file> .png and .pdf files
        Plots
    """
    #if not interactive:
    #    plt.ioff()
    #else:
    #    plt.ion()

    plt.figure(1)
    plt.clf()
    plt.rc('text', usetex=True)
    plt.rc('font', size=18)
    ax = plt.gca()
    # legendtext = [r"e-pdf", r"$\beta$-pdf"]
    legendtext = [r"$\beta$-pdf"]

    # plot histogram of data
    n, bins, patches = plt.hist(data, bins=16, normed=1, color=[1, 1, 0.6], alpha=0.5)

    # plot beta pdf (kernel density estimate)
    # plt.plot(kde_x, kde_y, 'r--', linewidth=2)

    # plot beta pdf (fitted)
    beta_x = np.linspace(a_beta, b_beta, 100)
    beta_y = scipy.stats.beta.pdf(beta_x, p_beta, q_beta, loc=a_beta, scale=b_beta - a_beta)

    plt.plot(beta_x, beta_y, linewidth=2, color=[0, 0, 1])

    # plot uniform pdf
    uni_y = 0
    if a_uni is not None and b_uni is not None:
        uni_x = np.hstack([a_beta, a_uni - 1E-6 * (b_uni - a_uni),
                           np.linspace(a_uni, b_uni, 100), b_uni + 1E-6 * (b_uni - a_uni), b_beta])
        uni_y = np.hstack([0, 0, 1.0 / (b_uni - a_uni) * np.ones(100), 0, 0])
        plt.plot(uni_x, uni_y, linewidth=2, color='r')
        legendtext.append("u-pdf")

        # configure plot
    plt.legend(legendtext, fontsize=18, loc="upper left")
    plt.grid(True)
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    ax.set_xlim(a_beta - 0.05 * (b_beta - a_beta), b_beta + 0.05 * (b_beta - a_beta))
    # ax.set_ylim(0, 1.1 * max([max(n), max(beta_y[np.logical_not(beta_y == np.inf)]), max(uni_y)]))

    if interactive > 0:
        plt.show()

    # save plot
    if fn_plot is not None:
        plt.savefig(fn_plot + ".pdf", format='pdf', bbox_inches='tight', pad_inches=0.01 * 4)
        plt.savefig(fn_plot + ".png", format='png', bbox_inches='tight', pad_inches=0.01 * 4, dpi=600)
