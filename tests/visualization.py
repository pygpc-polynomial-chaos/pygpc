# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


class Visualization:

    figNo = 0;
    horizontalPadding = 0.4
    fontSizeLabel     = 20
    fontSizeTitle     = 12
    graphLineWidth    = 2

    #
    # creates a new visualization in a new window
    #   any added subcharts will be addded to this window
    #
    # inputs:
    #   _dims - size of the newly created window
    #
    def __init__(self, _dims=[10,10]):
        self.fig = plt.figure(Visualization.figNo, figsize=(_dims['x'],_dims['y']))
        Visualization.figNo += 1
        plt.subplots_adjust(hspace=Visualization.horizontalPadding) # add some horizontal spacing to avoid overlap with labels


    def show(self):
       plt.show()

    #
    # adds a new subplot to the current visualization
    # -> multiple graphs can be overlaid onto one chart (e.g. scatterplot over heatmap)
    # 
    # inputs:
    #   _layoutID - 3-digit integer: denoting the position of the graph in figure
    #               (xyn : 'x'=width, 'y'=height of grid, 'n'=position within grid)
    #
    def createNewChart(self, _layoutID=None ):
        self.fig.add_subplot(_layoutID)

    #
    # helper function with basic commands that are common in every plot
    # _title      - string: title of the plot
    # _labels     - dictionary: 'x'(string) name of x-axis, 'y' (string) name of y-axis
    # _[x|y]Lim   - limits for the function argument|value
    #
    def basicSubPlot(self, _title, _labels, _xLim, _yLim):
        plt.title(_title,fontsize=Visualization.fontSizeTitle)  
        plt.ylabel(_labels['y'],fontsize=Visualization.fontSizeLabel)
        plt.xlabel(_labels['x'],fontsize=Visualization.fontSizeLabel)
        
        ax = plt.gca()
        if _xLim is not None:
            ax.set_xlim(_xLim[0],_xLim[1])
        if _yLim is not None:
            ax.set_ylim(_yLim[0],_yLim[1])
        

    #
    # draw a 1D line graph into the current figure
    #
    # inputs: 
    #  _title      - string: title of the graph
    #  _labels     - dictionary: 'x'(string) name of x-axis, 'y' (string) name of y-axis
    #  _data       - dictionary: with two arrays of 'pointSets' and their corresponding 'names'
    #  [x|y]Lim   - 2 x 1 array: limits for the x|y axis, [0]=lower bound, [1]=upper bound
    #
    def addLinePlot(self,  _title, _labels, _data, xLim=None, yLim=None):
        self.basicSubPlot(_title,_labels,_xLim=xLim,_yLim=yLim)

        for p in _data['pointSets']:
            plt.plot(p['x'],p['y'],linewidth = Visualization.graphLineWidth)

        plt.legend(_data['names'], loc = "upper left")
        plt.grid()


    #
    # draw a 2D heatmap into the current figure
    #
    # inputs: 
    #  _title      - string: title of the graph
    #  _labels     - dictionary: 'x'(string) name of x-axis, 'y' (string) name of y-axis
    #  _gridpoint  - 2 x n array: containing an the arrays of x|y positions of the grid points
    #  _datapoints - 1D array_ of the data-points that should be positoned into the grid
    #  [x|y]Lim    - 2 x 1 array: limits for the x|y axis, [0]=lower bound, [1]=upper bound
    #  vLim        - 2 x 1 array containing the upper[1]|lower[0] bound of the color-scale
    # cmap         - string: the colormap to use 
    #
    def addHeatMap(self, _title, _labels, _gridpoints, _datapoints, vLim=[None,None], xLim=None, yLim=None, colorMap=None):
        self.basicSubPlot(_title, _labels, _xLim=xLim, _yLim=yLim);

        plt.pcolormesh(_gridpoints[0], _gridpoints[1], _datapoints, vmin=vLim[0], vmax=vLim[1], cmap=colorMap)

        plt.colorbar()

    #
    # draws a scatterplot onto the current chart
    #
    # inputs:
    #   _shape          - dictionary: 'x'=positions on x-axis, 'y'= positions on y-axis
    #   _plotSize       - ??
    #   _colorSequence  - ??
    #   _colorMap       - string: the colormap to use
    #  vLim             - 2 x 1 array containing the upper[1]|lower[0] bound of the color-scale
    #
    def addScatterPlot(self, _shape, _plotSize, _colorSequence, colorMap=None, vLim=[None,None]):
        plt.scatter( _shape['x'], _shape['y'], s=_plotSize, c=_colorSequence, vmin=vLim[0], vmax=vLim[1], cmap=colorMap)        
