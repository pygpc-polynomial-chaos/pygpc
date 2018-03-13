# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:04:20 2016

@author: Konstantin Weise
"""

#% gPC example for 2 random variables using different grids and methods


import pygpc
import scipy.stats
import numpy as np
import PyRates
import matplotlib.pyplot as plt

from visualization import Visualization
#%% simulation parameters
DIM = 2                                # number of random variables
testfun  = "peaks"                     # test function
pdftype  = ["beta", "beta"]            # Type of input PDFs in each dimension
p        = [3, 4]                      # first shape parameter of beta distribution (also often: alpha)
q        = [3, 3]                    # second shape parameter of beta distribution (also often: beta)
pdfshape = [p, q]
a        = [0.5, -0.5]                 # lower bounds of random variables
b        = [1, 0]                    # upper bounds of random variables
limits   = [a, b]
order    = [10, 10]                      # expansion order in each dimension
order_max = 7                          # maximum order in all dimensions
interaction_order = 2                  # interaction order between variables

# random grid parameters:
N_rand = int(1.5*pygpc.calc_Nc(DIM,order[0])) # number of grid points

# Sparse grid parameters:
gridtype_sparse = ["jacobi", "jacobi"] # type of quadrature rule in each dimension
level    = [4,4]                       # level of sparse grid in each dimension
level_max = max(level) + DIM - 1       # maximum level in all dimensions
order_sequence_type = 'exp'            # exponential: order = 2**level + 1

# Tensored grid parameters:
N_tens = [order[0], order[1]]          # number of grid points in each dimension
gridtype_tens = ["jacobi", "jacobi"]   # type of quadrature rule in each dimension

# Monte Carlo simulations (bruteforce for comparison)
N_mc = int(1E5)                        # number of random samples 

# generate grids for computations
grid_rand   = pygpc.randomgrid(pdftype,pdfshape,limits,N_rand)
grid_mc     = pygpc.randomgrid(pdftype,pdfshape,limits,N_mc)

grid_SG     = pygpc.grid.sparsegrid(pdftype,gridtype_sparse,pdfshape,limits,level,level_max,interaction_order, order_sequence_type)
grid_tens   = pygpc.grid.tensgrid(pdftype, gridtype_tens, pdfshape, limits, N_tens)

#% generate gpc objects
gpc_reg  = pygpc.reg( pdftype, pdfshape, limits, order, order_max, interaction_order, grid_rand)
gpc_tens = pygpc.quad(pdftype, pdfshape, limits, order, order_max, interaction_order, grid_tens)
gpc_SG   = pygpc.quad(pdftype, pdfshape, limits, order, order_max, interaction_order, grid_SG)


#% evaluate model function on different grids
data_rand = pygpc.tf.peaks(grid_rand.coords)
data_mc   = pygpc.tf.peaks(grid_mc.coords)
data_tens = pygpc.tf.peaks(grid_tens.coords)
data_SG   = pygpc.tf.peaks(grid_SG.coords)
data_mc   = pygpc.tf.peaks(grid_mc.coords)

#% determine gpc coefficients
coeffs_reg  = gpc_reg.expand(data_rand)
coeffs_tens = gpc_tens.expand(data_tens)
coeffs_SG   = gpc_SG.expand(data_SG)


# perform postprocessing
print "Calculating mean ..."
out_mean_reg  = gpc_reg.mean(coeffs_reg)
out_mean_tens = gpc_tens.mean(coeffs_tens)
out_mean_SG   = gpc_SG.mean(coeffs_SG)
out_mean_mc   = np.mean(data_mc)

print "Calculating standard deviation ..."
out_std_reg  = gpc_reg.std(coeffs_reg)
out_std_tens = gpc_tens.std(coeffs_tens)
out_std_SG   = gpc_SG.std(coeffs_SG)
out_std_mc   = np.std(data_mc)

print "Calculating sobol coefficients ..."
out_sobol_reg, out_sobol_idx_reg    = gpc_reg.sobol(coeffs_reg)
out_sobol_tens, out_sobol_idx_tens  = gpc_tens.sobol(coeffs_tens)
out_sobol_SG, out_sobol_idx_SG      = gpc_SG.sobol(coeffs_SG)

print "Calculating global sensitivity indices ..."
out_globalsens_reg = gpc_reg.globalsens(coeffs_reg)
out_globalsens_tens = gpc_tens.globalsens(coeffs_tens)
out_globalsens_SG = gpc_SG.globalsens(coeffs_SG)

print "Calculating output PDFs ..."
pdf_x_reg, pdf_y_reg   = gpc_reg.pdf(coeffs_reg,N_mc)
pdf_x_tens, pdf_y_tens = gpc_tens.pdf(coeffs_tens,N_mc)
pdf_x_SG, pdf_y_SG     = gpc_SG.pdf(coeffs_SG,N_mc)

kde_mc = scipy.stats.gaussian_kde(data_mc.transpose(),bw_method=0.1/data_mc.std(ddof=1))
pdf_x_mc = np.linspace(data_mc.min(), data_mc.max(), 100)
pdf_y_mc = kde_mc(pdf_x_mc)

print "Calculate input PDFs ..."
x1_pdf = np.linspace(a[0]-1E-6,b[0]+1E-6,101)
x2_pdf = np.linspace(a[1]-1E-6,b[1]+1E-6,101)
px1_pdf = scipy.stats.beta.pdf(np.linspace(-1E-6,1+1E-6,101),p[0],q[0])/(b[0]-a[0])
px2_pdf = scipy.stats.beta.pdf(np.linspace(-1E-6,1+1E-6,101),p[1],q[1])/(b[1]-a[1])

print "Derive model function in uniform grid for comparison ..."
X1, X2  = np.meshgrid(np.linspace(a[0],b[0],201), np.linspace(a[1],b[1],101))
x1 = np.reshape(X1,X1.shape[0]*X1.shape[1])
x2 = np.reshape(X2,X2.shape[0]*X2.shape[1])
X1_norm, X2_norm  = np.meshgrid(np.linspace(-1,1,201), np.linspace(-1,1,101))
x1_norm = np.reshape(X1_norm,X1.shape[0]*X1.shape[1])
x2_norm = np.reshape(X2_norm,X2.shape[0]*X2.shape[1])

y_original = np.reshape(pygpc.tf.peaks(np.vstack([x1,x2]).transpose()),[X1.shape[0],X1.shape[1]])
y_gpc_reg  = np.reshape(gpc_reg.evaluate(coeffs_reg, np.vstack([x1_norm,x2_norm]).transpose()),[X1.shape[0],X1.shape[1]])
y_gpc_tens = np.reshape(gpc_tens.evaluate(coeffs_tens, np.vstack([x1_norm,x2_norm]).transpose()),[X1.shape[0],X1.shape[1]])
y_gpc_SG   = np.reshape(gpc_SG.evaluate(coeffs_SG, np.vstack([x1_norm,x2_norm]).transpose()),[X1.shape[0],X1.shape[1]])

########### PLOTTING ################
v1 = Visualization({'x':20,'y':15})

# input pdfs
inputPDFdata = {
    'pointSets' : [{ 'x' : x1_pdf,'y' : px1_pdf},{ 'x' : x2_pdf,'y' : px2_pdf}],
    'names' : ['$x_1$','$x_2$']
}
outputPDFdata = {
    'pointSets' : [{'x' : pdf_x_reg, 'y' : pdf_y_reg}, {'x' : pdf_x_tens, 'y' : pdf_y_tens}, {'x' : pdf_x_SG,'y' : pdf_y_SG}, {'x' : pdf_x_mc, 'y' : pdf_y_mc}],
    'names' : ['Random grid', 'Tensored grid', 'sparse grid', 'monte carlo']
}

v1.createNewChart(331);
v1.addLinePlot('PDFs of input variables',{'x':'$p(x_i)$','y':'$x_i$'},inputPDFdata, yLim=[0,1.2*np.max([px1_pdf,px2_pdf])])

v1.createNewChart(332);
v1.addHeatMap('Original model function', {'x':'$x_1$','y':'$x_2$'}, [X1,X2], y_original, xLim=[a[0],b[0]], yLim=[a[1],b[1]])

v1.createNewChart(333);
v1.addLinePlot('PDFs of output variables', {'x':'$y$','y':'$p(y)$'}, outputPDFdata, xLim=[pdf_x_mc.min(),pdf_x_mc.max()], yLim=[0,1.2*pdf_y_mc.max()])

v1.createNewChart(334);
v1.addHeatMap('Random LHS grid: No. of points: {}'.format(grid_rand.coords.shape[0]), {'x':'$x_1$','y':'$x_2$'}, [X1,X2], y_gpc_reg, vLim=[y_original.min(), y_original.max()], xLim=[a[0],b[0]], yLim=[a[1],b[1]])
v1.addScatterPlot({ 'x' : grid_rand.coords[:,0], 'y' : grid_rand.coords[:,1]},_plotSize=30, _colorSequence='w')

v1.createNewChart(335);
v1.addHeatMap('Tensored grid: No. of points: {}'.format(grid_tens.coords.shape[0]), {'x':'$x_1$','y':'$x_2$'}, [X1,X2], y_gpc_tens, vLim=[y_original.min(), y_original.max()], xLim=[a[0],b[0]], yLim=[a[1],b[1]])
v1.addScatterPlot( { 'x' : grid_tens.coords[:,0], 'y' : grid_tens.coords[:,1]}, _plotSize=grid_tens.weights/grid_tens.weights.max()*75, _colorSequence=(grid_tens.weights>=0).astype(int)*2-1, colorMap='gray', vLim=[-grid_tens.weights.max(),None])

v1.createNewChart(336);
v1.addHeatMap('Sparse grid: No. of points: {}'.format(grid_SG.coords.shape[0]), {'x':'$x_1$','y':'$x_2$'}, [X1,X2], y_gpc_SG, vLim=[y_original.min(), y_original.max()], xLim=[a[0],b[0]], yLim=[a[1],b[1]])
v1.addScatterPlot( {'x' : grid_SG.coords[:,0], 'y' : grid_SG.coords[:,1]}, _plotSize=np.abs(grid_SG.weights)/np.abs(grid_SG.weights).max()*75, _colorSequence=(grid_SG.weights>=0).astype(int)*2-1, colorMap='gray' )

v1.createNewChart(337);
v1.addHeatMap('Random LHS grid: model error', {'x':'$x_1$','y':'$x_2$'}, [X1,X2], y_original-y_gpc_reg, vLim=[-np.max(y_original-y_gpc_reg), None], colorMap='RdBu_r',xLim=[a[0],b[0]], yLim=[a[1],b[1]])
v1.addScatterPlot({'x' : grid_rand.coords[:,0], 'y' : grid_rand.coords[:,1]}, _colorSequence='w', _plotSize=30)

v1.createNewChart(338);
v1.addHeatMap('Tensored grid: model error', {'x':'$x_1$','y':'$x_2$'}, [X1,X2], y_original-y_gpc_tens, vLim=[-np.max(y_original-y_gpc_tens), None], colorMap='RdBu_r',xLim=[a[0],b[0]], yLim=[a[1],b[1]])
v1.addScatterPlot( { 'x' : grid_tens.coords[:,0], 'y' : grid_tens.coords[:,1]}, _plotSize=grid_tens.weights/grid_tens.weights.max()*75, _colorSequence=(grid_tens.weights>=0).astype(int)*2-1, colorMap='gray', vLim=[-grid_tens.weights.max(),None])

v1.createNewChart(339);
v1.addHeatMap('Sparse grid: model error', {'x':'$x_1$','y':'$x_2$'}, [X1,X2], y_original-y_gpc_SG, vLim=[-np.max(y_original-y_gpc_SG), None], colorMap='RdBu_r',xLim=[a[0],b[0]], yLim=[a[1],b[1]])
v1.addScatterPlot( {'x' : grid_SG.coords[:,0], 'y' : grid_SG.coords[:,1]}, _plotSize=np.abs(grid_SG.weights)/np.abs(grid_SG.weights).max()*75, _colorSequence=(grid_SG.weights>=0).astype(int)*2-1, colorMap='gray' )

v1.show()
