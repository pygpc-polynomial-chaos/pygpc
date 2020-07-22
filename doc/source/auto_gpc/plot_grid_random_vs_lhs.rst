.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_gpc_plot_grid_random_vs_lhs.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_gpc_plot_grid_random_vs_lhs.py:


Grid: Random vs LHS
===================

Choosing a sampling scheme
--------------------------

To calculate the coefficients of the gPC matrix, a number of random samples needs to be
picked to represent the propability space :math:`\Theta` and enable descrete evaluations of the
polynomials. As for the computation of the coefficients, the input parameters :math:`\mathbf{\xi}`
can be sampled in a number of different ways. In **pygpc** the grid :math:`\mathcal{G}` for this
application is constructed in `pygpc/Grid.py <../../../../pygpc/Grid.py>`_.

Random Sampling
^^^^^^^^^^^^^^^
In the case of random sampling the samples will be randomly from their Probability Density Function (PDF)
:math:`f(\xi)`.

Latin Hypercube Sampling (LHS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To increase the information of each individual sampling point and to prevent undersampling, LHS is a simple
alternative to enhance the space-filling properties of the sampling scheme first established by
McKay et al. (2000).

.. [1] McKay, M. D., Beckman, R. J., & Conover, W. J. (2000). A comparison of three methods for selecting
   values of input variables in the analysis of output from a computer code. Technometrics, 42(1), 55-61.

To draw :math:`n` independent samples from a number of :math:`d`-dimensional parameters
a matrix :math:`\Pi` is constructed with

.. math::

    \pi_{ij} = \frac{p_{ij} - u}{n}

where :math:`P` is a :math:`d \times n` matrix of randomly perturbed integers
:math:`p_{ij} \in \mathbb{N}, {1,...,n}` and u is uniform random number :math:`u \in [0,1]`.

Constructing a simple LHS design
--------------------------------
We are going to create a simple LHS design for 2 random variables with 5 sampling points:
sphinx_gallery_thumbnail_number = 3:


.. code-block:: default


    # Windows users have to encapsulate the code into a main function to avoid multiprocessing errors.
    # def main():

    import pygpc
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import OrderedDict

    # define parameters
    parameters = OrderedDict()
    parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
    parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])

    # create LHS grid
    grid = pygpc.LHS(parameters_random=parameters, n_grid=25, options={"seed": 1})

    # plot
    fig = plt.figure(figsize=(4, 4))
    plt.scatter(grid.coords_norm[:, 0], grid.coords_norm[:, 1])
    plt.xlabel("$x_1$", fontsize=12)
    plt.ylabel("$x_2$", fontsize=12)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.tight_layout()




.. image:: /auto_gpc/images/sphx_glr_plot_grid_random_vs_lhs_001.png
    :class: sphx-glr-single-img





LHS Designs can further be improved upon, since the pseudo-random sampling procedure
can lead to samples with high spurious correlation and the space filling capability
in itself leaves room for improvement, some optimization criteria have been found to
be adequate for compensating the initial designs shortcomings.

Optimization Criteria of LHS designs
------------------------------------
Spearman Rank Correlation
^^^^^^^^^^^^^^^^^^^^^^^^^
For a sample size of :math:`n` the scores of each variable are converted to their Ranks :math:`rg_{X_i}`
the Spearman Rank Correlation Coefficient is then the Pearson Correlation Coefficient applied to the rank
variables :math:`rg_{X_i}`:

.. math::

    r_s = \rho_{rg_{X_i}, rg_{X_j}} = \frac{cov(rg_{X_i}, rg_{X_j})}{\sigma_{rg_{X_i}} \sigma_{rg_{X_i}}}

where :math:`\rho` is the pearson correlation coefficient, :math:`\sigma` is the standard deviation
and :math:`cov` is the covariance of the rank variables

Maximum-Minimal-Distance
^^^^^^^^^^^^^^^^^^^^^^^^
For creating a so called maximin distance design that maximizes the minimum inter-site distance, proposed by
Johnson et al.

.. math::

    \min_{1 \leqslant i, j \leqslant n, i \neq j} d(x_i,x_j),

where :math:`d` is the distance between two samples :math:`x_i` and :math:`x_j` and
:math:`n` is the number of samples in a sample design.

.. math::

    d(x_i,x_j) = d_ij = [ \sum_{k=1}^{m}|x_ik - x_jk| ^ t]^\frac{1}{t}, t \in {1,2}

There is however a more elegant way of computing this optimization criterion as shown by Morris and Mitchell (1995),
called the :math:`\varphi_P` criterion.

.. math::

    \min\varphi_P \quad \text{subject to} \quad \varphi_P = [ \sum_{k = 1} ^ {s} J_id_i  ^ p]^\frac{1}{p},

where :math:`s` is the number of distinct distances, :math:`J` is an vector of indices of the distances
and :math:`p` is an integer. With a very large :math:`p` this criterion is equivalent to the maximin criterion

.. Morris, M. D. and Mitchell, T. J. ( (1995). Exploratory Designs for Computer Experiments.J. Statist. Plann.
   Inference 43, 381-402.

LHS with enhanced stochastic evolutionary algorithm (ESE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To achieve optimized designs with a more stable method and possibly quicker then by simply evaluating
the criteria over a number of repetitions **pygpc** can use an ESE for achieving sufficient
:math:`\varphi_P`-value. This algorithm is more appealing in its efficacy and proves to
[sth about the resulting error or std in a low sample size].
This method originated from Jin et al. (2005).

.. Jin, R., Chen, W., Sudjianto, A. (2005). An efficient algorithm for constructing optimal
   design of computer experiments. Journal of statistical planning and inference, 134(1), 268-287.

Comparison between a standard random grid and different LHS designs
-------------------------------------------------------------------


.. code-block:: default


    from scipy.stats import spearmanr
    import seaborn as sns

    # define parameters
    parameters = OrderedDict()
    parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
    parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])

    # define grids for each criteria
    grids = []
    grids.append(pygpc.Random(parameters_random=parameters, n_grid=30, options={"seed": 1}))
    grids.append(pygpc.LHS(parameters_random=parameters, n_grid=30, options={"criterion": None, "seed": 1}))
    grids.append(pygpc.LHS(parameters_random=parameters, n_grid=30, options={"criterion": "corr", "seed": 1}))
    grids.append(pygpc.LHS(parameters_random=parameters, n_grid=30, options={"criterion": "maximin", "seed": 1}))
    grids.append(pygpc.LHS(parameters_random=parameters, n_grid=30, options={"criterion": "ese", "seed": 1}))

    # calculate criteria
    corrs = []
    phis = []
    name = []
    variables = []

    for i_g, g in enumerate(grids):
        corr = spearmanr(g.coords_norm[:, 0], g.coords_norm[:, 1])[0]
        corrs.append(corr)
        phis.append(pygpc.PhiP(g.coords_norm))

    variables.append(corrs)
    name.append('corr')
    variables.append(phis)
    name.append('phi')

    # plot results
    fig = plt.figure(figsize=(16, 3))
    titles = ['Random', 'LHS (standard)', 'LHS (corr opt)', 'LHS (Phi-P opt)', 'LHS (ESE)']

    for i_g, g in enumerate(grids):
        text = name[0] + ' = {:0.2f} '.format(variables[0][i_g]) + "\n" + \
               name[1] + ' = {:0.2f}'.format(variables[1][i_g])
        plot_index = 151 + i_g
        plt.gcf().text((0.15 + i_g * 0.16), 0.08, text, fontsize=14)
        plt.subplot(plot_index)
        plt.scatter(g.coords_norm[:, 0], g.coords_norm[:, 1], color=sns.color_palette("bright", 5)[i_g])
        plt.title(titles[i_g])
        plt.gca().set_aspect('equal', adjustable='box')
    plt.subplots_adjust(bottom=0.3)




.. image:: /auto_gpc/images/sphx_glr_plot_grid_random_vs_lhs_002.png
    :class: sphx-glr-single-img





The initial LHS (standard) has already good space filling properties compared
to the random sampling scheme (eg. less under sampled areas and less clustered areas,
visually and quantitatively represented by the optimization criteria). The LHS (ESE)
shows the best correlation and :math:`\varphi_P` criterion.

Convergence and stability comparison in gPC
-------------------------------------------
We are going to compare the different grids in a practical gPC example considering the Ishigami function.
We are going to conduct gPC analysis for different approximation orders (grid sizes).
Because we are working with random grids, we are interested in (i) the rate of convergence
and (ii) the stability of the convergence. For that reason, we will repeat the analysis several times.

Setting up the problem
^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default

    import pygpc
    import numpy as np
    from collections import OrderedDict
    import matplotlib.pyplot as plt

    # grids to compare
    grids = [pygpc.Random, pygpc.LHS, pygpc.LHS, pygpc.LHS, pygpc.LHS]
    grids_options = [{"seed": 1},
                     {"criterion": None, "seed": 1},
                     {"criterion": "corr", "seed": 1},
                     {"criterion": "maximin", "seed": 1},
                     {"criterion": "ese", "seed": 1}]
    grid_legend = ["Random", "LHS (standard)", "LHS (corr opt)", "LHS (Phi-P opt)", "LHS (ESE)"]
    n_grid = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    repetitions = 3

    err = np.zeros((len(grids), len(n_grid), repetitions))

    # Model
    model = pygpc.testfunctions.Ishigami()

    # Problem
    parameters = OrderedDict()
    parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
    parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
    parameters["x3"] = 0.
    parameters["a"] = 7.
    parameters["b"] = 0.1

    problem = pygpc.Problem(model, parameters)

    # gPC options
    options = dict()
    options["method"] = "reg"
    options["solver"] = "LarsLasso"
    options["interaction_order"] = problem.dim
    options["order_max_norm"] = 1
    options["n_cpu"] = 0
    options["adaptive_sampling"] = False
    options["gradient_enhanced"] = False
    options["fn_results"] = None
    options["error_type"] = "nrmsd"
    options["error_norm"] = "relative"
    options["matrix_ratio"] = None
    options["eps"] = 0.001
    options["backend"] = "omp"
    options["order"] = [12] * problem.dim
    options["order_max"] = 12








Running the analysis
^^^^^^^^^^^^^^^^^^^^


.. code-block:: default

    for i_g, g in enumerate(grids):
        for i_n_g, n_g in enumerate(n_grid):
            for i_n, n in enumerate(range(repetitions)):

                options["grid"] = g
                options["grid_options"] = grids_options[i_g]
                options["n_grid"] = n_g

                # define algorithm
                algorithm = pygpc.Static(problem=problem, options=options)

                # Initialize gPC Session
                session = pygpc.Session(algorithm=algorithm)

                # run gPC session
                session, coeffs, results = session.run()

                err[i_g, i_n_g, i_n] = pygpc.validate_gpc_mc(session=session,
                                                             coeffs=coeffs,
                                                             n_samples=int(1e4),
                                                             n_cpu=0,
                                                             output_idx=0,
                                                             fn_out=None,
                                                             plot=False)

    err_mean = np.mean(err, axis=2)
    err_std = np.std(err, axis=2)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.000331878662109375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.4260202559264827
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0003032684326171875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.422094186235923
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0003256797790527344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.4262335885572447
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00032138824462890625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.032034987306079676
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0004220008850097656 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.0317765261472653
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00031638145446777344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.03172299128251933
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003197193145751953 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.0007214028567693966
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003228187561035156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.0007267414297315524
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00032591819763183594 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.0007145379842093909
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.000316619873046875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 8.47189101420577e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0004918575286865234 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 8.641668464166989e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0003066062927246094 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 8.387042191632727e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0004792213439941406 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.584474889775224e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0003361701965332031 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.751010064445245e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0003609657287597656 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.6118692556715895e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0003199577331542969 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.462449293110859e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00032019615173339844 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.5259871556401493e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0003552436828613281 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.437060316682357e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003178119659423828 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.480643303352494e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.00038123130798339844 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.5891326719088285e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003097057342529297 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.5648482718662643e-05
    Performing 80 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 80 [                                        ] 1.2%
    Total parallel function evaluation: 0.0003275871276855469 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.006488321279553e-05
    Performing 80 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 80 [                                        ] 1.2%
    Total parallel function evaluation: 0.0004432201385498047 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.0567497231897054e-05
    Performing 80 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 80 [                                        ] 1.2%
    Total parallel function evaluation: 0.00030803680419921875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.997580569695608e-05
    Performing 90 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003657341003417969 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.768109944139084e-05
    Performing 90 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003752708435058594 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.7682762073084395e-05
    Performing 90 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00032138824462890625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.7666967374968137e-05
    Performing 100 simulations!
    It/Sub-it: 12/2 Performing simulation 001 from 100 [                                        ] 1.0%
    Total parallel function evaluation: 0.0005676746368408203 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.736115109706e-05
    Performing 100 simulations!
    It/Sub-it: 12/2 Performing simulation 001 from 100 [                                        ] 1.0%
    Total parallel function evaluation: 0.0003104209899902344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.7291421506679874e-05
    Performing 100 simulations!
    It/Sub-it: 12/2 Performing simulation 001 from 100 [                                        ] 1.0%
    Total parallel function evaluation: 0.0003333091735839844 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.728906493688456e-05
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.00031828880310058594 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.33784712798249866
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0002949237823486328 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.33875586404443897
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0004134178161621094 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.3368284088899952
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0004162788391113281 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.12946851339711798
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00029015541076660156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.13475203039050104
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0003151893615722656 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.13202576255035775
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002830028533935547 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.02133634939674418
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00029397010803222656 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.02140166085045133
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00041985511779785156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.02123634641465256
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00029468536376953125 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.4400079913807906e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0002918243408203125 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.4098770540174096e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.000286102294921875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.400851217317341e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00039196014404296875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.25582749541967e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0002913475036621094 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.252253274542509e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0002999305725097656 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.241449164788153e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0003495216369628906 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.0433857529797296e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0005078315734863281 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.06015703748343e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00041747093200683594 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.1440995862787676e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003077983856201172 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.154068087856755e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.00028514862060546875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.1000072705144816e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003788471221923828 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.073021107667178e-05
    Performing 80 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 80 [                                        ] 1.2%
    Total parallel function evaluation: 0.00028824806213378906 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.504562112366034e-05
    Performing 80 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 80 [                                        ] 1.2%
    Total parallel function evaluation: 0.0002980232238769531 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.4290277280776734e-05
    Performing 80 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 80 [                                        ] 1.2%
    Total parallel function evaluation: 0.00030612945556640625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.4877814032678e-05
    Performing 90 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00029277801513671875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.723680730531488e-05
    Performing 90 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00036835670471191406 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.6839311448223464e-05
    Performing 90 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003235340118408203 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.7398131253667044e-05
    Performing 100 simulations!
    It/Sub-it: 12/2 Performing simulation 001 from 100 [                                        ] 1.0%
    Total parallel function evaluation: 0.0002944469451904297 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.19676613524469e-05
    Performing 100 simulations!
    It/Sub-it: 12/2 Performing simulation 001 from 100 [                                        ] 1.0%
    Total parallel function evaluation: 0.0002949237823486328 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.249854338338043e-05
    Performing 100 simulations!
    It/Sub-it: 12/2 Performing simulation 001 from 100 [                                        ] 1.0%
    Total parallel function evaluation: 0.0002906322479248047 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.185677638262436e-05
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0002713203430175781 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.30981750927687335
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0003123283386230469 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.31459643983145374
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0002732276916503906 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.32304627997659596
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00027632713317871094 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.10126363174920293
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.000274658203125 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.10166494927048546
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002796649932861328 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.1023984181557152
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002701282501220703 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.00013203059593634828
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002741813659667969 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.0001299565394407675
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00030040740966796875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.00012771304743606644
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00027298927307128906 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 6.329476053799148e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0002765655517578125 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 6.0994245194195784e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0002753734588623047 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 6.603643099521361e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0002727508544921875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.904247675128103e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0002751350402832031 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.646073368391517e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00029659271240234375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.897615832956632e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00027370452880859375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.140418296364698e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0002734661102294922 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.022139749097405e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00027489662170410156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.144191851272719e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002791881561279297 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.482404749825158e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.00027942657470703125 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.648058913794643e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002751350402832031 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.816427387822623e-05
    Performing 80 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 80 [                                        ] 1.2%
    Total parallel function evaluation: 0.000278472900390625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.1931826089838607e-05
    Performing 80 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 80 [                                        ] 1.2%
    Total parallel function evaluation: 0.00027942657470703125 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.151187763190365e-05
    Performing 80 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 80 [                                        ] 1.2%
    Total parallel function evaluation: 0.0002753734588623047 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.185424731679513e-05
    Performing 90 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002810955047607422 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.429544232017448e-05
    Performing 90 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002777576446533203 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.404725485471132e-05
    Performing 90 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00027751922607421875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.428345344395716e-05
    Performing 100 simulations!
    It/Sub-it: 12/2 Performing simulation 001 from 100 [                                        ] 1.0%
    Total parallel function evaluation: 0.0002777576446533203 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.1666753431272197e-05
    Performing 100 simulations!
    It/Sub-it: 12/2 Performing simulation 001 from 100 [                                        ] 1.0%
    Total parallel function evaluation: 0.0002810955047607422 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.4085205109482075e-05
    Performing 100 simulations!
    It/Sub-it: 12/2 Performing simulation 001 from 100 [                                        ] 1.0%
    Total parallel function evaluation: 0.0002779960632324219 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.384551854024696e-05
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.00030517578125 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.1715548484114736
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.00028228759765625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.1694900332160488
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0002944469451904297 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.16922956618457013
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002827644348144531 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.04012620214132834
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00027751922607421875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.039799047946877525
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002779960632324219 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.03925414881878154
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00027751922607421875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.822498050185848e-05
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00027561187744140625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.9575581449128664e-05
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.000278472900390625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.879940319199146e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0002789497375488281 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.291037775639837e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0002734661102294922 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.242432801932218e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00027823448181152344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.201487667661247e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0002751350402832031 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.460847204055828e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00028395652770996094 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.447292303710735e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0002777576446533203 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.54550706237947e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0002834796905517578 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.8352113560132246e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0003185272216796875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.90179856034802e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0002803802490234375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.869841389218998e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002918243408203125 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.320900532261367e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002803802490234375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.373092756034332e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002849102020263672 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.418196377848246e-05
    Performing 80 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 80 [                                        ] 1.2%
    Total parallel function evaluation: 0.0002834796905517578 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.458004754222035e-05
    Performing 80 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 80 [                                        ] 1.2%
    Total parallel function evaluation: 0.00028133392333984375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.4450391380589244e-05
    Performing 80 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 80 [                                        ] 1.2%
    Total parallel function evaluation: 0.0002853870391845703 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.4100949062037226e-05
    Performing 90 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002846717834472656 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.6616446230966005e-05
    Performing 90 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002827644348144531 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.44710224149509e-05
    Performing 90 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00028204917907714844 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.7070331840531346e-05
    Performing 100 simulations!
    It/Sub-it: 12/2 Performing simulation 001 from 100 [                                        ] 1.0%
    Total parallel function evaluation: 0.0002880096435546875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.456005653481151e-05
    Performing 100 simulations!
    It/Sub-it: 12/2 Performing simulation 001 from 100 [                                        ] 1.0%
    Total parallel function evaluation: 0.0002846717834472656 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.2510779899802866e-05
    Performing 100 simulations!
    It/Sub-it: 12/2 Performing simulation 001 from 100 [                                        ] 1.0%
    Total parallel function evaluation: 0.0002865791320800781 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.3420524849124836e-05
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0002770423889160156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.40738533421903356
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0002815723419189453 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.4114913964194423
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0002868175506591797 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.4010143687766864
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002830028533935547 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.02127981397326073
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00027251243591308594 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.020969694549018137
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002796649932861328 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.02122873101530851
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002799034118652344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 6.035940045574909e-05
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002789497375488281 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.985468050973058e-05
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002753734588623047 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 6.0072261265130966e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0002849102020263672 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.66525688580501e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00028133392333984375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.605597267778891e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00027871131896972656 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.682235536433076e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00028204917907714844 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.993524408092161e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0002818107604980469 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.013498319642327e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0002796649932861328 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.965057114969233e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0002999305725097656 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.02940460360737e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00028061866760253906 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.085392953798704e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00028395652770996094 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.1349510132108315e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.00028228759765625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.535463256358389e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002856254577636719 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.5504655669156695e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002791881561279297 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.5044204571554146e-05
    Performing 80 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 80 [                                        ] 1.2%
    Total parallel function evaluation: 0.0002849102020263672 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.973348538823735e-05
    Performing 80 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 80 [                                        ] 1.2%
    Total parallel function evaluation: 0.0002810955047607422 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.838346695065073e-05
    Performing 80 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 80 [                                        ] 1.2%
    Total parallel function evaluation: 0.00028896331787109375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.9543366019067943e-05
    Performing 90 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00028586387634277344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.2508813465832435e-05
    Performing 90 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00028824806213378906 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.390645437458072e-05
    Performing 90 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003199577331542969 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.3176754560790086e-05
    Performing 100 simulations!
    It/Sub-it: 12/2 Performing simulation 001 from 100 [                                        ] 1.0%
    Total parallel function evaluation: 0.0002956390380859375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.502413871553474e-05
    Performing 100 simulations!
    It/Sub-it: 12/2 Performing simulation 001 from 100 [                                        ] 1.0%
    Total parallel function evaluation: 0.0002942085266113281 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.602261719235864e-05
    Performing 100 simulations!
    It/Sub-it: 12/2 Performing simulation 001 from 100 [                                        ] 1.0%
    Total parallel function evaluation: 0.00029158592224121094 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.4407153283322244e-05




Results
^^^^^^^
Even after a small set of repetitions the :math:`\varphi_P` optimizing ESE will produce
the best results regarding the aforementioned criteria, while also having less variation
in its pseudo-random design. Thus is it possible to half the the root-mean-squared error
:math:`\varepsilon` by using the ESE algorithm compared to completely random sampling the
grid points, while also having a consistently small standard deviation.


.. code-block:: default


    fig, ax = plt.subplots(1, 2, figsize=[12, 5])

    for i in range(len(grids)):
        ax[0].errorbar(n_grid, err_mean[i, :], err_std[i, :], capsize=3, elinewidth=.5)
        ax[1].plot(n_grid, err_std[i, :])

    for a in ax:
        a.legend(grid_legend)
        a.set_xlabel("$N_g$", fontsize=12)
        a.grid()

    ax[0].set_ylabel("$\epsilon$", fontsize=12)
    ax[1].set_ylabel("std($\epsilon$)", fontsize=12)

    ax[0].set_title("gPC error vs original model (mean and std)")
    _ = ax[1].set_title("gPC error vs original model (std)")


    # On Windows subprocesses will import (i.e. execute) the main module at start.
    # You need to insert an if __name__ == '__main__': guard in the main module to avoid
    # creating subprocesses recursively.
    #
    # if __name__ == '__main__':
    #     main()



.. image:: /auto_gpc/images/sphx_glr_plot_grid_random_vs_lhs_003.png
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 3 minutes  1.623 seconds)


.. _sphx_glr_download_auto_gpc_plot_grid_random_vs_lhs.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_grid_random_vs_lhs.py <plot_grid_random_vs_lhs.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_grid_random_vs_lhs.ipynb <plot_grid_random_vs_lhs.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
