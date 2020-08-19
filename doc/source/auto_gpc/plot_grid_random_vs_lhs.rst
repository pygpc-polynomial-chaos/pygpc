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
    grids_options = [{"seed": None},
                     {"criterion": None, "seed": None},
                     {"criterion": "corr", "seed": None},
                     {"criterion": "maximin", "seed": None},
                     {"criterion": "ese", "seed": None}]
    grid_legend = ["Random", "LHS (standard)", "LHS (corr opt)", "LHS (Phi-P opt)", "LHS (ESE)"]
    n_grid = [10, 20, 30, 40, 50, 60, 70]
    repetitions = 5

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
                                                             n_cpu=options["n_cpu"],
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
    Total parallel function evaluation: 0.0003485679626464844 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.19755365265299765
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.00029969215393066406 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.2945476760289986
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.00030541419982910156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.3464050390361741
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0002956390380859375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.17706765737224972
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.00030422210693359375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.38325856931194713
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0003063678741455078 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.12604037445812322
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00032782554626464844 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.11735421636764806
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00031304359436035156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.40663944858200735
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.000301361083984375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.041485463601710945
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00030541419982910156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.1412459978710678
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00039005279541015625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 6.854401553692866e-05
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003864765167236328 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.10182872889239372
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003058910369873047 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.00025602198162659765
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003592967987060547 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.00015048895556008492
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003123283386230469 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 7.02681490638806e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0005488395690917969 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.5306782112980576e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00034689903259277344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.864217429702191e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0003228187561035156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.159027811469723e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0004477500915527344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.00025340173161749027
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00031757354736328125 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 6.214541697068508e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00033736228942871094 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 8.029689834290589e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00031447410583496094 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.144351281684398e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00031495094299316406 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.25026354225531e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00032138824462890625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.5874062734349996e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.000331878662109375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.961213542892071e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0004730224609375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 8.084476183804018e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00033020973205566406 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 7.059031318747449e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00031185150146484375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.5556083147620654e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00038552284240722656 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.974085752300046e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0003483295440673828 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.3024903114834415e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003154277801513672 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.292115508161578e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003132820129394531 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.661661997605614e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003001689910888672 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.258795894840553e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.00045752525329589844 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.83451507161631e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0004305839538574219 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 6.258987731487619e-05
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.00028705596923828125 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.19585803179810599
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0003070831298828125 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.2153511355205322
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0003960132598876953 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.3272894618309138
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.00036716461181640625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.27398543743938303
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.00034046173095703125 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.39463530300001876
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0005154609680175781 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.0018699569238093414
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0004134178161621094 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.051894870254133904
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00027823448181152344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.00020943737528189143
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00030231475830078125 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.12370575014991285
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0003578662872314453 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.14514782520737318
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.000408172607421875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.0012597221063081145
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003178119659423828 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.729612097386835e-05
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002865791320800781 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.8701669268386844e-05
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003669261932373047 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.00020547030621017332
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002875328063964844 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 7.63031778572795e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00033783912658691406 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.146210934125938e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0004887580871582031 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.2611987559345384e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0003266334533691406 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.202043453410942e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00030159950256347656 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.064185120786438e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00028824806213378906 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.576664552289412e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00028896331787109375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.444491013568837e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00029969215393066406 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.1938866618468085e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00031828880310058594 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.6925289906080596e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00033593177795410156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.120007007745794e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00038814544677734375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.8368352444854256e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0002892017364501953 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.842957231441179e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0003638267517089844 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.7765936992249355e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00028824806213378906 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.823012957270837e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0003066062927246094 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.596875058744001e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0003604888916015625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.615257949402974e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.00032067298889160156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.5219294503029856e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003409385681152344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.719527419664532e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002968311309814453 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.8473576826690866e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.00038695335388183594 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.8709812425891045e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002903938293457031 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.402093902263666e-05
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.00027060508728027344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.18046469322361958
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.00027060508728027344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.34883960898421934
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.00027561187744140625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.3365312131312721
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.00027060508728027344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.1603314193963193
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.00027251243591308594 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.2585507210664835
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002770423889160156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.022106419796904017
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002703666687011719 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.014686219343160858
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00027298927307128906 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.0020399432077401396
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00026988983154296875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.00176127619705257
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002739429473876953 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.04731353047531475
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.000270843505859375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.50250378181648e-05
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002734661102294922 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 7.236593759403834e-05
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00028967857360839844 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.959474899978603e-05
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002739429473876953 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.810729144901875e-05
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002734661102294922 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 6.577964284994818e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00027489662170410156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.175660119524946e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00027370452880859375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.187127846531518e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.000274658203125 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.7692194981087025e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00027179718017578125 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.222884508263286e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0002815723419189453 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 6.351520575149752e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00027179718017578125 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.96533002884418e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00027251243591308594 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.314615798719763e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00027108192443847656 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.504699747837628e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00027632713317871094 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.990701166229657e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0002741813659667969 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.545016593492322e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0002713203430175781 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.3725583383835827e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00032401084899902344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.413982714550449e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0002758502960205078 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.954204354236297e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00027489662170410156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.7767160184887604e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00027489662170410156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.860706510210362e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002827644348144531 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.785437897573601e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.00027632713317871094 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.125595043272911e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002760887145996094 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.0191205523632627e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.00027632713317871094 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.775404204784432e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.00028204917907714844 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.075696858625546e-05
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0002853870391845703 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.419769441628686
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0002770423889160156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.13863811712653648
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0003724098205566406 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.3261045918642647
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0002989768981933594 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.17552134659358748
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0003592967987060547 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.2514788138408687
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00029778480529785156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.07761467648028422
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002796649932861328 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.010162134854044004
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002808570861816406 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.001980268056206882
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002758502960205078 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.011855059774983504
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002818107604980469 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.09030046647235833
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002772808074951172 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.00020770120466713239
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00027751922607421875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.5198890360038805e-05
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00027823448181152344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.188055725929686e-05
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00028228759765625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.00010649255273154102
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00027632713317871094 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.0018825615741467343
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00027871131896972656 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 7.407144406422104e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.000278472900390625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.301335848590408e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00028014183044433594 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.9553489769847343e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00031495094299316406 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.423163612305177e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0003104209899902344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.4095362236615984e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0002791881561279297 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.232335724406371e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00027871131896972656 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.471596927493795e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.000278472900390625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.111521122570672e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0002796649932861328 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.998507632454414e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0002868175506591797 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.0722951453154396e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00028324127197265625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.2176801293064084e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0002865791320800781 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.7943681944503314e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0002810955047607422 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.05780006947807e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00028228759765625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.858135349007465e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00028586387634277344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.645622240964408e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002815723419189453 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.035119193486365e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002753734588623047 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.0655565335614066e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002887248992919922 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.2151050891350775e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002837181091308594 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.6634886704787985e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.00027942657470703125 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.9433198284319086e-05
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.00028228759765625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.5281952949593232
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.00032138824462890625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.19902331664744713
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0002799034118652344 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.29920649781606545
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0002834796905517578 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.44722742944117155
    Performing 10 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0002777576446533203 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.4602219009750048
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00028252601623535156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.012377646930363351
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002789497375488281 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.003722930035213273
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002963542938232422 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.0015932163814298917
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00028061866760253906 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.01609744390845806
    Performing 20 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002803802490234375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 0.021187241017291577
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003333091735839844 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.42728680543644e-05
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002841949462890625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.9301314733618725e-05
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002887248992919922 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 5.7523158310613626e-05
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002810955047607422 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.488546231208406e-05
    Performing 30 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003097057342529297 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.10555364518406e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0003006458282470703 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.709639646758685e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00027489662170410156 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.002756236629629e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0002830028533935547 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.436178815996952e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.00027751922607421875 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.0648700176037e-05
    Performing 40 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Total parallel function evaluation: 0.0002791881561279297 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.736302259820427e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0002796649932861328 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.14137154237145e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0002949237823486328 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.949702127257142e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0002779960632324219 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.8746436117926025e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.00027632713317871094 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.188663194972195e-05
    Performing 50 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 50 [                                        ] 2.0%
    Total parallel function evaluation: 0.0002789497375488281 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.07485928780507e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0002841949462890625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 4.2602547478396287e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00028061866760253906 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.785930220456604e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0002956390380859375 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.27756513452918e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.00028324127197265625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.6773933355606096e-05
    Performing 60 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 60 [                                        ] 1.7%
    Total parallel function evaluation: 0.0002796649932861328 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.568803596175786e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002884864807128906 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.771095651296478e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002791881561279297 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.717909025601475e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.00029850006103515625 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.898804617363668e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003018379211425781 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.50473442918947e-05
    Performing 70 simulations!
    It/Sub-it: 12/2 Performing simulation 01 from 70 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002875328063964844 sec
    Determine gPC coefficients using 'LarsLasso' solver ...
    -> relative nrmsd error = 3.892007021583865e-05




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

    ax[0].set_yscale("log")
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

   **Total running time of the script:** ( 3 minutes  31.022 seconds)


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
