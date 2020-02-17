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
application is constructed in `pygpc/Grid.py <../../../../pygpc/Grid.py>`_

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
:math:`p_{ij} \in \mathbb{N}, {1,...,n}` and u is uniform random number :math:`u \in [0,1]`

Constructing a simple LHS design
--------------------------------
We are going to create a simple LHS design for 2 random variables with 5 sampling points:
sphinx_gallery_thumbnail_number = 3


.. code-block:: default


    import pygpc
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import OrderedDict

    # define parameters
    parameters = OrderedDict()
    parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
    parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])

    # define grid
    lhs = pygpc.LHS(parameters_random=parameters, n_grid=0)

    # draw samples
    pi = lhs.get_lhs_grid(dim=2, n=5)

    # plot
    fig = plt.figure(figsize=(4, 4))
    plt.scatter(pi[:,0], pi[:,1])
    plt.grid(True)

    # \\rho_{rg_{X_i}, rg_{X_j}} = \\frac{cov(rg_{X_i}, rg_{X_j})}{\\sigma_{rg} \\sigma_{rg}}




.. image:: /auto_gpc/images/sphx_glr_plot_grid_random_vs_lhs_001.png
    :class: sphx-glr-single-img





LHS Designs can further be improved upon, since the pseudo-random sampling procedure
can lead to samples with high spurious correlation and the space filling capability
in it self leaves room for improvement, some optimization criteria have been found to
be adequate for compensating the initial designs shortcomings.

Optimization Criteria of LHS designs
-----------------------------------
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
    lhs_basic = pygpc.LHS(parameters_random=parameters, n_grid=0)
    lhs_corr = pygpc.LHS(parameters_random=parameters, n_grid=0)
    lhs_maximin = pygpc.LHS(parameters_random=parameters, n_grid=0, options='maximin')
    lhs_ese = pygpc.LHS(parameters_random=parameters, n_grid=0, options='ese')

    # draw samples
    dim = 5
    n = 30
    samples = []

    samples.append(np.random.rand(n, dim))
    samples.append(lhs_basic.get_lhs_grid(dim, n))
    samples.append(lhs_corr.get_lhs_grid(dim, n, crit='corr'))
    samples.append(lhs_maximin.get_lhs_grid(dim, n, crit='maximin'))
    samples.append(lhs_ese.get_lhs_grid(dim, n, crit='ese'))

    # calculate criteria
    corrs = []
    phis = []
    name = []
    variables = []

    for i in range(5):
        corr = spearmanr(samples[i][:, 0], samples[i][:, 1])[0]
        corrs.append(corr)

    for i in range(5):
        phip = lhs_basic.PhiP(samples[i])
        phis.append(phip)

    variables.append(corrs)
    name.append('corr')
    variables.append(phis)
    name.append('phi')

    # plot results
    fig = plt.figure(figsize=(16, 3))
    titles = ['Random', 'LHS (standard)', 'LHS (corr opt)', 'LHS (Phi-P opt)', 'LHS (ESE)']

    for i in range(5):
        text = name[0] + ' = {:0.2f} '.format(variables[0][i]) + "\n" + \
               name[1] + ' = {:0.2f}'.format(variables[1][i])
        plot_index = 151 + i
        plt.gcf().text((0.15 + i * 0.16), 0.08, text, fontsize=14)
        plt.subplot(plot_index)
        plt.scatter(samples[i][:, 0], samples[i][:, 1], color=sns.color_palette("bright", 5)[i])
        plt.title(titles[i])
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
    grids_options = [None, None, "corr", "maximin", "ese"]
    grid_legend = ["Random", "LHS (standard)", "LHS (corr opt)", "LHS (Phi-P opt)", "LHS (ESE)"]
    order = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    repetitions = 5

    err = np.zeros((len(grids), len(order), repetitions))
    n_grid = np.zeros(len(order))

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
    options["solver"] = "Moore-Penrose"
    options["interaction_order"] = problem.dim
    options["order_max_norm"] = 1
    options["n_cpu"] = 0
    options["adaptive_sampling"] = False
    options["gradient_enhanced"] = False
    options["fn_results"] = None
    options["error_type"] = "nrmsd"
    options["error_norm"] = "relative"
    options["matrix_ratio"] = 2
    options["eps"] = 0.001
    options["backend"] = "omp"








Running the analysis
^^^^^^^^^^^^^^^^^^^^


.. code-block:: default

    for i_g, g in enumerate(grids):
        for i_o, o in enumerate(order):
            for i_n, n in enumerate(range(repetitions)):

                options["order"] = [o] * problem.dim
                options["order_max"] = o
                options["grid"] = g
                options["grid_options"] = grids_options[i_g]

                n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
                                                       order_glob_max=options["order_max"],
                                                       order_inter_max=options["interaction_order"],
                                                       dim=problem.dim)

                grid = g(parameters_random=problem.parameters_random,
                         n_grid=options["matrix_ratio"] * n_coeffs,
                         options=options["grid_options"])

                # define algorithm
                algorithm = pygpc.Static(problem=problem, options=options, grid=grid)

                # Initialize gPC Session
                session = pygpc.Session(algorithm=algorithm)

                # run gPC session
                session, coeffs, results = session.run()

                err[i_g, i_o, i_n] = pygpc.validate_gpc_mc(session=session,
                                                           coeffs=coeffs,
                                                           n_samples=int(1e4),
                                                           n_cpu=0,
                                                           output_idx=0,
                                                           fn_out=None,
                                                           plot=False)

            n_grid[i_o] = grid.n_grid

    err_mean = np.mean(err, axis=2)
    err_std = np.std(err, axis=2)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0002033710479736328 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.5433877647220274
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00033402442932128906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.35533444647025336
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00028204917907714844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3107113487337948
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0003132820129394531 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.5046456898501558
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00035572052001953125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.916731366487928
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0003688335418701172 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.7818565603118796
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00033092498779296875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.6494551515455131
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00032448768615722656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3509103383133358
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0003650188446044922 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4158490457480852
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002110004425048828 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.6983672574198179
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00022673606872558594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.21194051848276757
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003154277801513672 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3421853440752368
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003685951232910156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.27981337328156625
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00040411949157714844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.33936073822427604
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003829002380371094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24317971225588145
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.00038909912109375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.783858137240265
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0004105567932128906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.9154761827379193
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0003235340118408203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.5638954778655182
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.00038909912109375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4500510463177611
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.00037598609924316406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24866569048267492
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0003657341003417969 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.1376163159424955
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00037217140197753906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.11746693838078044
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0003905296325683594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3902567009981677
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0003757476806640625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06896804440152277
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00038361549377441406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.11556464991646165
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0004088878631591797 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.12579170954420935
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0004296302795410156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.13098110069861155
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00022125244140625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09293014919091706
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003523826599121094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.11152742773085539
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00024700164794921875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08190476796566484
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00021028518676757812 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.014907047154488413
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002186298370361328 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.03765167358929649
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00022077560424804688 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.03950274684374611
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00024700164794921875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.017046476345880975
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00023126602172851562 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.029195059464124606
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00030875205993652344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07805053619502961
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0002434253692626953 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01245914106637819
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0002536773681640625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.034043286868280995
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00022721290588378906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.011042445563831141
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00022840499877929688 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.052315909110233386
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00022792816162109375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0034049829604337056
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00044345855712890625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.013335304832181536
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00043487548828125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0032014324612934545
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0004379749298095703 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0028362748818654693
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0002582073211669922 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0035916306080994815
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00031065940856933594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.44375599624489764
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00039958953857421875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.30428135191610484
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.001828908920288086 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3018383384794647
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.004378795623779297 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4682438331004955
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0003910064697265625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.33290118338144375
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0003600120544433594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3665857148957504
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0004239082336425781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3435866278707418
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0003581047058105469 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.33319412862364495
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00039649009704589844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3477341313207258
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002810955047607422 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3529372003316574
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003826618194580078 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2709283759771363
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00022792816162109375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2703035400768187
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00037384033203125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.228197262388667
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003790855407714844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.20373484970104566
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003478527069091797 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2950445041514866
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.00038623809814453125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.21845339753705234
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.00039505958557128906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.29971149327907853
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.00036597251892089844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.23759897666291208
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.00030541419982910156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.31501552854451453
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.00033664703369140625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2775224390040885
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0004029273986816406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09667714708605198
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0004134178161621094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.17181615472335793
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00039839744567871094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06871165957418535
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0003917217254638672 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09603239336693829
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00039887428283691406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.18691602538303814
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.000209808349609375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.10064911739589719
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002276897430419922 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08320392826010665
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002319812774658203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07122295764331023
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00022673606872558594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07494570512607182
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002167224884033203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.11187571574394137
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002357959747314453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.024445840618764463
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00031113624572753906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.017714862497755418
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00022554397583007812 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.021134327535440433
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002396106719970703 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01638304879078302
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00024509429931640625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.008913987197463781
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00023627281188964844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.021312309940690916
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00025081634521484375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01870276529456801
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00023102760314941406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01726241042698299
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0002624988555908203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.025964372813364837
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0004322528839111328 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.02175437540600101
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0005049705505371094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0030973421611853033
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00043654441833496094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.001357978337621759
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0004532337188720703 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0031723635923449228
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00024175643920898438 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0029853743186859464
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00041866302490234375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0017703485543464318
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00020360946655273438 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3134852104559497
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00019693374633789062 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3479773501035288
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0002033710479736328 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3730880816758231
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00023436546325683594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.36375431675567815
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0002067089080810547 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3115083675397189
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00020885467529296875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3889961767647942
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00032448768615722656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4132070935314119
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002276897430419922 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3325814769074769
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00021958351135253906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4700429910746371
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00019979476928710938 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3130645921844643
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002493858337402344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.26437774610332376
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00022077560424804688 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2536387946384991
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002429485321044922 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.34974558379996445
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0002224445343017578 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.41589061121965704
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00022339820861816406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2191706517628579
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0002651214599609375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3703360242775355
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.00021719932556152344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3280409300754925
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.00021195411682128906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.403587944048173
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.00021195411682128906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3776506937239868
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.00019884109497070312 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2585078941379881
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00021457672119140625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.28894936316660025
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.000213623046875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0943320739240014
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00021028518676757812 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.13954506431335656
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00023508071899414062 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.10364549958058003
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00030732154846191406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.1955727760556265
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00039505958557128906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09690683548723716
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00021076202392578125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.11333848996217184
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00039196014404296875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08154416156789843
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00021409988403320312 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.071878486006071
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00022673606872558594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09191753456857867
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00021266937255859375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.03182586993866362
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00033473968505859375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.024174718642728937
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00029850006103515625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.018637756475459465
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002200603485107422 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.013897171475178358
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00044798851013183594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.016719951423219612
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0004248619079589844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.011711291561500074
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00043487548828125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.014919126773078734
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00022220611572265625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.02171465699030314
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0002560615539550781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.018847170903202665
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00023174285888671875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.014872006938964283
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00024962425231933594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0015961521063070114
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00023603439331054688 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.00203524995728045
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00027108192443847656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.004550351006278217
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00042366981506347656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0050319924647534145
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0004563331604003906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.001979864224633544
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0002048015594482422 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2891115760236314
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00024056434631347656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2804418362899546
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0002808570861816406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.29826337347124326
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0003764629364013672 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.281778838509062
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0003561973571777344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2797160140888466
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00024890899658203125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3295194487164702
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00023818016052246094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3452556273864659
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00022482872009277344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3465599702175005
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002846717834472656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2965758179076342
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0003795623779296875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3179866298734377
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00037169456481933594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.22270736732916113
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00038814544677734375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.34149352926695353
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003924369812011719 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24354005561195624
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00028824806213378906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24283729235149956
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00037860870361328125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.18142653665862973
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.00021958351135253906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.401032687393366
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0002551078796386719 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.22419706678774237
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0003764629364013672 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3957471388629047
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0003001689910888672 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.18951887537722845
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0002467632293701172 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3198747602598206
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00021719932556152344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07789939876626199
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.000255584716796875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.12335400442087695
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00020503997802734375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09392048472288271
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0002193450927734375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07586688690544903
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00021696090698242188 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.16373800387363854
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00020122528076171875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24007953475142219
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00022125244140625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0821522320549761
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0004334449768066406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09652136935958858
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0004057884216308594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.059172534936107204
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0004067420959472656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08154310501321563
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0004246234893798828 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0099429096550294
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00041222572326660156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.020238754897906993
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0004088878631591797 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.03365179356918201
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0004134178161621094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.017787752361037364
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00040841102600097656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.012144331295879803
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00024199485778808594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0253374231456162
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00026416778564453125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.016265763068112733
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00022363662719726562 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.014619830034328023
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00023245811462402344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.014376784565417244
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00023651123046875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.03660367931599181
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0004482269287109375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0015748147609385601
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00024008750915527344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0013474946798613604
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0004286766052246094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0023518463931562128
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0004761219024658203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.004177274470333241
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0002777576446533203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0015124636664336482
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00021147727966308594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.31130058345697625
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00020241737365722656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2916808120845216
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00024127960205078125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2775475179726965
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00019741058349609375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3483290643831366
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00019788742065429688 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.34270599332678897
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.000213623046875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3043516897485346
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00021076202392578125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2825795849927684
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002205371856689453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2908821715941556
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00020766258239746094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3257375008749576
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0002319812774658203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2904926511024976
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00021576881408691406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.19283372070311453
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00021839141845703125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24394226312552802
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.000209808349609375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.21341930125165026
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00020122528076171875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.18259827659236527
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00019860267639160156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.1846514838382906
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0003008842468261719 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.19078861433446231
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0002086162567138672 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.20233011590601746
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.00022363662719726562 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.20731009676033907
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.00020837783813476562 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.25998054886676186
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0002162456512451172 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.19035294913778336
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00020194053649902344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.05584186828278266
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00021910667419433594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.058159449981994384
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0002090930938720703 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06413043877593776
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00022292137145996094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0698153612419219
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00021576881408691406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08285280496948222
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00046062469482421875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0620910115995249
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00041413307189941406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09955311007943424
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0004010200500488281 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06747096659389984
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00042319297790527344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06190530093286553
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0004322528839111328 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08887951515541992
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00042510032653808594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.011782443354500901
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0004429817199707031 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.009833851743714324
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00041365623474121094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.00939618675763768
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00023889541625976562 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.009040671081349132
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002498626708984375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01207058347518034
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0002181529998779297 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.011951753431249277
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00042247772216796875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.007959565018033942
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0005156993865966797 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.017722553787040636
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00021600723266601562 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01543982892941221
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00041174888610839844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.012567779871451926
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0004353523254394531 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.001496314091386239
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00041174888610839844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0011331621208578826
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0002620220184326172 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.002091793603079849
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00024127960205078125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0016422469576698917
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00042247772216796875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0017034897879076377
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%




Results
^^^^^^^
Even after a small set of repetitions the :math:`\varphi_P` optimizing ESE will produce
the best results regarding the aforementioned criteria, while also having less variation
in its pseudo-random design. Thus is it possible to half the the root-mean-squared error
:math:`\varepsilon` by using the ESE algorithm compared to completely random sampling the
grid points, while also having a consistently small standard deviation.


.. code-block:: default


    fig, ax = plt.subplots(1, 2, figsize=[12,5])

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
    ax[1].set_title("gPC error vs original model (std)")



.. image:: /auto_gpc/images/sphx_glr_plot_grid_random_vs_lhs_003.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    Text(0.5, 1.0, 'gPC error vs original model (std)')




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  56.256 seconds)


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
