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
    pi = lhs.get_lhs_grid(dim=2, n=25)

    # plot
    fig = plt.figure(figsize=(4, 4))
    plt.scatter(pi[:,0], pi[:,1])
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
    Total parallel function evaluation: 0.0006010532379150391 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.33064507199672044
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0008633136749267578 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.546098954314407
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0007734298706054688 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4263138736625479
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0009257793426513672 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.5128003794180802
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0010445117950439453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.48442435750147
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0010204315185546875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4870152032722626
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0009396076202392578 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4438132859966959
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0010492801666259766 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.5091959639996891
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.001043558120727539 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.6065382365076738
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0008640289306640625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4728019457375221
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0007789134979248047 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4003509954063689
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0009174346923828125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.5183392305040071
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0007104873657226562 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2348862412452027
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0013415813446044922 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2443455339992755
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0009629726409912109 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.22892393294383165
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.001043081283569336 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.250598273044519
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010986328125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4277207937834574
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.001157522201538086 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4256508359603072
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010380744934082031 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4265253661321647
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0011715888977050781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.21066158910501878
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0010416507720947266 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2008759615008369
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0010445117950439453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07418144838518666
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.001033782958984375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3316678489865609
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0010962486267089844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0903255095178826
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0004773139953613281 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.26993777246251666
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00029015541076660156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08396134479840141
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00031948089599609375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.10292865211092164
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00030612945556640625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.16359025781968953
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003056526184082031 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2699352102364959
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003437995910644531 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07617144240208278
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002949237823486328 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.035638466374620884
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0006039142608642578 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.012805303842427163
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002968311309814453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.019878145719511325
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002980232238769531 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.028003153149977474
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0010056495666503906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.03600098308742412
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00030684471130371094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01632949012003583
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00030922889709472656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.021657733063319987
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003101825714111328 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.02862854583718134
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003027915954589844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.10006473674455015
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003211498260498047 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.015589468235661907
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003032684326171875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.011631870958792867
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0006048679351806641 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0065291967143541445
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003056526184082031 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0033817564830945657
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003070831298828125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0030896610894716125
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00031685829162597656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0038538672557341816
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0006897449493408203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.28311767109009667
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0007345676422119141 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.35648640945345705
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0008270740509033203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4438647141703332
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0009067058563232422 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3529306724894221
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.000885009765625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.38999969741027807
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0010421276092529297 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3514774123358588
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.000766754150390625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.7170720945242569
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0009424686431884766 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.5917666448707116
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0010442733764648438 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.32257493892606665
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0010521411895751953 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.30611342191094665
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.001043081283569336 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24303829102785302
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0010380744934082031 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.30731009851285146
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0010437965393066406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2136054774105432
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0011968612670898438 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.22713356147858996
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0009264945983886719 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24535495218527623
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0011713504791259766 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.26076814400811077
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0011928081512451172 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.435233273587392
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0007913112640380859 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4917183564201574
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010349750518798828 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2494516631597067
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010988712310791016 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.29192511289925993
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0010991096496582031 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08865376322453639
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0010287761688232422 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06791943581757984
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00030517578125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06461846901154326
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0008914470672607422 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07869008614918431
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0004367828369140625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07712660907120475
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00030040740966796875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.10820879617916758
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00030517578125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09450387406264128
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002982616424560547 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.11971273750602482
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002880096435546875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09151187106931298
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00029969215393066406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07454120964895185
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00036334991455078125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01668034466826537
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0005981922149658203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01535394505730123
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002942085266113281 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.040484197975678975
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003063678741455078 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.018114513610723207
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002989768981933594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.013934406000613367
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00030112266540527344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.018022336296665473
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003044605255126953 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.030395917740972635
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003037452697753906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.011544233217754609
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003056526184082031 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.028723377786839817
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003440380096435547 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01869103963983947
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0006041526794433594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0011742300784264
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0004956722259521484 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0015492917871002836
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00030922889709472656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0012719054579548637
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.000492095947265625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.001315143688032023
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00034928321838378906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0023432229408058675
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0006692409515380859 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.274830641125929
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0007638931274414062 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.38411284123850714
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0007534027099609375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2987775087989496
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0009481906890869141 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.29618675880168854
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0010094642639160156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3125815358191114
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0009171962738037109 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3315425122119171
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0009768009185791016 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.37763248906054336
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0009427070617675781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.32736820994805393
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0009951591491699219 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3055563875451013
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0012569427490234375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3414937850385655
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0013120174407958984 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2990029057518833
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0012068748474121094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.18959228737101122
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0015349388122558594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.30684126190991806
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0008647441864013672 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24987767388007046
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0010268688201904297 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.25099285517172726
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010728836059570312 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.26520484823688684
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0011947154998779297 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2135279312229912
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.001056671142578125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4044620530949811
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010690689086914062 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.21521719591194058
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0009219646453857422 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.34373476781598344
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0010869503021240234 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.10946297910720477
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0010225772857666016 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.15580252927540952
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0009932518005371094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06748637244957922
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0007874965667724609 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06101468165459255
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0009744167327880859 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08571561922764076
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.000293731689453125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06491700528824733
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003173351287841797 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.12243732004049984
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003094673156738281 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.1299588298898395
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003437995910644531 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0917677539440354
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003039836883544922 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.15164235543063392
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00030517578125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.013969789373005397
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002932548522949219 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.045559689128408805
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0004496574401855469 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.011991122097368204
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00029468536376953125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.009283915881619977
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003368854522705078 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.030780151966862287
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00030541419982910156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.014942179667226664
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0002932548522949219 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01796965881293655
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0002951622009277344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01516881604011407
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00030493736267089844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.04968044182994594
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0002968311309814453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01810342840595371
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003120899200439453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0018729245902574497
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003185272216796875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.004851189112356976
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003135204315185547 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.002544833817044029
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003037452697753906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.008074896289049649
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00030541419982910156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.00142599999009998
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0007150173187255859 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.5063504671950788
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00074005126953125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.28613485041071357
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00087738037109375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3079980291830451
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0009429454803466797 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.33911037518484816
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0009708404541015625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.28983294849163266
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0010647773742675781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.43896670398788057
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0009334087371826172 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.42087979042720414
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0010426044464111328 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4414841864698674
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.001018524169921875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3333203879810072
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0009443759918212891 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.392287994327959
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00109100341796875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.22457097125104236
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0010285377502441406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.19636452395409998
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0010406970977783203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.29860461568088925
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0011298656463623047 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.33534178971130646
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0010075569152832031 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.19837892410833857
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010251998901367188 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.6274634028984615
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010442733764648438 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.35599866490537385
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0011217594146728516 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.39921440055905055
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0011713504791259766 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3123226050758855
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010495185852050781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2670388428467607
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0010371208190917969 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.10700419059727015
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0011012554168701172 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06391024588825961
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.001199960708618164 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08139972615556093
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0010368824005126953 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07406252813444344
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0009527206420898438 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.10711987156000559
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002968311309814453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09729081197338989
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00030493736267089844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.13188565758479792
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002906322479248047 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0891078948321494
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003025531768798828 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.1480383307282351
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002968311309814453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07967294377244258
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0005075931549072266 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.014037072393499363
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003001689910888672 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01520106900374366
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002949237823486328 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.013258192550748272
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0005924701690673828 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0130552107494772
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0005979537963867188 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01527833019409868
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003337860107421875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.012998615473158977
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00031447410583496094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.029827290199309305
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003151893615722656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01571084129919073
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00031495094299316406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.014803768077538177
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00034880638122558594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.016140046289339117
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00031447410583496094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0013287676008800992
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00031113624572753906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0010733443389153554
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003075599670410156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.001095275127374414
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003113746643066406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0009282210866505476
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00031495094299316406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0017476358329698606
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0006906986236572266 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.28017476019932774
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0006339550018310547 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2919182718079702
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0006730556488037109 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3014429851492073
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0008435249328613281 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.34043736319774764
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0005691051483154297 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.27929496435633244
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0009281635284423828 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2929541127562987
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0007674694061279297 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2769025611172429
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0009245872497558594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2862137836758401
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0007331371307373047 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2894506141366571
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0008416175842285156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2775572328478013
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0008108615875244141 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.1878765677014089
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0009162425994873047 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.20799283723123496
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0007870197296142578 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.16672597973821507
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.000843048095703125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2698709607935811
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0007364749908447266 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.198061864660411
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0007750988006591797 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.1719590395285683
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0007731914520263672 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.21129273553488032
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.00087738037109375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.23032282938638798
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0006875991821289062 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.23492062670920322
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0006587505340576172 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.21889754592168462
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0006544589996337891 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07086704666958446
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0009083747863769531 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.05569950586396955
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.000823974609375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06041474097848286
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0007369518280029297 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06635091196331189
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0008008480072021484 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08995783753927986
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00030350685119628906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08107332829199629
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00031113624572753906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06405140755921684
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00028586387634277344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.051659500305332
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003018379211425781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.05013855566796517
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002970695495605469 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.056689036112082355
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0004246234893798828 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.012827114936215142
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003237724304199219 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.010189193152549693
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002856254577636719 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0161290091996297
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0004949569702148438 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.009027057090789116
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002951622009277344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.012200677124867144
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00047707557678222656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.014434359870105896
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003275871276855469 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.013776215772404832
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003044605255126953 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.008825101646481832
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00029921531677246094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.012443854030906526
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003178119659423828 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.013861219124120415
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0002989768981933594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0015745285671160698
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0006234645843505859 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0023504485018389564
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003046989440917969 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0014094312035241882
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003185272216796875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0007236649860295108
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003151893615722656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0015295307576652704
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
    _ = ax[1].set_title("gPC error vs original model (std)")



.. image:: /auto_gpc/images/sphx_glr_plot_grid_random_vs_lhs_003.png
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 2 minutes  23.306 seconds)


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
