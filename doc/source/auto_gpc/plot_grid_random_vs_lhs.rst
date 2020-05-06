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
sphinx_gallery_thumbnail_number = 2:


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
    Total parallel function evaluation: 0.0005900859832763672 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3411633210083902
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0008275508880615234 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.44848974797225527
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0008263587951660156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.28910713165976953
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0008637905120849609 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.39799602969677733
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0009238719940185547 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.359646461090493
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0006506443023681641 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4414778346445425
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0005881786346435547 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.35606580318341335
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0006697177886962891 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2939809471686884
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0009303092956542969 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4144054741641762
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0010826587677001953 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.8272296157525125
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0005936622619628906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2112820690910254
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.001016855239868164 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.31577731406237625
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0011680126190185547 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24272859136616318
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0010035037994384766 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.30520664292418076
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0011904239654541016 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3316519618673423
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010051727294921875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.5403150034520617
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0005588531494140625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.7564867599045157
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010018348693847656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.40744423645003813
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0009963512420654297 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3271833816873176
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.001102447509765625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2806299906259488
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0004153251647949219 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.18165280030438652
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0010788440704345703 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2867074565407547
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0009260177612304688 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06350224246730579
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0004355907440185547 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06973696665623687
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0010101795196533203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07850876986605662
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002899169921875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0855653644637648
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00029158592224121094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.20664752843423523
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002865791320800781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07080626970998671
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002903938293457031 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.10549837722635039
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003254413604736328 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09679615392663718
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0005865097045898438 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01923432719269401
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00028395652770996094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.017941694768275517
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002853870391845703 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.05415670306329138
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0006527900695800781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.028663322438428365
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002865791320800781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.018896522047999815
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0004134178161621094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.032689319817172754
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00039315223693847656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.038613244436630334
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0002949237823486328 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.019337578784561067
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0002963542938232422 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.027741139830127996
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0005159378051757812 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.026398982516264678
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003237724304199219 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.020608323911383338
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.000347137451171875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0024000306224635595
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00036525726318359375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.002854995796224686
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00029969215393066406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01146564097158087
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00029397010803222656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.001199790674804371
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0007376670837402344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.29932972274492425
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0007309913635253906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.5388455398472665
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0007252693176269531 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.31441077210620577
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0012135505676269531 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3547545124910063
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0009443759918212891 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3675508399756489
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0010149478912353516 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.32352391967467437
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.001107931137084961 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3957527834365172
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0010442733764648438 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.43970305338295956
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0012543201446533203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4157690585381963
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0010373592376708984 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.47458399154330794
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0012023448944091797 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.25322372823168393
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0007686614990234375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.5011014551681982
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0017354488372802734 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.48109640704515927
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0010256767272949219 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.22076087331684588
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0011982917785644531 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.25094410741711665
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0011496543884277344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3507710109608819
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010569095611572266 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.45475948009968264
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010497570037841797 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.27863925177090204
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0011854171752929688 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.37591223950022024
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0005061626434326172 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.31726900927069285
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0005414485931396484 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06469138375463693
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.000522613525390625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06859278748748805
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0005054473876953125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06770516651097826
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0012445449829101562 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07869103387503372
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0005183219909667969 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07737830559720509
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00030422210693359375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.22390456975051629
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003044605255126953 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08172659048004609
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003478527069091797 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.16245443774931675
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003342628479003906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07055522929115866
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0014641284942626953 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.16942165366673956
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00034427642822265625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.03235996208174953
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0002980232238769531 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.030950062802995188
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003600120544433594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.019020625543507516
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0004591941833496094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.1344469387768134
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003972053527832031 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.012173097362928088
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003342628479003906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.015252838155550641
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00067138671875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.019091022825153455
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003275871276855469 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.015401801813778699
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00029659271240234375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.021515703797031237
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003082752227783203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.018956323985787626
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00030922889709472656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0037028875043636014
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00030422210693359375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.002086852059655248
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00031447410583496094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0019441844980985438
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003192424774169922 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0026027538283884656
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0005025863647460938 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.002052313537877079
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00043702125549316406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3940208868546257
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0007870197296142578 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.41268385470115315
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0008504390716552734 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3478118179111896
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0009245872497558594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.37654806046983785
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0010797977447509766 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.31022490567882455
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0011548995971679688 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3470296575918378
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0005223751068115234 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.34436318467865085
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0010569095611572266 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3495543226766119
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0009839534759521484 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.32009591390110387
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0005469322204589844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4148664991826893
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0012450218200683594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.20995434066189136
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0017147064208984375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.27463846270825165
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0009310245513916016 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4018602875713958
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0006871223449707031 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24711422550969533
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0008852481842041016 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.35498102029533884
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0008456707000732422 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.20453825231537384
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0008635520935058594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24477993922983807
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0009713172912597656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2538793510274124
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010859966278076172 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2356103212771226
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0005464553833007812 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3411522987914836
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0014204978942871094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09649480895202463
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0010442733764648438 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08749902557888443
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0011296272277832031 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08662673199177419
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0010313987731933594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07863817232574341
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0010256767272949219 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0839618476797768
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00031495094299316406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06644028154496553
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00031304359436035156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08576928847306217
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00030040740966796875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.11302307273291565
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003046989440917969 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07959819314554577
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002911090850830078 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08778115560952977
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00030684471130371094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.014895472572471377
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003008842468261719 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.015047542260383592
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0006816387176513672 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.018125472207841893
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003020763397216797 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01079519358732736
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003008842468261719 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.012911826906354208
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0002949237823486328 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.03169178343650243
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0002970695495605469 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.013916801053795405
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0002961158752441406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.013659526222163605
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00029730796813964844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.011714227770511444
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.000301361083984375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01902213444102801
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003070831298828125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0009113841260259473
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0006310939788818359 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0018075896657403021
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003578662872314453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.001783945042699453
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003008842468261719 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0016971257301365721
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00030493736267089844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.015530608805399542
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0007839202880859375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2905357333797566
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0007994174957275391 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.29484126347336265
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0009989738464355469 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.29812039054768036
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0014119148254394531 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.31761538609155776
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0011610984802246094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.30554328767878813
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0007879734039306641 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2992697092549257
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0016620159149169922 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3696260453514938
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.001013040542602539 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3596026035053065
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0011005401611328125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.29433532074430674
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0010347366333007812 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.28742659092609346
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.001026153564453125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.17831596913365974
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0010480880737304688 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2648670927179873
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0003273487091064453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.20221296248639387
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00046563148498535156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.17845588478450045
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0011005401611328125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.1932376166842211
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010647773742675781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24964002001943292
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010187625885009766 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.26064205105284133
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0011963844299316406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3099079387786072
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0011909008026123047 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2779690070332276
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0011081695556640625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2355864837714504
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0007429122924804688 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06662745367718392
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.001155853271484375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.11780376792964443
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0008244514465332031 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.10124913065088557
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0012798309326171875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06430205882231732
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0009751319885253906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.10387435576805014
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003139972686767578 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07891766553418984
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00029349327087402344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.1362873542952538
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00028967857360839844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.1038076627490928
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0004515647888183594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.060654660385862924
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.000396728515625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3726363582044131
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00029540061950683594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.010353814368737738
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00029397010803222656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.026941609157030616
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00031447410583496094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.025996054042762694
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00029850006103515625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.012672281946114163
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003609657287597656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.013356990872096691
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00030493736267089844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.029599598921770783
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003020763397216797 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.019837062007909318
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003082752227783203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.014203890230221663
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00030517578125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.025755597303088972
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00030875205993652344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.014118904451032302
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00030684471130371094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0015320527999491663
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003845691680908203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0015593758670349005
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00030541419982910156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.00226390925926712
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003135204315185547 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0019304930812090383
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0005314350128173828 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0032805087472853028
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0007469654083251953 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2918112340328755
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0007762908935546875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.29686284958278514
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0008122920989990234 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.28291038265693963
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0007634162902832031 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3493551800629417
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0016489028930664062 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.35469437232307105
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.001054525375366211 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.28093569709352395
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0007946491241455078 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.27741325237052716
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0004520416259765625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.27029881222699836
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0004451274871826172 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2798981126104185
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0007240772247314453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2879928212703381
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0008802413940429688 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.19356918524234265
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.001636505126953125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.19496586135691615
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0008590221405029297 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.1956872820674673
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00030517578125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2162801416398415
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0007011890411376953 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.20446090136772183
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.002270221710205078 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.19748404774296532
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0005679130554199219 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.23506088146045456
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010573863983154297 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.179954745727931
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0009665489196777344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.20187793448895155
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0003235340118408203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.19657102539291813
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.001470804214477539 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07083596698270449
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0007009506225585938 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0549478023822352
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0008461475372314453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06190017044627378
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0007772445678710938 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.05528155873440478
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0007262229919433594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.057055859422752875
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00036215782165527344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06710061545234026
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0002987384796142578 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0921763117955537
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00037479400634765625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06622318480667021
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003101825714111328 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.059619246939712045
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003077983856201172 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06426033915448133
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00038170814514160156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.008533910589588741
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00029659271240234375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.012913703694624766
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003066062927246094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.00905458886268688
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0005793571472167969 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.012498413214810227
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003478527069091797 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.012882925702526776
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003032684326171875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0185019687918211
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00031185150146484375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.012474349731702493
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.000293731689453125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.009477880560624992
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00048232078552246094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0113710098584639
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00036787986755371094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.013977630822348123
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.000423431396484375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0009585216263142648
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003383159637451172 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.001900153322860712
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003039836883544922 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0008244110022705227
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.000499725341796875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0013334739356011811
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00033164024353027344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0017861969703185457
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

   **Total running time of the script:** ( 2 minutes  29.445 seconds)


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
