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
    Total parallel function evaluation: 0.0008671283721923828 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.32965051700118025
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0009000301361083984 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.6150769230256643
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.001001119613647461 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.6915179091436132
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0008144378662109375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.32791380875668175
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0004878044128417969 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3831439516601068
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0011026859283447266 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.49184949197422534
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0006325244903564453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 1.1985018755042445
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0013265609741210938 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.39147684586079223
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0011019706726074219 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 1.1715184530478797
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0014374256134033203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.44410136613466655
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0012793540954589844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4385246905881125
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0008981227874755859 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.212901646214159
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.001247406005859375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.44471331020444843
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0007097721099853516 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4110210775513524
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0014166831970214844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2581950445612717
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.001295328140258789 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2996721849808704
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0013496875762939453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.505494815677068
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0014641284942626953 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.26591504300658003
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.001245737075805664 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4860639530321117
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0004582405090332031 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4977935665960531
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0011806488037109375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06711032233994649
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0011799335479736328 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07051482694076087
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0004668235778808594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.05155399347006535
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0013208389282226562 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.11886642801030035
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.001226663589477539 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.11282838181583091
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00031948089599609375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.10357162272631587
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.000324249267578125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07457283743600675
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00032329559326171875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09093108625661106
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003235340118408203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08612648739134901
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00033211708068847656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0849503248689929
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00033020973205566406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.013010808101611334
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003185272216796875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07833041962473923
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00032210350036621094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.020818266919192654
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003268718719482422 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.012854119883360401
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00032401084899902344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01653161267214408
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00031948089599609375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01131129282251151
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00035452842712402344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01943395071564083
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003159046173095703 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.058958252897242665
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0004937648773193359 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07037884012821895
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003304481506347656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.013638804127003818
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00051116943359375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0019079531073238063
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003523826599121094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0023078727340819167
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00036072731018066406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0023840357427296175
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003490447998046875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.005682194332135005
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0005853176116943359 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.001468632987137165
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0003924369812011719 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3398383863884569
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0004837512969970703 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.38321632573738507
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00043010711669921875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.30135182946468947
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0003345012664794922 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3535916276208592
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0005319118499755859 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.39307651579088304
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0005145072937011719 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.30624557753025966
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0005092620849609375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4098980477865768
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00047469139099121094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.5178826527537563
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.00046825408935546875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.47819824270720546
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0004413127899169922 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.44900621909638383
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0004315376281738281 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.34658492980448663
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0004849433898925781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3278940549294228
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00037026405334472656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.20401853057259006
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0014493465423583984 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.267489052159903
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0005900859832763672 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24081486480232342
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0005905628204345703 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2613371421969584
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0015349388122558594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2878961074950875
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.00036644935607910156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.22549029332435006
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0005254745483398438 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.6107827619230075
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0008976459503173828 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.21375133737259053
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0010068416595458984 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.14512817711035378
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0009620189666748047 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.14842794750063645
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0009763240814208984 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09441940556468362
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00048804283142089844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.11713550693449219
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0005507469177246094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07797252363951303
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00033473968505859375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.1873322102024641
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00035572052001953125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.12141045923330868
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00030112266540527344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.10805015891082988
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003216266632080078 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0713757131648799
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003440380096435547 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.11127680885586866
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.000316619873046875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.03241500422419163
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00047016143798828125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.053389092550640345
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003237724304199219 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.011254900463211684
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0005731582641601562 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01969669809336342
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0006642341613769531 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.03130580719777896
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003345012664794922 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0229790452411659
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00031828880310058594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.043508808943534434
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003311634063720703 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.05481980613850361
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00032258033752441406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.011954983344775962
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00031948089599609375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.020053178018417505
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003151893615722656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.003810436746165234
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003323554992675781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.00198801585059959
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00039458274841308594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0020081987111908715
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0006229877471923828 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0012891134261516497
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.000335693359375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0011481972280768338
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00078582763671875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.41981380650361577
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0004968643188476562 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.29843513308033365
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0009655952453613281 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3451055330517679
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0006542205810546875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3520984606550244
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.00041985511779785156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3215360754394527
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0006933212280273438 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.40241210767237184
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0016357898712158203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3894431318037418
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0012357234954833984 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4208043635381923
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.001165151596069336 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.37123272002703517
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.002246379852294922 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.35535057141061316
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0005037784576416016 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.20930422540598745
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0009114742279052734 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4026359943696481
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0019273757934570312 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24076051751442162
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0011734962463378906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4003118525174663
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.00037169456481933594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.19570186911958928
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0004279613494873047 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2978108861404222
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0011632442474365234 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.33358557411178674
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0009844303131103516 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.22158499924814867
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0004978179931640625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.23543902387189283
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0004010200500488281 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.32703087065823533
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0006306171417236328 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.12077888460339073
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0010213851928710938 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0754991558661975
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.000946044921875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09511303184097944
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0009818077087402344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08037996002996278
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0011458396911621094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.11830598082970273
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00033354759216308594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.1362167739456595
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0005519390106201172 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.17485496461469618
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00033283233642578125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.164065201474014
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.000324249267578125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.32853622113937164
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00032401084899902344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24301949968557743
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0006477832794189453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.037945713633389895
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0005857944488525391 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.015708596528527845
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003211498260498047 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.014888060389245462
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00032258033752441406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.025125666882901444
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003170967102050781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.018002615025590325
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00033211708068847656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01782568297797758
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00035834312438964844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01666575402369556
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00032782554626464844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.03677319116607281
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003330707550048828 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.014809739627359177
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.000331878662109375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0426393566557967
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00040793418884277344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.001904238983019215
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003399848937988281 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.002015366712179703
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00037384033203125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.004076780914595479
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003333091735839844 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.002664248428494406
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0005381107330322266 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0011526845635223715
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.001024484634399414 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2944380951704596
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0008933544158935547 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3033035379589684
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0011310577392578125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2931542216000166
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0010895729064941406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.32660835103385555
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0010187625885009766 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2943276699591107
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0011167526245117188 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.32298299102773287
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0010869503021240234 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3314051403331821
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0010161399841308594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3224110361084736
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0007634162902832031 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.34445390414421306
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0010237693786621094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.30462559442574044
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0006170272827148438 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.19150789334404358
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0015964508056640625 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.310502214268278
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0011432170867919922 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.22044835058575238
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0010149478912353516 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.23096704254474415
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0016393661499023438 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.20389503157294933
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0011687278747558594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.5021304857032262
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0012543201446533203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.4373890707133855
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0005080699920654297 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24621007065525075
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0004885196685791016 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.23681281446631386
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0010921955108642578 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2682994800429234
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0018007755279541016 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.11272552902936069
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.00045609474182128906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09953504705846752
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0004038810729980469 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0742480738818714
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0005676746368408203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07131838316130092
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.001220703125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.08719014977895458
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003230571746826172 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24254335707810443
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003199577331542969 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.10849042009749296
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003325939178466797 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09514246645353254
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003199577331542969 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.11181908513837323
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003218650817871094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.05950967739103951
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00032067298889160156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.015050108683973317
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003235340118408203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.012803719110882975
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0006463527679443359 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.03150945299606057
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00031948089599609375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.015466558433996877
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003170967102050781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.02421760733290502
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003421306610107422 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.03610754360641372
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003170967102050781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.027327044553491962
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003743171691894531 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.018521190615046725
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003170967102050781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.018740692427199675
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00032067298889160156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.029494026615114195
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.000324249267578125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.005704697429706211
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00033020973205566406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0017333398070345947
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00034165382385253906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.004336038929802115
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00032401084899902344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0033381906988823834
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00033164024353027344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0019041918463336765
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0006935596466064453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.34943974350078794
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0007596015930175781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.30794811536471467
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0007345676422119141 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2863778650115438
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.001005411148071289 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3029548270636301
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 12 simulations!
    It/Sub-it: 2/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.0010898113250732422 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.27674870471409513
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0008177757263183594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.28332283629905625
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0009739398956298828 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2704917515819616
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0009109973907470703 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.3510845271203993
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.001140594482421875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.27653144449898137
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 20 simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total parallel function evaluation: 0.0009253025054931641 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2714361870059388
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0010797977447509766 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.1853435968815733
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0008537769317626953 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.1760998228421143
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0009043216705322266 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.193651479282263
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0009007453918457031 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.24270604916851227
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 30 simulations!
    It/Sub-it: 4/2 Performing simulation 01 from 30 [=                                       ] 3.3%
    Total parallel function evaluation: 0.0010492801666259766 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2121888479008357
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0008363723754882812 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.22738777348965336
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0008270740509033203 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.2043033747150115
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0007681846618652344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.22273175491908226
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.0007491111755371094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.20629914437985059
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 42 simulations!
    It/Sub-it: 5/2 Performing simulation 01 from 42 [                                        ] 2.4%
    Total parallel function evaluation: 0.000904083251953125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.21102767731455996
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0007827281951904297 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06154854206420791
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0008656978607177734 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06290279901760319
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0008988380432128906 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.04911006635536939
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0008177757263183594 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.060455094797190564
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 56 simulations!
    It/Sub-it: 6/2 Performing simulation 01 from 56 [                                        ] 1.8%
    Total parallel function evaluation: 0.0008206367492675781 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06745509529849313
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.000377655029296875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.06229990007453454
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00033164024353027344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.05932528155424208
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.00032806396484375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.09081464149417279
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003826618194580078 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.07414588186460326
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 72 simulations!
    It/Sub-it: 7/2 Performing simulation 01 from 72 [                                        ] 1.4%
    Total parallel function evaluation: 0.0003211498260498047 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.056906010529400655
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003151893615722656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.011350001260062408
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00033020973205566406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01248846645203314
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.00031185150146484375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.009262661524466092
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.000514984130859375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.015668787362038153
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 90 simulations!
    It/Sub-it: 8/2 Performing simulation 01 from 90 [                                        ] 1.1%
    Total parallel function evaluation: 0.0003185272216796875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01265607027947034
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.0003249645233154297 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.011324181770244869
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.000316619873046875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01452316045223306
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00030922889709472656 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01012603055734501
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00032401084899902344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.010976811232021032
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 110 simulations!
    It/Sub-it: 9/2 Performing simulation 001 from 110 [                                        ] 0.9%
    Total parallel function evaluation: 0.00033593177795410156 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.01026497320436617
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0003266334533691406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0018375436519209956
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.00032973289489746094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0008757922107915476
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0005249977111816406 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0010529137911092555
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.000335693359375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.001021832465407841
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    Performing 132 simulations!
    It/Sub-it: 10/2 Performing simulation 001 from 132 [                                        ] 0.8%
    Total parallel function evaluation: 0.0005075931549072266 sec
    Determine gPC coefficients using 'Moore-Penrose' solver ...
    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    -> relative nrmsd error = 0.0029248871199077826
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

   **Total running time of the script:** ( 2 minutes  28.141 seconds)


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
