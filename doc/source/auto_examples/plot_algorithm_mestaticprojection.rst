.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_plot_algorithm_mestaticprojection.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_algorithm_mestaticprojection.py:


Algorithm: MEStaticProjection
=============================


.. code-block:: default

    import pygpc
    from collections import OrderedDict

    fn_results = 'tmp/mestaticprojection'   # filename of output
    save_session_format = ".hdf5"           # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)








Loading the model and defining the problem
------------------------------------------


.. code-block:: default


    # define model
    model = pygpc.testfunctions.DiscontinuousRidgeManufactureDecay()

    # define problem
    parameters = OrderedDict()
    parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
    parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
    problem = pygpc.Problem(model, parameters)








Setting up the algorithm
------------------------


.. code-block:: default


    # gPC options
    options = dict()
    options["method"] = "reg"
    options["solver"] = "Moore-Penrose"
    options["settings"] = None
    options["order"] = [3, 3]
    options["order_max"] = 3
    options["interaction_order"] = 2
    options["matrix_ratio"] = 2
    options["n_cpu"] = 0
    options["gradient_enhanced"] = True
    options["gradient_calculation"] = "FD_fwd"
    options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
    options["n_grid_gradient"] = 5
    options["error_type"] = "nrmsd"
    options["n_samples_validation"] = 1e3
    options["qoi"] = "all"
    options["classifier"] = "learning"
    options["classifier_options"] = {"clusterer": "KMeans",
                                     "n_clusters": 2,
                                     "classifier": "MLPClassifier",
                                     "classifier_solver": "lbfgs"}
    options["fn_results"] = fn_results
    options["save_session_format"] = save_session_format
    options["grid"] = pygpc.Random
    options["grid_options"] = None

    # define algorithm
    algorithm = pygpc.MEStaticProjection(problem=problem, options=options)








Running the gpc
---------------


.. code-block:: default


    # Initialize gPC Session
    session = pygpc.Session(algorithm=algorithm)

    # run gPC algorithm
    session, coeffs, results = session.run()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Performing 5 simulations!
    It/Sub-it: 3/2 Performing simulation 1 from 5 [========                                ] 20.0%
    Total function evaluation: 0.00027108192443847656 sec
    It/Sub-it: 3/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Gradient evaluation: 0.0005342960357666016 sec
    Extending grid from 3 to 8 grid points in domain 0 ...
    Performing 5 additional simulations!
    It/Sub-it: 3/2 Performing simulation 1 from 5 [========                                ] 20.0%
    Total function evaluation: 0.00045013427734375 sec
    It/Sub-it: 3/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Gradient evaluation: 0.0007169246673583984 sec
    Extending grid from 4 to 8 grid points in domain 1 ...
    Performing 4 additional simulations!
    It/Sub-it: 3/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Total function evaluation: 0.0003254413604736328 sec
    It/Sub-it: 3/2 Performing simulation 1 from 8 [=====                                   ] 12.5%
    Gradient evaluation: 0.0006885528564453125 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    It/Sub-it: N/A/N/A Performing simulation 0001 from 1000 [                                        ] 0.1%
    -> relative nrmsd error = 0.08941724013007549
    Extending grid from 6 to 8 grid points in domain 1 ...
    Performing 2 additional simulations!
    It/Sub-it: 3/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total function evaluation: 0.00027251243591308594 sec
    It/Sub-it: 3/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0006110668182373047 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 0.08563092613425938




Postprocessing
--------------


.. code-block:: default


    # read session
    session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)

    # Post-process gPC
    pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                 output_idx=None,
                                 calc_sobol=True,
                                 calc_global_sens=True,
                                 calc_pdf=True,
                                 algorithm="sampling",
                                 n_samples=1e3)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    > Loading gpc session object: tmp/mestaticprojection.hdf5
    > Loading gpc coeffs: tmp/mestaticprojection.hdf5
    > Adding results to: tmp/mestaticprojection.hdf5




Validation
----------
Validate gPC vs original model function (2D-surface)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default

    pygpc.validate_gpc_plot(session=session,
                            coeffs=coeffs,
                            random_vars=list(problem.parameters_random.keys()),
                            n_grid=[51, 51],
                            output_idx=[0],
                            fn_out=None,
                            folder=None,
                            n_cpu=session.n_cpu)



.. image:: /auto_examples/images/sphx_glr_plot_algorithm_mestaticprojection_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    It/Sub-it: N/A/N/A Performing simulation 0001 from 2601 [                                        ] 0.0%




Validate gPC vs original model function (Monte Carlo)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default

    nrmsd = pygpc.validate_gpc_mc(session=session,
                                  coeffs=coeffs,
                                  n_samples=int(1e4),
                                  output_idx=[0],
                                  fn_out=None,
                                  folder=None,
                                  plot=True,
                                  n_cpu=session.n_cpu)

    print("> Maximum NRMSD (gpc vs original): {:.2}%".format(max(nrmsd)))


.. image:: /auto_examples/images/sphx_glr_plot_algorithm_mestaticprojection_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    > Maximum NRMSD (gpc vs original): 0.087%





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  28.883 seconds)


.. _sphx_glr_download_auto_examples_plot_algorithm_mestaticprojection.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_algorithm_mestaticprojection.py <plot_algorithm_mestaticprojection.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_algorithm_mestaticprojection.ipynb <plot_algorithm_mestaticprojection.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
