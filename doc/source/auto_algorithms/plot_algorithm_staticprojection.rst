.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_algorithms_plot_algorithm_staticprojection.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_algorithms_plot_algorithm_staticprojection.py:


Algorithm: StaticProjection
===========================


.. code-block:: default

    import pygpc
    from collections import OrderedDict

    fn_results = 'tmp/staticprojection'   # filename of output
    save_session_format = ".hdf5"         # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)








Loading the model and defining the problem
------------------------------------------


.. code-block:: default


    # define model
    model = pygpc.testfunctions.GenzOscillatory()

    # define problem
    parameters = OrderedDict()
    parameters["x1"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    parameters["x2"] = pygpc.Beta(pdf_shape=[1., 1.], pdf_limits=[0., 1.])
    problem = pygpc.Problem(model, parameters)








Setting up the algorithm
------------------------


.. code-block:: default


    # gPC options
    options = dict()
    options["method"] = "reg"
    options["solver"] = "Moore-Penrose"
    options["settings"] = None
    options["order"] = [10]
    options["order_max"] = 10
    options["interaction_order"] = 1
    options["n_cpu"] = 0
    options["error_type"] = "nrmsd"
    options["n_samples_validation"] = 1e3
    options["error_norm"] = "relative"
    options["matrix_ratio"] = 2
    options["qoi"] = 0
    options["n_grid_gradient"] = 10
    options["fn_results"] = fn_results
    options["save_session_format"] = save_session_format
    options["gradient_enhanced"] = True
    options["gradient_calculation"] = "FD_fwd"
    options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
    options["grid"] = pygpc.Random
    options["grid_options"] = None

    # define algorithm
    algorithm = pygpc.StaticProjection(problem=problem, options=options)








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

    Performing 10 simulations!
    It/Sub-it: 10/1 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total function evaluation: 0.0003161430358886719 sec
    It/Sub-it: 10/1 Performing simulation 01 from 20 [==                                      ] 5.0%
    Gradient evaluation: 0.001135110855102539 sec
    Extending grid from 10 to 22 grid points ...
    Performing 22 additional simulations!
    It/Sub-it: 10/1 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total function evaluation: 0.00043320655822753906 sec
    It/Sub-it: 10/1 Performing simulation 01 from 24 [=                                       ] 4.2%
    Gradient evaluation: 0.0015056133270263672 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    It/Sub-it: N/A/N/A Performing simulation 0001 from 1000 [                                        ] 0.1%
    -> relative nrmsd error = 0.009663685503452138




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

    > Loading gpc session object: tmp/staticprojection.hdf5
    > Loading gpc coeffs: tmp/staticprojection.hdf5
    > Adding results to: tmp/staticprojection.hdf5




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



.. image:: /auto_algorithms/images/sphx_glr_plot_algorithm_staticprojection_001.png
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


.. image:: /auto_algorithms/images/sphx_glr_plot_algorithm_staticprojection_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    > Maximum NRMSD (gpc vs original): 0.01%





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  6.150 seconds)


.. _sphx_glr_download_auto_algorithms_plot_algorithm_staticprojection.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_algorithm_staticprojection.py <plot_algorithm_staticprojection.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_algorithm_staticprojection.ipynb <plot_algorithm_staticprojection.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
