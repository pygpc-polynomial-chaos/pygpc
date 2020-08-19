.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_algorithms_plot_algorithm_mestatic.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_algorithms_plot_algorithm_mestatic.py:


Algorithm: MEStatic
===================


.. code-block:: default

    # Windows users have to encapsulate the code into a main function to avoid multiprocessing errors.
    # def main():
    import pygpc
    from collections import OrderedDict

    fn_results = 'tmp/mestatic'       # filename of output
    save_session_format = ".pkl"     # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)








Loading the model and defining the problem
------------------------------------------


.. code-block:: default


    # define model
    model = pygpc.testfunctions.SurfaceCoverageSpecies()

    # define problem
    parameters = OrderedDict()
    parameters["rho_0"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
    parameters["beta"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 20])
    parameters["alpha"] = 1.
    problem = pygpc.Problem(model, parameters)








Setting up the algorithm
------------------------


.. code-block:: default


    # gPC options
    options = dict()
    options["method"] = "reg"
    options["solver"] = "Moore-Penrose"
    options["settings"] = None
    options["order"] = [10, 10]
    options["order_max"] = 10
    options["interaction_order"] = 2
    options["matrix_ratio"] = 2
    options["n_cpu"] = 0
    options["gradient_enhanced"] = True
    options["gradient_calculation"] = "FD_2nd"
    options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
    options["error_type"] = "loocv"
    options["qoi"] = "all"
    options["classifier"] = "learning"
    options["classifier_options"] = {"clusterer": "KMeans",
                                     "n_clusters": 2,
                                     "classifier": "MLPClassifier",
                                     "classifier_solver": "lbfgs"}
    options["fn_results"] = fn_results
    options["save_session_format"] = save_session_format
    options["grid"] = pygpc.Random
    options["grid_options"] = {"seed": 1}
    options["n_grid"] = 1000
    options["adaptive_sampling"] = False

    # define algorithm
    algorithm = pygpc.MEStatic(problem=problem, options=options)








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

    Determining gPC approximation for QOI #0:
    =========================================
    Performing 1000 simulations!
    It/Sub-it: 10/2 Performing simulation 0001 from 1000 [                                        ] 0.1%
    Total function evaluation: 1.298065185546875 sec
    Gradient evaluation: 0.0721883773803711 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    LOOCV 01 from 25 [=                                       ] 4.0%
    LOOCV 02 from 25 [===                                     ] 8.0%
    LOOCV 03 from 25 [====                                    ] 12.0%
    LOOCV 04 from 25 [======                                  ] 16.0%
    LOOCV 05 from 25 [========                                ] 20.0%
    LOOCV 06 from 25 [=========                               ] 24.0%
    LOOCV 07 from 25 [===========                             ] 28.0%
    LOOCV 08 from 25 [============                            ] 32.0%
    LOOCV 09 from 25 [==============                          ] 36.0%
    LOOCV 10 from 25 [================                        ] 40.0%
    LOOCV 11 from 25 [=================                       ] 44.0%
    LOOCV 12 from 25 [===================                     ] 48.0%
    LOOCV 13 from 25 [====================                    ] 52.0%
    LOOCV 14 from 25 [======================                  ] 56.0%
    LOOCV 15 from 25 [========================                ] 60.0%
    LOOCV 16 from 25 [=========================               ] 64.0%
    LOOCV 17 from 25 [===========================             ] 68.0%
    LOOCV 18 from 25 [============================            ] 72.0%
    LOOCV 19 from 25 [==============================          ] 76.0%
    LOOCV 20 from 25 [================================        ] 80.0%
    LOOCV 21 from 25 [=================================       ] 84.0%
    LOOCV 22 from 25 [===================================     ] 88.0%
    LOOCV 23 from 25 [====================================    ] 92.0%
    LOOCV 24 from 25 [======================================  ] 96.0%
    LOOCV 25 from 25 [========================================] 100.0%
    LOOCV computation time: 0.1267223358154297 sec
    -> relative loocv error = 0.018999762367704152
    LOOCV 01 from 25 [=                                       ] 4.0%
    LOOCV 02 from 25 [===                                     ] 8.0%
    LOOCV 03 from 25 [====                                    ] 12.0%
    LOOCV 04 from 25 [======                                  ] 16.0%
    LOOCV 05 from 25 [========                                ] 20.0%
    LOOCV 06 from 25 [=========                               ] 24.0%
    LOOCV 07 from 25 [===========                             ] 28.0%
    LOOCV 08 from 25 [============                            ] 32.0%
    LOOCV 09 from 25 [==============                          ] 36.0%
    LOOCV 10 from 25 [================                        ] 40.0%
    LOOCV 11 from 25 [=================                       ] 44.0%
    LOOCV 12 from 25 [===================                     ] 48.0%
    LOOCV 13 from 25 [====================                    ] 52.0%
    LOOCV 14 from 25 [======================                  ] 56.0%
    LOOCV 15 from 25 [========================                ] 60.0%
    LOOCV 16 from 25 [=========================               ] 64.0%
    LOOCV 17 from 25 [===========================             ] 68.0%
    LOOCV 18 from 25 [============================            ] 72.0%
    LOOCV 19 from 25 [==============================          ] 76.0%
    LOOCV 20 from 25 [================================        ] 80.0%
    LOOCV 21 from 25 [=================================       ] 84.0%
    LOOCV 22 from 25 [===================================     ] 88.0%
    LOOCV 23 from 25 [====================================    ] 92.0%
    LOOCV 24 from 25 [======================================  ] 96.0%
    LOOCV 25 from 25 [========================================] 100.0%
    LOOCV computation time: 0.12355804443359375 sec
    LOOCV 01 from 25 [=                                       ] 4.0%
    LOOCV 02 from 25 [===                                     ] 8.0%
    LOOCV 03 from 25 [====                                    ] 12.0%
    LOOCV 04 from 25 [======                                  ] 16.0%
    LOOCV 05 from 25 [========                                ] 20.0%
    LOOCV 06 from 25 [=========                               ] 24.0%
    LOOCV 07 from 25 [===========                             ] 28.0%
    LOOCV 08 from 25 [============                            ] 32.0%
    LOOCV 09 from 25 [==============                          ] 36.0%
    LOOCV 10 from 25 [================                        ] 40.0%
    LOOCV 11 from 25 [=================                       ] 44.0%
    LOOCV 12 from 25 [===================                     ] 48.0%
    LOOCV 13 from 25 [====================                    ] 52.0%
    LOOCV 14 from 25 [======================                  ] 56.0%
    LOOCV 15 from 25 [========================                ] 60.0%
    LOOCV 16 from 25 [=========================               ] 64.0%
    LOOCV 17 from 25 [===========================             ] 68.0%
    LOOCV 18 from 25 [============================            ] 72.0%
    LOOCV 19 from 25 [==============================          ] 76.0%
    LOOCV 20 from 25 [================================        ] 80.0%
    LOOCV 21 from 25 [=================================       ] 84.0%
    LOOCV 22 from 25 [===================================     ] 88.0%
    LOOCV 23 from 25 [====================================    ] 92.0%
    LOOCV 24 from 25 [======================================  ] 96.0%
    LOOCV 25 from 25 [========================================] 100.0%
    LOOCV computation time: 0.10218286514282227 sec




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

    > Loading gpc session object: tmp/mestatic.pkl
    > Loading gpc coeffs: tmp/mestatic.hdf5
    > Adding results to: tmp/mestatic.hdf5




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



.. image:: /auto_algorithms/images/sphx_glr_plot_algorithm_mestatic_001.png
    :class: sphx-glr-single-img





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

    # On Windows subprocesses will import (i.e. execute) the main module at start.
    # You need to insert an if __name__ == '__main__': guard in the main module to avoid
    # creating subprocesses recursively.
    #
    # if __name__ == '__main__':
    #     main()



.. image:: /auto_algorithms/images/sphx_glr_plot_algorithm_mestatic_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    > Maximum NRMSD (gpc vs original): 0.0078%





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  25.172 seconds)


.. _sphx_glr_download_auto_algorithms_plot_algorithm_mestatic.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_algorithm_mestatic.py <plot_algorithm_mestatic.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_algorithm_mestatic.ipynb <plot_algorithm_mestatic.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
