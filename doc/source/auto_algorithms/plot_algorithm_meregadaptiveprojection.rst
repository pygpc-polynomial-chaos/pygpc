.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_algorithms_plot_algorithm_meregadaptiveprojection.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_algorithms_plot_algorithm_meregadaptiveprojection.py:


Algorithm: MERegAdaptiveProjection
==================================


.. code-block:: default

    # Windows users have to encapsulate the code into a main function to avoid multiprocessing errors.
    # def main():
    import pygpc
    import numpy as np
    from collections import OrderedDict

    fn_results = 'tmp/meregadaptiveprojection'   # filename of output
    save_session_format = ".pkl"                # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)








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
    options["solver"] = "LarsLasso"
    options["settings"] = None
    options["order_start"] = 3
    options["order_end"] = 15
    options["interaction_order"] = 2
    options["matrix_ratio"] = 2
    options["n_cpu"] = 0
    options["projection"] = True
    options["adaptive_sampling"] = True
    options["gradient_enhanced"] = True
    options["gradient_calculation"] = "FD_fwd"
    options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
    options["error_type"] = "nrmsd"
    options["error_norm"] = "absolute"
    options["n_samples_validations"] = "absolute"
    options["qoi"] = 0
    options["classifier"] = "learning"
    options["classifier_options"] = {"clusterer": "KMeans",
                                     "n_clusters": 2,
                                     "classifier": "MLPClassifier",
                                     "classifier_solver": "lbfgs"}
    options["n_samples_discontinuity"] = 12
    options["eps"] = 0.75
    options["n_grid_init"] = 20
    options["backend"] = "omp"
    options["fn_results"] = fn_results
    options["save_session_format"] = save_session_format
    options["grid"] = pygpc.Random
    options["grid_options"] = {"seed": 1}

    # define algorithm
    algorithm = pygpc.MERegAdaptiveProjection(problem=problem, options=options)








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

    Performing 20 initial simulations!
    It/Sub-it: 3/2 Performing simulation 01 from 20 [==                                      ] 5.0%
    Total function evaluation: 0.006393909454345703 sec
    It/Sub-it: 3/2 Performing simulation 01 from 40 [=                                       ] 2.5%
    Gradient evaluation: 0.0010640621185302734 sec
    Determining gPC approximation for QOI #0:
    =========================================
    Determining gPC domains ...
    Determining validation set of size 10000 for NRMSD error calculation ...
    Refining domain boundary ...
    Performing 12 simulations to refine discontinuity location!
    It/Sub-it: Domain boundary/N/A Performing simulation 01 from 12 [===                                     ] 8.3%
    Total function evaluation: 0.008487462997436523 sec
    It/Sub-it: Domain boundary/N/A Performing simulation 01 from 24 [=                                       ] 4.2%
    Gradient evaluation: 0.0006842613220214844 sec
    Updating classifier ...
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
    -> Domain: 0 absolute nrmsd error = 0.5265864517311108
    -> Domain: 1 absolute nrmsd error = 0.6688101402146892
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...




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

    > Loading gpc session object: tmp/meregadaptiveprojection.pkl
    > Loading gpc coeffs: tmp/meregadaptiveprojection.hdf5
    > Adding results to: tmp/meregadaptiveprojection.hdf5




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



.. image:: /auto_algorithms/images/sphx_glr_plot_algorithm_meregadaptiveprojection_001.png
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



.. image:: /auto_algorithms/images/sphx_glr_plot_algorithm_meregadaptiveprojection_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    > Maximum NRMSD (gpc vs original): 0.1%





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  6.134 seconds)


.. _sphx_glr_download_auto_algorithms_plot_algorithm_meregadaptiveprojection.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_algorithm_meregadaptiveprojection.py <plot_algorithm_meregadaptiveprojection.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_algorithm_meregadaptiveprojection.ipynb <plot_algorithm_meregadaptiveprojection.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
