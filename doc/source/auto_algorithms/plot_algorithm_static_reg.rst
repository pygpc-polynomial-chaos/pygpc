.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_algorithms_plot_algorithm_static_reg.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_algorithms_plot_algorithm_static_reg.py:


Algorithm: Static (Regression)
==============================


.. code-block:: default

    # Windows users have to encapsulate the code into a main function to avoid multiprocessing errors.
    # def main():
    import pygpc
    from collections import OrderedDict

    fn_results = 'tmp/static_reg'   # filename of output
    save_session_format = ".pkl"    # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)








Loading the model and defining the problem
------------------------------------------


.. code-block:: default


    # define model
    model = pygpc.testfunctions.Peaks()

    # define problem
    parameters = OrderedDict()
    parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    parameters["x2"] = 1.25
    parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    problem = pygpc.Problem(model, parameters)








Setting up the algorithm
------------------------


.. code-block:: default


    # gPC options
    options = dict()
    options["method"] = "reg"
    options["solver"] = "Moore-Penrose"
    options["settings"] = None
    options["order"] = [9, 9]
    options["order_max"] = 9
    options["interaction_order"] = 2
    options["matrix_ratio"] = 20
    options["error_type"] = "nrmsd"
    options["n_samples_validation"] = 1e3
    options["n_cpu"] = 0
    options["fn_results"] = fn_results
    options["save_session_format"] = save_session_format
    options["gradient_enhanced"] = True
    options["gradient_calculation"] = "FD_1st2nd"
    options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
    options["backend"] = "omp"
    options["grid"] = pygpc.Random
    options["grid_options"] = {"seed": 1}
    options["n_grid"] = 1000

    # initialize algorithm
    algorithm = pygpc.Static(problem=problem, options=options)








Running the gpc
---------------


.. code-block:: default


    # initialize gPC Session
    session = pygpc.Session(algorithm=algorithm)

    # run gPC algorithm
    session, coeffs, results = session.run()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Performing 1000 simulations!
    It/Sub-it: 9/2 Performing simulation 0001 from 1000 [                                        ] 0.1%
    Total parallel function evaluation: 0.0006158351898193359 sec
    Gradient evaluation: 0.18607640266418457 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 0.001260059497119485
    Extending grid from 1000 to 1003 by 3 sampling points
    Performing 3 simulations!
    It/Sub-it: 9/2 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.0005822181701660156 sec
    Gradient evaluation: 0.2162613868713379 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 0.0012592677313095098




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
                                 algorithm="standard",
                                 n_samples=1e3)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    > Loading gpc session object: tmp/static_reg.pkl
    > Loading gpc coeffs: tmp/static_reg.hdf5
    > Adding results to: tmp/static_reg.hdf5




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



.. image:: /auto_algorithms/images/sphx_glr_plot_algorithm_static_reg_001.png
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



.. image:: /auto_algorithms/images/sphx_glr_plot_algorithm_static_reg_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    > Maximum NRMSD (gpc vs original): 0.0012%





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  1.952 seconds)


.. _sphx_glr_download_auto_algorithms_plot_algorithm_static_reg.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_algorithm_static_reg.py <plot_algorithm_static_reg.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_algorithm_static_reg.ipynb <plot_algorithm_static_reg.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
