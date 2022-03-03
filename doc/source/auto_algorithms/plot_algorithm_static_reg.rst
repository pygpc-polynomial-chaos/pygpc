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








We will run the gPC with 10 initial simulations and see how well the approximation is


.. code-block:: default

    options["n_grid"] = 10








We will use adaptive sampling here, which runs additional simulations if the approximation error is higher than eps


.. code-block:: default

    options["eps"] = 1e-3
    options["adaptive_sampling"] = True

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

    Performing 10 simulations!
    It/Sub-it: 9/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.00031495094299316406 sec
    Gradient evaluation: 0.0003292560577392578 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 1.2383417472210787
    Extending grid from 10 to 13 by 3 sampling points
    Performing 3 simulations!
    It/Sub-it: 9/2 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.00035762786865234375 sec
    Gradient evaluation: 0.00044608116149902344 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 1.1707822203475031
    Extending grid from 13 to 16 by 3 sampling points
    Performing 3 simulations!
    It/Sub-it: 9/2 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.00036835670471191406 sec
    Gradient evaluation: 0.0005092620849609375 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 1.143669885681281
    Extending grid from 16 to 19 by 3 sampling points
    Performing 3 simulations!
    It/Sub-it: 9/2 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.00028014183044433594 sec
    Gradient evaluation: 0.0006506443023681641 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 1.1338956758277758
    Extending grid from 19 to 22 by 3 sampling points
    Performing 3 simulations!
    It/Sub-it: 9/2 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.00027251243591308594 sec
    Gradient evaluation: 0.0006422996520996094 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 1.0507393608371591
    Extending grid from 22 to 25 by 3 sampling points
    Performing 3 simulations!
    It/Sub-it: 9/2 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.0002834796905517578 sec
    Gradient evaluation: 0.0007231235504150391 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 1.0450078846912312
    Extending grid from 25 to 28 by 3 sampling points
    Performing 3 simulations!
    It/Sub-it: 9/2 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.00027823448181152344 sec
    Gradient evaluation: 0.0007951259613037109 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 0.939045452482192
    Extending grid from 28 to 31 by 3 sampling points
    Performing 3 simulations!
    It/Sub-it: 9/2 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.0005109310150146484 sec
    Gradient evaluation: 0.001474618911743164 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 0.8970113909538253
    Extending grid from 31 to 34 by 3 sampling points
    Performing 3 simulations!
    It/Sub-it: 9/2 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.0005476474761962891 sec
    Gradient evaluation: 0.0010459423065185547 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 0.8649732897266762
    Extending grid from 34 to 37 by 3 sampling points
    Performing 3 simulations!
    It/Sub-it: 9/2 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.0004744529724121094 sec
    Gradient evaluation: 0.0011947154998779297 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 0.8620414300424817
    Extending grid from 37 to 40 by 3 sampling points
    Performing 3 simulations!
    It/Sub-it: 9/2 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.0004591941833496094 sec
    Gradient evaluation: 0.0021278858184814453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 0.75933106117979
    Extending grid from 40 to 43 by 3 sampling points
    Performing 3 simulations!
    It/Sub-it: 9/2 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.00048351287841796875 sec
    Gradient evaluation: 0.001931905746459961 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 0.6808892118796706
    Extending grid from 43 to 46 by 3 sampling points
    Performing 3 simulations!
    It/Sub-it: 9/2 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.0004837512969970703 sec
    Gradient evaluation: 0.0025284290313720703 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 0.5332952917915144
    Extending grid from 46 to 49 by 3 sampling points
    Performing 3 simulations!
    It/Sub-it: 9/2 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.0005266666412353516 sec
    Gradient evaluation: 0.002330780029296875 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 0.41267297645708223
    Extending grid from 49 to 52 by 3 sampling points
    Performing 3 simulations!
    It/Sub-it: 9/2 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.0002949237823486328 sec
    Gradient evaluation: 0.0014619827270507812 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 0.1884838131989019
    Extending grid from 52 to 55 by 3 sampling points
    Performing 3 simulations!
    It/Sub-it: 9/2 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.0005905628204345703 sec
    Gradient evaluation: 0.0026466846466064453 sec
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    -> relative nrmsd error = 3.6983801878890816e-06




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
                                 algorithm="standard")





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
                            output_idx=0,
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

    > Maximum NRMSD (gpc vs original): 3.2e-06%





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  3.116 seconds)


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
