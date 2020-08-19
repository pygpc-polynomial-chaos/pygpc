.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_algorithms_plot_algorithm_static_quad.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_algorithms_plot_algorithm_static_quad.py:


Algorithm: Static (Quadrature)
==============================


.. code-block:: default

    # Windows users have to encapsulate the code into a main function to avoid multiprocessing errors.
    # def main():
    import pygpc
    from collections import OrderedDict

    fn_results = 'tmp/static_quad'   # filename of output
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
    options["method"] = "quad"
    options["solver"] = "NumInt"
    options["settings"] = None
    options["order"] = [9, 9]
    options["order_max"] = 9
    options["interaction_order"] = 2
    options["error_type"] = "nrmsd"
    options["n_samples_validation"] = 1e3
    options["n_cpu"] = 0
    options["fn_results"] = fn_results
    options["save_session_format"] = save_session_format
    options["backend"] = "omp"
    options["grid"] = None
    options["grid_options"] = {"grid_type": ["jacobi", "jacobi"], "n_dim": [9, 9]}

    # generate grid
    grid = pygpc.TensorGrid(parameters_random=problem.parameters_random,
                            options=options["grid_options"])

    # initialize algorithm
    algorithm = pygpc.Static(problem=problem, options=options, grid=grid)








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

    Performing 81 simulations!
    It/Sub-it: 9/2 Performing simulation 01 from 81 [                                        ] 1.2%
    Total parallel function evaluation: 0.0004096031188964844 sec
    Determine gPC coefficients using 'NumInt' solver ...
    -> relative nrmsd error = 3.410388047395797e-08




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

    > Loading gpc session object: tmp/static_quad.pkl
    > Loading gpc coeffs: tmp/static_quad.hdf5
    > Adding results to: tmp/static_quad.hdf5




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



.. image:: /auto_algorithms/images/sphx_glr_plot_algorithm_static_quad_001.png
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



.. image:: /auto_algorithms/images/sphx_glr_plot_algorithm_static_quad_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    > Maximum NRMSD (gpc vs original): 3.6e-08%





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  1.372 seconds)


.. _sphx_glr_download_auto_algorithms_plot_algorithm_static_quad.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_algorithm_static_quad.py <plot_algorithm_static_quad.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_algorithm_static_quad.ipynb <plot_algorithm_static_quad.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
