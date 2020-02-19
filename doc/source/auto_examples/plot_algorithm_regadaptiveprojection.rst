.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_plot_algorithm_regadaptiveprojection.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_algorithm_regadaptiveprojection.py:


Algorithm: RegAdaptiveProjection
================================


.. code-block:: default

    import pygpc
    import numpy as np
    from collections import OrderedDict

    fn_results = 'tmp/regadaptiveprojection'   # filename of output
    save_session_format = ".hdf5"              # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)








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
    options["order_start"] = 2
    options["order_end"] = 15
    options["interaction_order"] = 2
    options["solver"] = "Moore-Penrose"
    options["settings"] = None
    options["seed"] = 1
    options["matrix_ratio"] = 2
    options["n_cpu"] = 0
    options["fn_results"] = fn_results
    options["save_session_format"] = save_session_format
    options["adaptive_sampling"] = False
    options["gradient_enhanced"] = True
    options["gradient_calculation"] = "FD_1st"
    options["gradient_calculation_options"] = {"dx": 0.5, "distance_weight": -2}
    options["n_grid_gradient"] = 5
    options["qoi"] = 0
    options["error_type"] = "loocv"
    options["eps"] = 1e-3
    options["grid"] = pygpc.Random
    options["grid_options"] = None

    # define algorithm
    algorithm = pygpc.RegAdaptiveProjection(problem=problem, options=options)








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
    It/Sub-it: 2/2 Performing simulation 1 from 5 [========                                ] 20.0%
    Total function evaluation: 0.004971027374267578 sec
    It/Sub-it: 2/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Gradient evaluation: 0.0005168914794921875 sec
    Order/Interaction order: 2/2
    ============================
    Extending grid from 5 to 6 by 1 sampling points
    Performing simulations 6 to 6
    It/Sub-it: 2/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.007063865661621094 sec
    Gradient evaluation: 0.0004069805145263672 sec
    Dimension of reduced problem: 2
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    LOOCV 1 from 6 [======                                  ] 16.7%
    LOOCV 2 from 6 [=============                           ] 33.3%
    LOOCV 3 from 6 [====================                    ] 50.0%
    LOOCV 4 from 6 [==========================              ] 66.7%
    LOOCV 5 from 6 [=================================       ] 83.3%
    LOOCV 6 from 6 [========================================] 100.0%
    LOOCV computation time: 0.0013766288757324219 sec
    -> relative loocv error = 0.9213533835561233
    Order/Interaction order: 3/1
    ============================
    Extending grid from 6 to 16 by 10 sampling points
    Performing simulations 7 to 16
    It/Sub-it: 3/1 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.006643772125244141 sec
    Gradient evaluation: 0.0011048316955566406 sec
    Dimension of reduced problem: 2
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    LOOCV 01 from 16 [==                                      ] 6.2%
    LOOCV 02 from 16 [=====                                   ] 12.5%
    LOOCV 03 from 16 [=======                                 ] 18.8%
    LOOCV 04 from 16 [==========                              ] 25.0%
    LOOCV 05 from 16 [============                            ] 31.2%
    LOOCV 06 from 16 [===============                         ] 37.5%
    LOOCV 07 from 16 [=================                       ] 43.8%
    LOOCV 08 from 16 [====================                    ] 50.0%
    LOOCV 09 from 16 [======================                  ] 56.2%
    LOOCV 10 from 16 [=========================               ] 62.5%
    LOOCV 11 from 16 [===========================             ] 68.8%
    LOOCV 12 from 16 [==============================          ] 75.0%
    LOOCV 13 from 16 [================================        ] 81.2%
    LOOCV 14 from 16 [===================================     ] 87.5%
    LOOCV 15 from 16 [=====================================   ] 93.8%
    LOOCV 16 from 16 [========================================] 100.0%
    LOOCV computation time: 0.004225969314575195 sec
    -> relative loocv error = 0.6605458102970638
    Order/Interaction order: 3/2
    ============================
    Extending grid from 16 to 20 by 4 sampling points
    Performing simulations 17 to 20
    It/Sub-it: 3/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Total parallel function evaluation: 0.007695436477661133 sec
    Gradient evaluation: 0.0017216205596923828 sec
    Dimension of reduced problem: 2
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    LOOCV 01 from 20 [==                                      ] 5.0%
    LOOCV 02 from 20 [====                                    ] 10.0%
    LOOCV 03 from 20 [======                                  ] 15.0%
    LOOCV 04 from 20 [========                                ] 20.0%
    LOOCV 05 from 20 [==========                              ] 25.0%
    LOOCV 06 from 20 [============                            ] 30.0%
    LOOCV 07 from 20 [==============                          ] 35.0%
    LOOCV 08 from 20 [================                        ] 40.0%
    LOOCV 09 from 20 [==================                      ] 45.0%
    LOOCV 10 from 20 [====================                    ] 50.0%
    LOOCV 11 from 20 [======================                  ] 55.0%
    LOOCV 12 from 20 [========================                ] 60.0%
    LOOCV 13 from 20 [==========================              ] 65.0%
    LOOCV 14 from 20 [============================            ] 70.0%
    LOOCV 15 from 20 [==============================          ] 75.0%
    LOOCV 16 from 20 [================================        ] 80.0%
    LOOCV 17 from 20 [==================================      ] 85.0%
    LOOCV 18 from 20 [====================================    ] 90.0%
    LOOCV 19 from 20 [======================================  ] 95.0%
    LOOCV 20 from 20 [========================================] 100.0%
    LOOCV computation time: 0.005218982696533203 sec
    -> relative loocv error = 0.7598247952190154
    Order/Interaction order: 4/1
    ============================
    Extending grid from 20 to 24 by 4 sampling points
    Performing simulations 21 to 24
    It/Sub-it: 4/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Total parallel function evaluation: 0.007445335388183594 sec
    Gradient evaluation: 0.0022497177124023438 sec
    Dimension of reduced problem: 2
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    LOOCV 01 from 24 [=                                       ] 4.2%
    LOOCV 02 from 24 [===                                     ] 8.3%
    LOOCV 03 from 24 [=====                                   ] 12.5%
    LOOCV 04 from 24 [======                                  ] 16.7%
    LOOCV 05 from 24 [========                                ] 20.8%
    LOOCV 06 from 24 [==========                              ] 25.0%
    LOOCV 07 from 24 [===========                             ] 29.2%
    LOOCV 08 from 24 [=============                           ] 33.3%
    LOOCV 09 from 24 [===============                         ] 37.5%
    LOOCV 10 from 24 [================                        ] 41.7%
    LOOCV 11 from 24 [==================                      ] 45.8%
    LOOCV 12 from 24 [====================                    ] 50.0%
    LOOCV 13 from 24 [=====================                   ] 54.2%
    LOOCV 14 from 24 [=======================                 ] 58.3%
    LOOCV 15 from 24 [=========================               ] 62.5%
    LOOCV 16 from 24 [==========================              ] 66.7%
    LOOCV 17 from 24 [============================            ] 70.8%
    LOOCV 18 from 24 [==============================          ] 75.0%
    LOOCV 19 from 24 [===============================         ] 79.2%
    LOOCV 20 from 24 [=================================       ] 83.3%
    LOOCV 21 from 24 [===================================     ] 87.5%
    LOOCV 22 from 24 [====================================    ] 91.7%
    LOOCV 23 from 24 [======================================  ] 95.8%
    LOOCV 24 from 24 [========================================] 100.0%
    LOOCV computation time: 0.0075418949127197266 sec
    -> relative loocv error = 0.5843434567886424
    Order/Interaction order: 4/2
    ============================
    Extending grid from 24 to 30 by 6 sampling points
    Performing simulations 25 to 30
    It/Sub-it: 4/2 Performing simulation 1 from 6 [======                                  ] 16.7%
    Total parallel function evaluation: 0.008378982543945312 sec
    Gradient evaluation: 0.0030248165130615234 sec
    Dimension of reduced problem: 2
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
    LOOCV computation time: 0.008553028106689453 sec
    -> relative loocv error = 1.068382159949837
    Order/Interaction order: 5/1
    ============================
    Extending grid from 30 to 34 by 4 sampling points
    Performing simulations 31 to 34
    It/Sub-it: 5/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Total parallel function evaluation: 0.008210420608520508 sec
    Gradient evaluation: 0.004120826721191406 sec
    Dimension of reduced problem: 2
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
    LOOCV computation time: 0.009468555450439453 sec
    -> relative loocv error = 0.2072620999202444
    Order/Interaction order: 5/2
    ============================
    Extending grid from 34 to 42 by 8 sampling points
    Performing simulations 35 to 42
    It/Sub-it: 5/2 Performing simulation 1 from 8 [=====                                   ] 12.5%
    Total parallel function evaluation: 0.008188962936401367 sec
    Gradient evaluation: 0.0051784515380859375 sec
    Dimension of reduced problem: 2
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
    LOOCV computation time: 0.008766651153564453 sec
    -> relative loocv error = 0.2648837300455835
    Order/Interaction order: 6/1
    ============================
    Extending grid from 42 to 46 by 4 sampling points
    Performing simulations 43 to 46
    It/Sub-it: 6/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Total parallel function evaluation: 0.008173227310180664 sec
    Gradient evaluation: 0.006327390670776367 sec
    Dimension of reduced problem: 2
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
    LOOCV computation time: 0.010375499725341797 sec
    -> relative loocv error = 0.3554472481289659
    Order/Interaction order: 6/2
    ============================
    Extending grid from 46 to 56 by 10 sampling points
    Performing simulations 47 to 56
    It/Sub-it: 6/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.008069515228271484 sec
    Gradient evaluation: 0.006972551345825195 sec
    Dimension of reduced problem: 2
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
    LOOCV computation time: 0.01117706298828125 sec
    -> relative loocv error = 0.4415264786443005
    Order/Interaction order: 7/1
    ============================
    Extending grid from 56 to 60 by 4 sampling points
    Performing simulations 57 to 60
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Total parallel function evaluation: 0.006445407867431641 sec
    Gradient evaluation: 0.007134675979614258 sec
    Dimension of reduced problem: 2
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
    LOOCV computation time: 0.012077808380126953 sec
    -> relative loocv error = 0.00813014022318355
    Order/Interaction order: 7/2
    ============================
    Extending grid from 60 to 72 by 12 sampling points
    Performing simulations 61 to 72
    It/Sub-it: 7/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.007128715515136719 sec
    Gradient evaluation: 0.008369207382202148 sec
    Dimension of reduced problem: 2
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
    LOOCV computation time: 0.015253543853759766 sec
    -> relative loocv error = 0.019545388505566225
    Order/Interaction order: 8/1
    ============================
    Extending grid from 72 to 76 by 4 sampling points
    Performing simulations 73 to 76
    It/Sub-it: 8/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Total parallel function evaluation: 0.0074117183685302734 sec
    Gradient evaluation: 0.009162664413452148 sec
    Dimension of reduced problem: 2
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
    LOOCV computation time: 0.017266273498535156 sec
    -> relative loocv error = 0.016651105871756684
    Order/Interaction order: 8/2
    ============================
    Extending grid from 76 to 90 by 14 sampling points
    Performing simulations 77 to 90
    It/Sub-it: 8/2 Performing simulation 01 from 14 [==                                      ] 7.1%
    Total parallel function evaluation: 0.007938623428344727 sec
    Gradient evaluation: 0.011148214340209961 sec
    Dimension of reduced problem: 2
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
    LOOCV computation time: 0.04645347595214844 sec
    -> relative loocv error = 0.009339242127129423
    Order/Interaction order: 9/1
    ============================
    Extending grid from 90 to 94 by 4 sampling points
    Performing simulations 91 to 94
    It/Sub-it: 9/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Total parallel function evaluation: 0.008446931838989258 sec
    Gradient evaluation: 0.023250818252563477 sec
    Dimension of reduced problem: 2
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
    LOOCV computation time: 0.04962468147277832 sec
    -> relative loocv error = 0.0057408694887532115
    Order/Interaction order: 9/2
    ============================
    Extending grid from 94 to 110 by 16 sampling points
    Performing simulations 95 to 110
    It/Sub-it: 9/2 Performing simulation 01 from 16 [==                                      ] 6.2%
    Total parallel function evaluation: 0.00871133804321289 sec
    Gradient evaluation: 0.013361454010009766 sec
    Dimension of reduced problem: 2
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
    LOOCV computation time: 0.05423617362976074 sec
    -> relative loocv error = 0.0006909438275922728
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...




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

    > Loading gpc session object: tmp/regadaptiveprojection.hdf5
    > Loading gpc coeffs: tmp/regadaptiveprojection.hdf5
    > Adding results to: tmp/regadaptiveprojection.hdf5




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



.. image:: /auto_examples/images/sphx_glr_plot_algorithm_regadaptiveprojection_001.png
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


.. image:: /auto_examples/images/sphx_glr_plot_algorithm_regadaptiveprojection_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    > Maximum NRMSD (gpc vs original): 0.028%





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  30.154 seconds)


.. _sphx_glr_download_auto_examples_plot_algorithm_regadaptiveprojection.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_algorithm_regadaptiveprojection.py <plot_algorithm_regadaptiveprojection.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_algorithm_regadaptiveprojection.ipynb <plot_algorithm_regadaptiveprojection.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
