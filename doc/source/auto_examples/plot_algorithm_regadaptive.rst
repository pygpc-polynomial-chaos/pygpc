.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_plot_algorithm_regadaptive.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_algorithm_regadaptive.py:


Algorithm: RegAdaptive
======================


.. code-block:: default

    import pygpc
    import numpy as np
    from collections import OrderedDict

    fn_results = 'tmp/regadaptive'   # filename of output
    save_session_format = ".hdf5"    # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)








Loading the model and defining the problem
------------------------------------------


.. code-block:: default


    # define model
    model = pygpc.testfunctions.Ishigami()

    # define problem
    parameters = OrderedDict()
    parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
    parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-np.pi, np.pi])
    parameters["x3"] = 0.
    parameters["a"] = 7.
    parameters["b"] = 0.1

    problem = pygpc.Problem(model, parameters)








Setting up the algorithm
------------------------


.. code-block:: default


    # gPC options
    options = dict()
    options["order_start"] = 5
    options["order_end"] = 20
    options["solver"] = "LarsLasso"
    options["interaction_order"] = 2
    options["order_max_norm"] = 0.7
    options["n_cpu"] = 0
    options["adaptive_sampling"] = True
    options["gradient_enhanced"] = True
    options["gradient_calculation"] = "FD_fwd"
    options["gradient_calculation_options"] = {"dx": 0.001, "distance_weight": -2}
    options["fn_results"] = fn_results
    options["save_session_format"] = save_session_format
    options["eps"] = 0.0075
    options["grid"] = pygpc.Random
    options["grid_options"] = None

    # define algorithm
    algorithm = pygpc.RegAdaptive(problem=problem, options=options)








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

    Initializing gPC object...
    Initializing gPC matrix...
    Order/Interaction order: 5/2
    ============================
    Starting adaptive sampling:
    Extending grid from 14 to 14 by 0 sampling points
    Performing simulations 1 to 14
    It/Sub-it: 5/2 Performing simulation 01 from 14 [==                                      ] 7.1%
    Total parallel function evaluation: 0.0047070980072021484 sec
    It/Sub-it: 5/2 Performing simulation 01 from 28 [=                                       ] 3.6%
    Gradient evaluation: 0.0004267692565917969 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
    LOOCV 01 from 14 [==                                      ] 7.1%
    LOOCV 02 from 14 [=====                                   ] 14.3%
    LOOCV 03 from 14 [========                                ] 21.4%
    LOOCV 04 from 14 [===========                             ] 28.6%
    LOOCV 05 from 14 [==============                          ] 35.7%
    LOOCV 06 from 14 [=================                       ] 42.9%
    LOOCV 07 from 14 [====================                    ] 50.0%
    LOOCV 08 from 14 [======================                  ] 57.1%
    LOOCV 09 from 14 [=========================               ] 64.3%
    LOOCV 10 from 14 [============================            ] 71.4%
    LOOCV 11 from 14 [===============================         ] 78.6%
    LOOCV 12 from 14 [==================================      ] 85.7%
    LOOCV 13 from 14 [=====================================   ] 92.9%
    LOOCV 14 from 14 [========================================] 100.0%
    LOOCV computation time: 0.04826092720031738 sec
    -> relative loocv error = 2.438444228689643
    Extending grid from 14 to 15 by 1 sampling points
    Performing simulations 15 to 15
    It/Sub-it: 5/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.006860971450805664 sec
    It/Sub-it: 5/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0005555152893066406 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
    LOOCV 01 from 15 [==                                      ] 6.7%
    LOOCV 02 from 15 [=====                                   ] 13.3%
    LOOCV 03 from 15 [========                                ] 20.0%
    LOOCV 04 from 15 [==========                              ] 26.7%
    LOOCV 05 from 15 [=============                           ] 33.3%
    LOOCV 06 from 15 [================                        ] 40.0%
    LOOCV 07 from 15 [==================                      ] 46.7%
    LOOCV 08 from 15 [=====================                   ] 53.3%
    LOOCV 09 from 15 [========================                ] 60.0%
    LOOCV 10 from 15 [==========================              ] 66.7%
    LOOCV 11 from 15 [=============================           ] 73.3%
    LOOCV 12 from 15 [================================        ] 80.0%
    LOOCV 13 from 15 [==================================      ] 86.7%
    LOOCV 14 from 15 [=====================================   ] 93.3%
    LOOCV 15 from 15 [========================================] 100.0%
    LOOCV computation time: 0.07265472412109375 sec
    -> relative loocv error = 2.40693805625442
    Order/Interaction order: 6/1
    ============================
    Starting adaptive sampling:
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
    LOOCV 01 from 15 [==                                      ] 6.7%
    LOOCV 02 from 15 [=====                                   ] 13.3%
    LOOCV 03 from 15 [========                                ] 20.0%
    LOOCV 04 from 15 [==========                              ] 26.7%
    LOOCV 05 from 15 [=============                           ] 33.3%
    LOOCV 06 from 15 [================                        ] 40.0%
    LOOCV 07 from 15 [==================                      ] 46.7%
    LOOCV 08 from 15 [=====================                   ] 53.3%
    LOOCV 09 from 15 [========================                ] 60.0%
    LOOCV 10 from 15 [==========================              ] 66.7%
    LOOCV 11 from 15 [=============================           ] 73.3%
    LOOCV 12 from 15 [================================        ] 80.0%
    LOOCV 13 from 15 [==================================      ] 86.7%
    LOOCV 14 from 15 [=====================================   ] 93.3%
    LOOCV 15 from 15 [========================================] 100.0%
    LOOCV computation time: 0.0448002815246582 sec
    -> relative loocv error = 0.39947365955182007
    Order/Interaction order: 6/2
    ============================
    Starting adaptive sampling:
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
    LOOCV 01 from 15 [==                                      ] 6.7%
    LOOCV 02 from 15 [=====                                   ] 13.3%
    LOOCV 03 from 15 [========                                ] 20.0%
    LOOCV 04 from 15 [==========                              ] 26.7%
    LOOCV 05 from 15 [=============                           ] 33.3%
    LOOCV 06 from 15 [================                        ] 40.0%
    LOOCV 07 from 15 [==================                      ] 46.7%
    LOOCV 08 from 15 [=====================                   ] 53.3%
    LOOCV 09 from 15 [========================                ] 60.0%
    LOOCV 10 from 15 [==========================              ] 66.7%
    LOOCV 11 from 15 [=============================           ] 73.3%
    LOOCV 12 from 15 [================================        ] 80.0%
    LOOCV 13 from 15 [==================================      ] 86.7%
    LOOCV 14 from 15 [=====================================   ] 93.3%
    LOOCV 15 from 15 [========================================] 100.0%
    LOOCV computation time: 0.04532337188720703 sec
    -> relative loocv error = 0.39947365955182007
    Extending grid from 15 to 16 by 1 sampling points
    Performing simulations 16 to 16
    It/Sub-it: 6/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.0067195892333984375 sec
    It/Sub-it: 6/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0005447864532470703 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.05519843101501465 sec
    -> relative loocv error = 0.38380649898895747
    Order/Interaction order: 7/1
    ============================
    Starting adaptive sampling:
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.07771801948547363 sec
    -> relative loocv error = 0.4465279255981306
    Extending grid from 16 to 18 by 2 sampling points
    Performing simulations 17 to 18
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.00603485107421875 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0005581378936767578 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
    LOOCV 01 from 18 [==                                      ] 5.6%
    LOOCV 02 from 18 [====                                    ] 11.1%
    LOOCV 03 from 18 [======                                  ] 16.7%
    LOOCV 04 from 18 [========                                ] 22.2%
    LOOCV 05 from 18 [===========                             ] 27.8%
    LOOCV 06 from 18 [=============                           ] 33.3%
    LOOCV 07 from 18 [===============                         ] 38.9%
    LOOCV 08 from 18 [=================                       ] 44.4%
    LOOCV 09 from 18 [====================                    ] 50.0%
    LOOCV 10 from 18 [======================                  ] 55.6%
    LOOCV 11 from 18 [========================                ] 61.1%
    LOOCV 12 from 18 [==========================              ] 66.7%
    LOOCV 13 from 18 [============================            ] 72.2%
    LOOCV 14 from 18 [===============================         ] 77.8%
    LOOCV 15 from 18 [=================================       ] 83.3%
    LOOCV 16 from 18 [===================================     ] 88.9%
    LOOCV 17 from 18 [=====================================   ] 94.4%
    LOOCV 18 from 18 [========================================] 100.0%
    LOOCV computation time: 0.07771039009094238 sec
    -> relative loocv error = 0.38874632484120825
    Extending grid from 18 to 20 by 2 sampling points
    Performing simulations 19 to 20
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.006591081619262695 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0006101131439208984 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.11656832695007324 sec
    -> relative loocv error = 0.3344286323913514
    Extending grid from 20 to 22 by 2 sampling points
    Performing simulations 21 to 22
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0066907405853271484 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.000579833984375 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
    LOOCV 01 from 22 [=                                       ] 4.5%
    LOOCV 02 from 22 [===                                     ] 9.1%
    LOOCV 03 from 22 [=====                                   ] 13.6%
    LOOCV 04 from 22 [=======                                 ] 18.2%
    LOOCV 05 from 22 [=========                               ] 22.7%
    LOOCV 06 from 22 [==========                              ] 27.3%
    LOOCV 07 from 22 [============                            ] 31.8%
    LOOCV 08 from 22 [==============                          ] 36.4%
    LOOCV 09 from 22 [================                        ] 40.9%
    LOOCV 10 from 22 [==================                      ] 45.5%
    LOOCV 11 from 22 [====================                    ] 50.0%
    LOOCV 12 from 22 [=====================                   ] 54.5%
    LOOCV 13 from 22 [=======================                 ] 59.1%
    LOOCV 14 from 22 [=========================               ] 63.6%
    LOOCV 15 from 22 [===========================             ] 68.2%
    LOOCV 16 from 22 [=============================           ] 72.7%
    LOOCV 17 from 22 [==============================          ] 77.3%
    LOOCV 18 from 22 [================================        ] 81.8%
    LOOCV 19 from 22 [==================================      ] 86.4%
    LOOCV 20 from 22 [====================================    ] 90.9%
    LOOCV 21 from 22 [======================================  ] 95.5%
    LOOCV 22 from 22 [========================================] 100.0%
    LOOCV computation time: 0.14147114753723145 sec
    -> relative loocv error = 0.2910140233687745
    Order/Interaction order: 7/2
    ============================
    Starting adaptive sampling:
    Extending grid from 22 to 24 by 2 sampling points
    Performing simulations 23 to 24
    It/Sub-it: 7/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.006499290466308594 sec
    It/Sub-it: 7/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0007047653198242188 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.19118642807006836 sec
    -> relative loocv error = 0.2849320348195273
    Order/Interaction order: 8/1
    ============================
    Starting adaptive sampling:
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.2112870216369629 sec
    -> relative loocv error = 0.10138091838500503
    Order/Interaction order: 8/2
    ============================
    Starting adaptive sampling:
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.21493005752563477 sec
    -> relative loocv error = 0.10138091838500503
    Extending grid from 24 to 26 by 2 sampling points
    Performing simulations 25 to 26
    It/Sub-it: 8/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.007595539093017578 sec
    It/Sub-it: 8/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0006909370422363281 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.25082993507385254 sec
    -> relative loocv error = 0.10687897505647138
    Order/Interaction order: 9/1
    ============================
    Starting adaptive sampling:
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.2734076976776123 sec
    -> relative loocv error = 0.08824530424929942
    Order/Interaction order: 9/2
    ============================
    Starting adaptive sampling:
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.2571711540222168 sec
    -> relative loocv error = 0.09514361008222162
    Extending grid from 26 to 28 by 2 sampling points
    Performing simulations 27 to 28
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.006521940231323242 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0006010532379150391 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.25153613090515137 sec
    -> relative loocv error = 0.0720667042659574
    Extending grid from 28 to 30 by 2 sampling points
    Performing simulations 29 to 30
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0072138309478759766 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0006871223449707031 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.2567722797393799 sec
    -> relative loocv error = 0.037076566860627304
    Extending grid from 30 to 32 by 2 sampling points
    Performing simulations 31 to 32
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.007799386978149414 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0006849765777587891 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.2733595371246338 sec
    -> relative loocv error = 0.04801371088384904
    Extending grid from 32 to 34 by 2 sampling points
    Performing simulations 33 to 34
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.006981372833251953 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0006501674652099609 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.2845590114593506 sec
    -> relative loocv error = 0.058386218050903806
    Extending grid from 34 to 36 by 2 sampling points
    Performing simulations 35 to 36
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.007174015045166016 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.00070953369140625 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.2680635452270508 sec
    -> relative loocv error = 0.032904909339131296
    Extending grid from 36 to 38 by 2 sampling points
    Performing simulations 37 to 38
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0070343017578125 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0007264614105224609 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.2510342597961426 sec
    -> relative loocv error = 0.02574636863849896
    Order/Interaction order: 10/1
    =============================
    Starting adaptive sampling:
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.24895048141479492 sec
    -> relative loocv error = 0.019531094308595204
    Order/Interaction order: 10/2
    =============================
    Starting adaptive sampling:
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.24300718307495117 sec
    -> relative loocv error = 0.019383749410442356
    Order/Interaction order: 11/1
    =============================
    Starting adaptive sampling:
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.2669246196746826 sec
    -> relative loocv error = 0.1692559499352664
    Extending grid from 38 to 41 by 3 sampling points
    Performing simulations 39 to 41
    It/Sub-it: 11/1 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.007742404937744141 sec
    It/Sub-it: 11/1 Performing simulation 1 from 6 [======                                  ] 16.7%
    Gradient evaluation: 0.0007090568542480469 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.2667562961578369 sec
    -> relative loocv error = 0.09715415950154638
    Extending grid from 41 to 44 by 3 sampling points
    Performing simulations 42 to 44
    It/Sub-it: 11/1 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.007992029190063477 sec
    It/Sub-it: 11/1 Performing simulation 1 from 6 [======                                  ] 16.7%
    Gradient evaluation: 0.00084686279296875 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.2915642261505127 sec
    -> relative loocv error = 0.038956959612655426
    Extending grid from 44 to 47 by 3 sampling points
    Performing simulations 45 to 47
    It/Sub-it: 11/1 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.007342100143432617 sec
    It/Sub-it: 11/1 Performing simulation 1 from 6 [======                                  ] 16.7%
    Gradient evaluation: 0.0007867813110351562 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.2994248867034912 sec
    -> relative loocv error = 0.03653355704878156
    Order/Interaction order: 11/2
    =============================
    Starting adaptive sampling:
    Extending grid from 47 to 50 by 3 sampling points
    Performing simulations 48 to 50
    It/Sub-it: 11/2 Performing simulation 1 from 3 [=============                           ] 33.3%
    Total parallel function evaluation: 0.007734060287475586 sec
    It/Sub-it: 11/2 Performing simulation 1 from 6 [======                                  ] 16.7%
    Gradient evaluation: 0.0010390281677246094 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
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
    LOOCV computation time: 0.2586994171142578 sec
    -> relative loocv error = 0.0017641200959276557
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

    > Loading gpc session object: tmp/regadaptive.hdf5
    > Loading gpc coeffs: tmp/regadaptive.hdf5
    > Adding results to: tmp/regadaptive.hdf5




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



.. image:: /auto_examples/images/sphx_glr_plot_algorithm_regadaptive_001.png
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


.. image:: /auto_examples/images/sphx_glr_plot_algorithm_regadaptive_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    > Maximum NRMSD (gpc vs original): 0.0037%





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  33.385 seconds)


.. _sphx_glr_download_auto_examples_plot_algorithm_regadaptive.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_algorithm_regadaptive.py <plot_algorithm_regadaptive.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_algorithm_regadaptive.ipynb <plot_algorithm_regadaptive.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
