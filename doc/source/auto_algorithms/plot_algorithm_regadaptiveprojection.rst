.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_algorithms_plot_algorithm_regadaptiveprojection.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_algorithms_plot_algorithm_regadaptiveprojection.py:


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
    Total function evaluation: 0.0030748844146728516 sec
    It/Sub-it: 2/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Gradient evaluation: 0.001102447509765625 sec
    Order/Interaction order: 2/2
    ============================
    Extending grid from 5 to 6 by 1 sampling points
    Performing simulations 6 to 6
    It/Sub-it: 2/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.003629446029663086 sec
    Gradient evaluation: 0.0005719661712646484 sec
    Dimension of reduced problem: 2
    Determine gPC coefficients using 'Moore-Penrose' solver (gradient enhanced)...
    LOOCV 1 from 6 [======                                  ] 16.7%
    LOOCV 2 from 6 [=============                           ] 33.3%
    LOOCV 3 from 6 [====================                    ] 50.0%
    LOOCV 4 from 6 [==========================              ] 66.7%
    LOOCV 5 from 6 [=================================       ] 83.3%
    LOOCV 6 from 6 [========================================] 100.0%
    LOOCV computation time: 0.001665353775024414 sec
    -> relative loocv error = 0.9213533835561233
    Order/Interaction order: 3/1
    ============================
    Extending grid from 6 to 16 by 10 sampling points
    Performing simulations 7 to 16
    It/Sub-it: 3/1 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.0031790733337402344 sec
    Gradient evaluation: 0.0012786388397216797 sec
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
    LOOCV computation time: 0.004656553268432617 sec
    -> relative loocv error = 0.6605458102970638
    Order/Interaction order: 3/2
    ============================
    Extending grid from 16 to 20 by 4 sampling points
    Performing simulations 17 to 20
    It/Sub-it: 3/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Total parallel function evaluation: 0.0027816295623779297 sec
    Gradient evaluation: 0.0019273757934570312 sec
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
    LOOCV computation time: 0.0056056976318359375 sec
    -> relative loocv error = 0.7598247952190154
    Order/Interaction order: 4/1
    ============================
    Extending grid from 20 to 24 by 4 sampling points
    Performing simulations 21 to 24
    It/Sub-it: 4/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Total parallel function evaluation: 0.0032732486724853516 sec
    Gradient evaluation: 0.0024797916412353516 sec
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
    LOOCV computation time: 0.007004261016845703 sec
    -> relative loocv error = 0.5843434567886424
    Order/Interaction order: 4/2
    ============================
    Extending grid from 24 to 30 by 6 sampling points
    Performing simulations 25 to 30
    It/Sub-it: 4/2 Performing simulation 1 from 6 [======                                  ] 16.7%
    Total parallel function evaluation: 0.002947568893432617 sec
    Gradient evaluation: 0.0036046504974365234 sec
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
    LOOCV computation time: 0.008507728576660156 sec
    -> relative loocv error = 1.0133730159943064
    Order/Interaction order: 5/1
    ============================
    Extending grid from 30 to 34 by 4 sampling points
    Performing simulations 31 to 34
    It/Sub-it: 5/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Total parallel function evaluation: 0.0029909610748291016 sec
    Gradient evaluation: 0.004152059555053711 sec
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
    LOOCV computation time: 0.00952291488647461 sec
    -> relative loocv error = 0.21286249864308587
    Order/Interaction order: 5/2
    ============================
    Extending grid from 34 to 42 by 8 sampling points
    Performing simulations 35 to 42
    It/Sub-it: 5/2 Performing simulation 1 from 8 [=====                                   ] 12.5%
    Total parallel function evaluation: 0.0028426647186279297 sec
    Gradient evaluation: 0.005892276763916016 sec
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
    LOOCV computation time: 0.014688491821289062 sec
    -> relative loocv error = 0.09629723274223112
    Order/Interaction order: 6/1
    ============================
    Extending grid from 42 to 46 by 4 sampling points
    Performing simulations 43 to 46
    It/Sub-it: 6/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Total parallel function evaluation: 0.003147602081298828 sec
    Gradient evaluation: 0.0061798095703125 sec
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
    LOOCV computation time: 0.012925863265991211 sec
    -> relative loocv error = 0.32533983301403346
    Order/Interaction order: 6/2
    ============================
    Extending grid from 46 to 56 by 10 sampling points
    Performing simulations 47 to 56
    It/Sub-it: 6/2 Performing simulation 01 from 10 [====                                    ] 10.0%
    Total parallel function evaluation: 0.003072023391723633 sec
    Gradient evaluation: 0.0075075626373291016 sec
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
    LOOCV computation time: 0.01371312141418457 sec
    -> relative loocv error = 0.0730587274064606
    Order/Interaction order: 7/1
    ============================
    Extending grid from 56 to 60 by 4 sampling points
    Performing simulations 57 to 60
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Total parallel function evaluation: 0.004889726638793945 sec
    Gradient evaluation: 0.015430688858032227 sec
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
    LOOCV computation time: 0.02870488166809082 sec
    -> relative loocv error = 0.016629945168130524
    Order/Interaction order: 7/2
    ============================
    Extending grid from 60 to 72 by 12 sampling points
    Performing simulations 61 to 72
    It/Sub-it: 7/2 Performing simulation 01 from 12 [===                                     ] 8.3%
    Total parallel function evaluation: 0.003160715103149414 sec
    Gradient evaluation: 0.009650707244873047 sec
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
    LOOCV computation time: 0.03191995620727539 sec
    -> relative loocv error = 0.020598412108697676
    Order/Interaction order: 8/1
    ============================
    Extending grid from 72 to 76 by 4 sampling points
    Performing simulations 73 to 76
    It/Sub-it: 8/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Total parallel function evaluation: 0.004383087158203125 sec
    Gradient evaluation: 0.01884293556213379 sec
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
    LOOCV computation time: 0.05741381645202637 sec
    -> relative loocv error = 0.02550741378425769
    Order/Interaction order: 8/2
    ============================
    Extending grid from 76 to 90 by 14 sampling points
    Performing simulations 77 to 90
    It/Sub-it: 8/2 Performing simulation 01 from 14 [==                                      ] 7.1%
    Total parallel function evaluation: 0.005419492721557617 sec
    Gradient evaluation: 0.024228572845458984 sec
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
    LOOCV computation time: 0.1956641674041748 sec
    -> relative loocv error = 0.006034857100203359
    Order/Interaction order: 9/1
    ============================
    Extending grid from 90 to 94 by 4 sampling points
    Performing simulations 91 to 94
    It/Sub-it: 9/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Total parallel function evaluation: 0.0061130523681640625 sec
    Gradient evaluation: 0.026105880737304688 sec
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
    LOOCV computation time: 0.2218165397644043 sec
    -> relative loocv error = 0.00044549272742646555
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



.. image:: /auto_algorithms/images/sphx_glr_plot_algorithm_regadaptiveprojection_001.png
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


.. image:: /auto_algorithms/images/sphx_glr_plot_algorithm_regadaptiveprojection_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    > Maximum NRMSD (gpc vs original): 0.03%





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  15.113 seconds)


.. _sphx_glr_download_auto_algorithms_plot_algorithm_regadaptiveprojection.py:


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
