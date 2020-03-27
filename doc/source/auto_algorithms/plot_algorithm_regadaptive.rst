.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_algorithms_plot_algorithm_regadaptive.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_algorithms_plot_algorithm_regadaptive.py:


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
    Total parallel function evaluation: 0.009347677230834961 sec
    It/Sub-it: 5/2 Performing simulation 01 from 28 [=                                       ] 3.6%
    Gradient evaluation: 0.0005791187286376953 sec
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
    LOOCV computation time: 0.07821393013000488 sec
    -> relative loocv error = 6.346391583734324
    Extending grid from 14 to 15 by 1 sampling points
    Performing simulations 15 to 15
    It/Sub-it: 5/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.002409696578979492 sec
    It/Sub-it: 5/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0006697177886962891 sec
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
    LOOCV computation time: 0.11508679389953613 sec
    -> relative loocv error = 24.546614303594165
    Extending grid from 15 to 16 by 1 sampling points
    Performing simulations 16 to 16
    It/Sub-it: 5/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.0025353431701660156 sec
    It/Sub-it: 5/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0006875991821289062 sec
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
    LOOCV computation time: 0.09920573234558105 sec
    -> relative loocv error = 21.645434447819717
    Extending grid from 16 to 17 by 1 sampling points
    Performing simulations 17 to 17
    It/Sub-it: 5/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.0025315284729003906 sec
    It/Sub-it: 5/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0006327629089355469 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
    LOOCV 01 from 17 [==                                      ] 5.9%
    LOOCV 02 from 17 [====                                    ] 11.8%
    LOOCV 03 from 17 [=======                                 ] 17.6%
    LOOCV 04 from 17 [=========                               ] 23.5%
    LOOCV 05 from 17 [===========                             ] 29.4%
    LOOCV 06 from 17 [==============                          ] 35.3%
    LOOCV 07 from 17 [================                        ] 41.2%
    LOOCV 08 from 17 [==================                      ] 47.1%
    LOOCV 09 from 17 [=====================                   ] 52.9%
    LOOCV 10 from 17 [=======================                 ] 58.8%
    LOOCV 11 from 17 [=========================               ] 64.7%
    LOOCV 12 from 17 [============================            ] 70.6%
    LOOCV 13 from 17 [==============================          ] 76.5%
    LOOCV 14 from 17 [================================        ] 82.4%
    LOOCV 15 from 17 [===================================     ] 88.2%
    LOOCV 16 from 17 [=====================================   ] 94.1%
    LOOCV 17 from 17 [========================================] 100.0%
    LOOCV computation time: 0.07691359519958496 sec
    -> relative loocv error = 5.092349121271938
    Extending grid from 17 to 18 by 1 sampling points
    Performing simulations 18 to 18
    It/Sub-it: 5/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.002673625946044922 sec
    It/Sub-it: 5/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0007059574127197266 sec
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
    LOOCV computation time: 0.07849693298339844 sec
    -> relative loocv error = 5.579273264602323
    Order/Interaction order: 6/1
    ============================
    Starting adaptive sampling:
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
    LOOCV computation time: 0.09241461753845215 sec
    -> relative loocv error = 0.4680176315553291
    Order/Interaction order: 6/2
    ============================
    Starting adaptive sampling:
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
    LOOCV computation time: 0.09682774543762207 sec
    -> relative loocv error = 0.4680176315553291
    Extending grid from 18 to 19 by 1 sampling points
    Performing simulations 19 to 19
    It/Sub-it: 6/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.0025229454040527344 sec
    It/Sub-it: 6/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0009763240814208984 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
    LOOCV 01 from 19 [==                                      ] 5.3%
    LOOCV 02 from 19 [====                                    ] 10.5%
    LOOCV 03 from 19 [======                                  ] 15.8%
    LOOCV 04 from 19 [========                                ] 21.1%
    LOOCV 05 from 19 [==========                              ] 26.3%
    LOOCV 06 from 19 [============                            ] 31.6%
    LOOCV 07 from 19 [==============                          ] 36.8%
    LOOCV 08 from 19 [================                        ] 42.1%
    LOOCV 09 from 19 [==================                      ] 47.4%
    LOOCV 10 from 19 [=====================                   ] 52.6%
    LOOCV 11 from 19 [=======================                 ] 57.9%
    LOOCV 12 from 19 [=========================               ] 63.2%
    LOOCV 13 from 19 [===========================             ] 68.4%
    LOOCV 14 from 19 [=============================           ] 73.7%
    LOOCV 15 from 19 [===============================         ] 78.9%
    LOOCV 16 from 19 [=================================       ] 84.2%
    LOOCV 17 from 19 [===================================     ] 89.5%
    LOOCV 18 from 19 [=====================================   ] 94.7%
    LOOCV 19 from 19 [========================================] 100.0%
    LOOCV computation time: 0.11323046684265137 sec
    -> relative loocv error = 0.9616313412705704
    Extending grid from 19 to 20 by 1 sampling points
    Performing simulations 20 to 20
    It/Sub-it: 6/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.002396106719970703 sec
    It/Sub-it: 6/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.000637054443359375 sec
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
    LOOCV computation time: 0.1788170337677002 sec
    -> relative loocv error = 1.0284087038324914
    Extending grid from 20 to 21 by 1 sampling points
    Performing simulations 21 to 21
    It/Sub-it: 6/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.0027472972869873047 sec
    It/Sub-it: 6/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0006711483001708984 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
    LOOCV 01 from 21 [=                                       ] 4.8%
    LOOCV 02 from 21 [===                                     ] 9.5%
    LOOCV 03 from 21 [=====                                   ] 14.3%
    LOOCV 04 from 21 [=======                                 ] 19.0%
    LOOCV 05 from 21 [=========                               ] 23.8%
    LOOCV 06 from 21 [===========                             ] 28.6%
    LOOCV 07 from 21 [=============                           ] 33.3%
    LOOCV 08 from 21 [===============                         ] 38.1%
    LOOCV 09 from 21 [=================                       ] 42.9%
    LOOCV 10 from 21 [===================                     ] 47.6%
    LOOCV 11 from 21 [====================                    ] 52.4%
    LOOCV 12 from 21 [======================                  ] 57.1%
    LOOCV 13 from 21 [========================                ] 61.9%
    LOOCV 14 from 21 [==========================              ] 66.7%
    LOOCV 15 from 21 [============================            ] 71.4%
    LOOCV 16 from 21 [==============================          ] 76.2%
    LOOCV 17 from 21 [================================        ] 81.0%
    LOOCV 18 from 21 [==================================      ] 85.7%
    LOOCV 19 from 21 [====================================    ] 90.5%
    LOOCV 20 from 21 [======================================  ] 95.2%
    LOOCV 21 from 21 [========================================] 100.0%
    LOOCV computation time: 0.14316034317016602 sec
    -> relative loocv error = 1.7862592056172635
    Extending grid from 21 to 22 by 1 sampling points
    Performing simulations 22 to 22
    It/Sub-it: 6/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.002537965774536133 sec
    It/Sub-it: 6/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0007114410400390625 sec
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
    LOOCV computation time: 0.13703083992004395 sec
    -> relative loocv error = 1.3263092944577417
    Extending grid from 22 to 23 by 1 sampling points
    Performing simulations 23 to 23
    It/Sub-it: 6/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.0036509037017822266 sec
    It/Sub-it: 6/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0011289119720458984 sec
    Determine gPC coefficients using 'LarsLasso' solver (gradient enhanced)...
    LOOCV 01 from 23 [=                                       ] 4.3%
    LOOCV 02 from 23 [===                                     ] 8.7%
    LOOCV 03 from 23 [=====                                   ] 13.0%
    LOOCV 04 from 23 [======                                  ] 17.4%
    LOOCV 05 from 23 [========                                ] 21.7%
    LOOCV 06 from 23 [==========                              ] 26.1%
    LOOCV 07 from 23 [============                            ] 30.4%
    LOOCV 08 from 23 [=============                           ] 34.8%
    LOOCV 09 from 23 [===============                         ] 39.1%
    LOOCV 10 from 23 [=================                       ] 43.5%
    LOOCV 11 from 23 [===================                     ] 47.8%
    LOOCV 12 from 23 [====================                    ] 52.2%
    LOOCV 13 from 23 [======================                  ] 56.5%
    LOOCV 14 from 23 [========================                ] 60.9%
    LOOCV 15 from 23 [==========================              ] 65.2%
    LOOCV 16 from 23 [===========================             ] 69.6%
    LOOCV 17 from 23 [=============================           ] 73.9%
    LOOCV 18 from 23 [===============================         ] 78.3%
    LOOCV 19 from 23 [=================================       ] 82.6%
    LOOCV 20 from 23 [==================================      ] 87.0%
    LOOCV 21 from 23 [====================================    ] 91.3%
    LOOCV 22 from 23 [======================================  ] 95.7%
    LOOCV 23 from 23 [========================================] 100.0%
    LOOCV computation time: 0.15355992317199707 sec
    -> relative loocv error = 1.2074126505018035
    Extending grid from 23 to 24 by 1 sampling points
    Performing simulations 24 to 24
    It/Sub-it: 6/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.0026044845581054688 sec
    It/Sub-it: 6/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0007188320159912109 sec
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
    LOOCV computation time: 0.1382904052734375 sec
    -> relative loocv error = 0.9634505556480429
    Extending grid from 24 to 25 by 1 sampling points
    Performing simulations 25 to 25
    It/Sub-it: 6/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.0024933815002441406 sec
    It/Sub-it: 6/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0006978511810302734 sec
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
    LOOCV computation time: 0.13787841796875 sec
    -> relative loocv error = 1.1224915247164058
    Extending grid from 25 to 26 by 1 sampling points
    Performing simulations 26 to 26
    It/Sub-it: 6/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.0026154518127441406 sec
    It/Sub-it: 6/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0011730194091796875 sec
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
    LOOCV computation time: 0.1349034309387207 sec
    -> relative loocv error = 1.0758104736911205
    Order/Interaction order: 7/1
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
    LOOCV computation time: 0.3668184280395508 sec
    -> relative loocv error = 3.2123069410099236
    Extending grid from 26 to 28 by 2 sampling points
    Performing simulations 27 to 28
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.002978801727294922 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0008060932159423828 sec
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
    LOOCV computation time: 0.32221221923828125 sec
    -> relative loocv error = 2.7089773419726395
    Extending grid from 28 to 30 by 2 sampling points
    Performing simulations 29 to 30
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0030062198638916016 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.001074075698852539 sec
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
    LOOCV computation time: 0.3425891399383545 sec
    -> relative loocv error = 2.029274642121952
    Extending grid from 30 to 32 by 2 sampling points
    Performing simulations 31 to 32
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.002859354019165039 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0007545948028564453 sec
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
    LOOCV computation time: 0.29767346382141113 sec
    -> relative loocv error = 0.36573554128509483
    Extending grid from 32 to 34 by 2 sampling points
    Performing simulations 33 to 34
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0034742355346679688 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0012607574462890625 sec
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
    LOOCV computation time: 0.28299713134765625 sec
    -> relative loocv error = 1.995753768359557
    Extending grid from 34 to 36 by 2 sampling points
    Performing simulations 35 to 36
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0025644302368164062 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.00090789794921875 sec
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
    LOOCV computation time: 0.21989130973815918 sec
    -> relative loocv error = 1.6189879084773597
    Extending grid from 36 to 38 by 2 sampling points
    Performing simulations 37 to 38
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0025625228881835938 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0008158683776855469 sec
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
    LOOCV computation time: 0.17846417427062988 sec
    -> relative loocv error = 1.1732911134738055
    Extending grid from 38 to 40 by 2 sampling points
    Performing simulations 39 to 40
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0034973621368408203 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0011072158813476562 sec
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
    LOOCV computation time: 0.19246482849121094 sec
    -> relative loocv error = 1.0691741771681635
    Order/Interaction order: 7/2
    ============================
    Starting adaptive sampling:
    Extending grid from 40 to 42 by 2 sampling points
    Performing simulations 41 to 42
    It/Sub-it: 7/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0026073455810546875 sec
    It/Sub-it: 7/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0009179115295410156 sec
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
    LOOCV computation time: 0.18038368225097656 sec
    -> relative loocv error = 0.8371078401532531
    Order/Interaction order: 8/1
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
    LOOCV computation time: 0.3228278160095215 sec
    -> relative loocv error = 0.3273853018169959
    Order/Interaction order: 8/2
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
    LOOCV computation time: 0.30828261375427246 sec
    -> relative loocv error = 0.10661940866404261
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
    LOOCV computation time: 0.367281436920166 sec
    -> relative loocv error = 0.23500914098459408
    Extending grid from 42 to 44 by 2 sampling points
    Performing simulations 43 to 44
    It/Sub-it: 9/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.003448009490966797 sec
    It/Sub-it: 9/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0013852119445800781 sec
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
    LOOCV computation time: 0.4295799732208252 sec
    -> relative loocv error = 0.20127589356178738
    Extending grid from 44 to 46 by 2 sampling points
    Performing simulations 45 to 46
    It/Sub-it: 9/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.003410816192626953 sec
    It/Sub-it: 9/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0013968944549560547 sec
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
    LOOCV computation time: 0.4400947093963623 sec
    -> relative loocv error = 0.1808714908710107
    Order/Interaction order: 9/2
    ============================
    Starting adaptive sampling:
    Extending grid from 46 to 48 by 2 sampling points
    Performing simulations 47 to 48
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.003352642059326172 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.00109100341796875 sec
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
    LOOCV computation time: 0.4319291114807129 sec
    -> relative loocv error = 0.11756318276547569
    Extending grid from 48 to 50 by 2 sampling points
    Performing simulations 49 to 50
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0026412010192871094 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0009555816650390625 sec
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
    LOOCV computation time: 0.4381227493286133 sec
    -> relative loocv error = 0.4887065191533469
    Extending grid from 50 to 52 by 2 sampling points
    Performing simulations 51 to 52
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0025599002838134766 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0009207725524902344 sec
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
    LOOCV computation time: 0.39443445205688477 sec
    -> relative loocv error = 0.10213702025261334
    Extending grid from 52 to 54 by 2 sampling points
    Performing simulations 53 to 54
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.007559776306152344 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0009453296661376953 sec
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
    LOOCV computation time: 0.32290101051330566 sec
    -> relative loocv error = 0.43704361840744477
    Extending grid from 54 to 56 by 2 sampling points
    Performing simulations 55 to 56
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.002544879913330078 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0009987354278564453 sec
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
    LOOCV computation time: 0.33071327209472656 sec
    -> relative loocv error = 0.05350848455273294
    Extending grid from 56 to 58 by 2 sampling points
    Performing simulations 57 to 58
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0024785995483398438 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0010082721710205078 sec
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
    LOOCV computation time: 0.33894801139831543 sec
    -> relative loocv error = 0.4860040378645886
    Extending grid from 58 to 60 by 2 sampling points
    Performing simulations 59 to 60
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0034089088439941406 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0010569095611572266 sec
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
    LOOCV computation time: 0.3211233615875244 sec
    -> relative loocv error = 0.10656677945495856
    Extending grid from 60 to 62 by 2 sampling points
    Performing simulations 61 to 62
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0024476051330566406 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0010058879852294922 sec
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
    LOOCV computation time: 0.39766430854797363 sec
    -> relative loocv error = 0.1317806896189883
    Extending grid from 62 to 64 by 2 sampling points
    Performing simulations 63 to 64
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.00516963005065918 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.002425670623779297 sec
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
    LOOCV computation time: 0.44802355766296387 sec
    -> relative loocv error = 0.33645025557451996
    Extending grid from 64 to 66 by 2 sampling points
    Performing simulations 65 to 66
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.003900766372680664 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0018126964569091797 sec
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
    LOOCV computation time: 0.33710408210754395 sec
    -> relative loocv error = 0.20412859200528743
    Extending grid from 66 to 68 by 2 sampling points
    Performing simulations 67 to 68
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.005835533142089844 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0012331008911132812 sec
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
    LOOCV computation time: 0.3952672481536865 sec
    -> relative loocv error = 0.026620539623192673
    Extending grid from 68 to 70 by 2 sampling points
    Performing simulations 69 to 70
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.003027200698852539 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0011172294616699219 sec
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
    LOOCV computation time: 0.23850154876708984 sec
    -> relative loocv error = 0.057218391207881164
    Extending grid from 70 to 72 by 2 sampling points
    Performing simulations 71 to 72
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.002731800079345703 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0010824203491210938 sec
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
    LOOCV computation time: 0.23493599891662598 sec
    -> relative loocv error = 0.09457618947700538
    Extending grid from 72 to 74 by 2 sampling points
    Performing simulations 73 to 74
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0030066967010498047 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0012524127960205078 sec
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
    LOOCV computation time: 0.24086594581604004 sec
    -> relative loocv error = 0.10580381488459009
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
    LOOCV computation time: 0.41880154609680176 sec
    -> relative loocv error = 0.0018111552862871924
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



.. image:: /auto_algorithms/images/sphx_glr_plot_algorithm_regadaptive_001.png
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


.. image:: /auto_algorithms/images/sphx_glr_plot_algorithm_regadaptive_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    It/Sub-it: N/A/N/A Performing simulation 00001 from 10000 [                                        ] 0.0%
    > Maximum NRMSD (gpc vs original): 0.0015%





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  27.631 seconds)


.. _sphx_glr_download_auto_algorithms_plot_algorithm_regadaptive.py:


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
