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
    Total parallel function evaluation: 0.0065267086029052734 sec
    It/Sub-it: 5/2 Performing simulation 01 from 28 [=                                       ] 3.6%
    Gradient evaluation: 0.0011358261108398438 sec
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
    LOOCV computation time: 0.09157347679138184 sec
    -> relative loocv error = 51.9863838405722
    Extending grid from 14 to 15 by 1 sampling points
    Performing simulations 15 to 15
    It/Sub-it: 5/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.002535104751586914 sec
    It/Sub-it: 5/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0006215572357177734 sec
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
    LOOCV computation time: 0.09639239311218262 sec
    -> relative loocv error = 357.20791981944313
    Extending grid from 15 to 16 by 1 sampling points
    Performing simulations 16 to 16
    It/Sub-it: 5/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.0023958683013916016 sec
    It/Sub-it: 5/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0006017684936523438 sec
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
    LOOCV computation time: 0.08013701438903809 sec
    -> relative loocv error = 27.671499511298745
    Extending grid from 16 to 17 by 1 sampling points
    Performing simulations 17 to 17
    It/Sub-it: 5/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.002380847930908203 sec
    It/Sub-it: 5/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0006299018859863281 sec
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
    LOOCV computation time: 0.0791623592376709 sec
    -> relative loocv error = 22.75046957574746
    Order/Interaction order: 6/1
    ============================
    Starting adaptive sampling:
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
    LOOCV computation time: 0.11909079551696777 sec
    -> relative loocv error = 0.6089839519932241
    Order/Interaction order: 6/2
    ============================
    Starting adaptive sampling:
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
    LOOCV computation time: 0.11872267723083496 sec
    -> relative loocv error = 0.6089839519932241
    Extending grid from 17 to 18 by 1 sampling points
    Performing simulations 18 to 18
    It/Sub-it: 6/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.0024080276489257812 sec
    It/Sub-it: 6/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0006377696990966797 sec
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
    LOOCV computation time: 0.11734485626220703 sec
    -> relative loocv error = 0.7448634644156447
    Extending grid from 18 to 19 by 1 sampling points
    Performing simulations 19 to 19
    It/Sub-it: 6/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.0023126602172851562 sec
    It/Sub-it: 6/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0006575584411621094 sec
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
    LOOCV computation time: 0.1328141689300537 sec
    -> relative loocv error = 0.5883031327755891
    Extending grid from 19 to 20 by 1 sampling points
    Performing simulations 20 to 20
    It/Sub-it: 6/2 Performing simulation 1 from 1 [========================================] 100.0%
    Total parallel function evaluation: 0.002363443374633789 sec
    It/Sub-it: 6/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Gradient evaluation: 0.0006644725799560547 sec
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
    LOOCV computation time: 0.14393091201782227 sec
    -> relative loocv error = 0.6215772359932087
    Order/Interaction order: 7/1
    ============================
    Starting adaptive sampling:
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
    LOOCV computation time: 0.14589500427246094 sec
    -> relative loocv error = 0.7696451151156614
    Extending grid from 20 to 22 by 2 sampling points
    Performing simulations 21 to 22
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0023245811462402344 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0006871223449707031 sec
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
    LOOCV computation time: 0.1995983123779297 sec
    -> relative loocv error = 0.9222118586440365
    Extending grid from 22 to 24 by 2 sampling points
    Performing simulations 23 to 24
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0023360252380371094 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0006897449493408203 sec
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
    LOOCV computation time: 0.2348923683166504 sec
    -> relative loocv error = 2.5699641755047824
    Extending grid from 24 to 26 by 2 sampling points
    Performing simulations 25 to 26
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0023093223571777344 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0007026195526123047 sec
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
    LOOCV computation time: 0.2600576877593994 sec
    -> relative loocv error = 8.732087429925889
    Extending grid from 26 to 28 by 2 sampling points
    Performing simulations 27 to 28
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.002286195755004883 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0007615089416503906 sec
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
    LOOCV computation time: 0.19543933868408203 sec
    -> relative loocv error = 3.24722241893233
    Extending grid from 28 to 30 by 2 sampling points
    Performing simulations 29 to 30
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.002549409866333008 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0007648468017578125 sec
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
    LOOCV computation time: 0.19535398483276367 sec
    -> relative loocv error = 1.5711564721773181
    Extending grid from 30 to 32 by 2 sampling points
    Performing simulations 31 to 32
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0023238658905029297 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0007598400115966797 sec
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
    LOOCV computation time: 0.1552600860595703 sec
    -> relative loocv error = 0.7480467526575861
    Extending grid from 32 to 34 by 2 sampling points
    Performing simulations 33 to 34
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0024178028106689453 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0007758140563964844 sec
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
    LOOCV computation time: 0.1397113800048828 sec
    -> relative loocv error = 0.2767175081875749
    Extending grid from 34 to 36 by 2 sampling points
    Performing simulations 35 to 36
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.002376556396484375 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0008039474487304688 sec
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
    LOOCV computation time: 0.11386227607727051 sec
    -> relative loocv error = 0.16238714620609965
    Extending grid from 36 to 38 by 2 sampling points
    Performing simulations 37 to 38
    It/Sub-it: 7/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0023071765899658203 sec
    It/Sub-it: 7/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0007994174957275391 sec
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
    LOOCV computation time: 0.12223005294799805 sec
    -> relative loocv error = 0.21921302653480834
    Order/Interaction order: 7/2
    ============================
    Starting adaptive sampling:
    Extending grid from 38 to 40 by 2 sampling points
    Performing simulations 39 to 40
    It/Sub-it: 7/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.00211334228515625 sec
    It/Sub-it: 7/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0008587837219238281 sec
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
    LOOCV computation time: 0.1122589111328125 sec
    -> relative loocv error = 0.25082731421724724
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
    LOOCV computation time: 0.20055174827575684 sec
    -> relative loocv error = 0.08077804600212488
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
    LOOCV computation time: 0.1963481903076172 sec
    -> relative loocv error = 0.08833104100239765
    Extending grid from 40 to 42 by 2 sampling points
    Performing simulations 41 to 42
    It/Sub-it: 8/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.002386331558227539 sec
    It/Sub-it: 8/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0008370876312255859 sec
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
    LOOCV computation time: 0.20212340354919434 sec
    -> relative loocv error = 0.05206808236121816
    Extending grid from 42 to 44 by 2 sampling points
    Performing simulations 43 to 44
    It/Sub-it: 8/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0024247169494628906 sec
    It/Sub-it: 8/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0008764266967773438 sec
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
    LOOCV computation time: 0.1892092227935791 sec
    -> relative loocv error = 0.044352842685294915
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
    LOOCV computation time: 0.31962132453918457 sec
    -> relative loocv error = 0.1938110688658174
    Extending grid from 44 to 46 by 2 sampling points
    Performing simulations 45 to 46
    It/Sub-it: 9/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.002346038818359375 sec
    It/Sub-it: 9/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0008642673492431641 sec
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
    LOOCV computation time: 0.2903604507446289 sec
    -> relative loocv error = 0.09627534849806313
    Extending grid from 46 to 48 by 2 sampling points
    Performing simulations 47 to 48
    It/Sub-it: 9/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0023043155670166016 sec
    It/Sub-it: 9/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0008955001831054688 sec
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
    LOOCV computation time: 0.30542802810668945 sec
    -> relative loocv error = 0.14096042200516773
    Extending grid from 48 to 50 by 2 sampling points
    Performing simulations 49 to 50
    It/Sub-it: 9/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0023283958435058594 sec
    It/Sub-it: 9/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0009510517120361328 sec
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
    LOOCV computation time: 0.2988123893737793 sec
    -> relative loocv error = 0.062409928888233414
    Extending grid from 50 to 52 by 2 sampling points
    Performing simulations 51 to 52
    It/Sub-it: 9/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.002299785614013672 sec
    It/Sub-it: 9/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0009415149688720703 sec
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
    LOOCV computation time: 0.3042416572570801 sec
    -> relative loocv error = 0.040804404584236925
    Extending grid from 52 to 54 by 2 sampling points
    Performing simulations 53 to 54
    It/Sub-it: 9/1 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0023179054260253906 sec
    It/Sub-it: 9/1 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0009326934814453125 sec
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
    LOOCV computation time: 0.2920799255371094 sec
    -> relative loocv error = 0.022285946625815196
    Order/Interaction order: 9/2
    ============================
    Starting adaptive sampling:
    Extending grid from 54 to 56 by 2 sampling points
    Performing simulations 55 to 56
    It/Sub-it: 9/2 Performing simulation 1 from 2 [====================                    ] 50.0%
    Total parallel function evaluation: 0.0021851062774658203 sec
    It/Sub-it: 9/2 Performing simulation 1 from 4 [==========                              ] 25.0%
    Gradient evaluation: 0.0009348392486572266 sec
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
    LOOCV computation time: 0.2953042984008789 sec
    -> relative loocv error = 0.01391934873851469
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
    LOOCV computation time: 0.26936912536621094 sec
    -> relative loocv error = 0.00545877769161675
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
    > Maximum NRMSD (gpc vs original): 0.0013%





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  16.401 seconds)


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
