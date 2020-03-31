.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_features_plot_parallelization.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_features_plot_parallelization.py:


Parallel processing capabilities of pygpc
=========================================

pygpc is capable of to evaluate multiple sampling points, i.e. multiple model instances, in parallel.
Depending on your model and its hardware requirements there exist three ways to evaluate your model
controlled by the algorithm options "n_cpu":

- n_cpu = 0 : Use this option if your model is capable of to evaluate sampling points in parallel. In this way,
  arrays are passed to your model for each parameter
- n_cpu = 1 : The model is called in serial for every sampling point. A single floating point number is passed for
  each parameter.
- n_cpu > 1 : A multiprocessing.Pool will be opened and n_cpu sampling points are calculated in parallel.
  In each thread, a single floating point number is passed for each parameter.

Example
^^^^^^^


.. code-block:: default


    import time
    import pygpc
    import numpy as np
    import multiprocessing
    import seaborn as sns
    from matplotlib import pyplot as plt
    from collections import OrderedDict

    SurfaceCoverageSpecies = pygpc.SurfaceCoverageSpecies()

    # generate grid with 1000 sampling points
    grid = pygpc.Random(parameters_random=SurfaceCoverageSpecies.problem.parameters_random, n_grid=100)

    # define different values for n_cpu
    n_cpu_list = [0, 1, multiprocessing.cpu_count()]

    t_eval = dict()

    # evaluate model with different values for n_cpu
    for n_cpu in n_cpu_list:
        # initialize computation class; this is done in the algorithm with options["n_cpu"]
        com = pygpc.Computation(n_cpu=n_cpu)

        # run model and determine computation time
        t_n_cpu = []

        start = time.time()
        res = com.run(model=SurfaceCoverageSpecies.model, problem=SurfaceCoverageSpecies.problem, coords=grid.coords)
        stop = time.time()

        t_eval[str(n_cpu)] = stop - start

    # plot results
    plt.figure(figsize=[4, 4])
    for ind, t in enumerate(t_eval):
        plt.bar(ind, t_eval[t], color=sns.color_palette("pastel", len(t_eval))[ind])

    plt.xlabel("n_cpu", fontsize=11)
    plt.ylabel("Computation time in s", fontsize=11)
    plt.xticks(range(len(t_eval)), t_eval.keys())
    plt.title("Parallel model evaluation", fontsize=12)
    plt.tight_layout()



.. image:: /auto_features/images/sphx_glr_plot_parallelization_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    It/Sub-it: N/A/N/A Performing simulation 001 from 100 [                                        ] 1.0%
    It/Sub-it: N/A/N/A Performing simulation 001 from 100 [                                        ] 1.0%
    It/Sub-it: N/A/N/A Performing simulation 002 from 100 [                                        ] 2.0%
    It/Sub-it: N/A/N/A Performing simulation 003 from 100 [=                                       ] 3.0%
    It/Sub-it: N/A/N/A Performing simulation 004 from 100 [=                                       ] 4.0%
    It/Sub-it: N/A/N/A Performing simulation 005 from 100 [==                                      ] 5.0%
    It/Sub-it: N/A/N/A Performing simulation 006 from 100 [==                                      ] 6.0%
    It/Sub-it: N/A/N/A Performing simulation 007 from 100 [==                                      ] 7.0%
    It/Sub-it: N/A/N/A Performing simulation 008 from 100 [===                                     ] 8.0%
    It/Sub-it: N/A/N/A Performing simulation 009 from 100 [===                                     ] 9.0%
    It/Sub-it: N/A/N/A Performing simulation 010 from 100 [====                                    ] 10.0%
    It/Sub-it: N/A/N/A Performing simulation 011 from 100 [====                                    ] 11.0%
    It/Sub-it: N/A/N/A Performing simulation 012 from 100 [====                                    ] 12.0%
    It/Sub-it: N/A/N/A Performing simulation 013 from 100 [=====                                   ] 13.0%
    It/Sub-it: N/A/N/A Performing simulation 014 from 100 [=====                                   ] 14.0%
    It/Sub-it: N/A/N/A Performing simulation 015 from 100 [======                                  ] 15.0%
    It/Sub-it: N/A/N/A Performing simulation 016 from 100 [======                                  ] 16.0%
    It/Sub-it: N/A/N/A Performing simulation 017 from 100 [======                                  ] 17.0%
    It/Sub-it: N/A/N/A Performing simulation 018 from 100 [=======                                 ] 18.0%
    It/Sub-it: N/A/N/A Performing simulation 019 from 100 [=======                                 ] 19.0%
    It/Sub-it: N/A/N/A Performing simulation 020 from 100 [========                                ] 20.0%
    It/Sub-it: N/A/N/A Performing simulation 021 from 100 [========                                ] 21.0%
    It/Sub-it: N/A/N/A Performing simulation 022 from 100 [========                                ] 22.0%
    It/Sub-it: N/A/N/A Performing simulation 023 from 100 [=========                               ] 23.0%
    It/Sub-it: N/A/N/A Performing simulation 024 from 100 [=========                               ] 24.0%
    It/Sub-it: N/A/N/A Performing simulation 025 from 100 [==========                              ] 25.0%
    It/Sub-it: N/A/N/A Performing simulation 026 from 100 [==========                              ] 26.0%
    It/Sub-it: N/A/N/A Performing simulation 027 from 100 [==========                              ] 27.0%
    It/Sub-it: N/A/N/A Performing simulation 028 from 100 [===========                             ] 28.0%
    It/Sub-it: N/A/N/A Performing simulation 029 from 100 [===========                             ] 29.0%
    It/Sub-it: N/A/N/A Performing simulation 030 from 100 [============                            ] 30.0%
    It/Sub-it: N/A/N/A Performing simulation 031 from 100 [============                            ] 31.0%
    It/Sub-it: N/A/N/A Performing simulation 032 from 100 [============                            ] 32.0%
    It/Sub-it: N/A/N/A Performing simulation 033 from 100 [=============                           ] 33.0%
    It/Sub-it: N/A/N/A Performing simulation 034 from 100 [=============                           ] 34.0%
    It/Sub-it: N/A/N/A Performing simulation 035 from 100 [==============                          ] 35.0%
    It/Sub-it: N/A/N/A Performing simulation 036 from 100 [==============                          ] 36.0%
    It/Sub-it: N/A/N/A Performing simulation 037 from 100 [==============                          ] 37.0%
    It/Sub-it: N/A/N/A Performing simulation 038 from 100 [===============                         ] 38.0%
    It/Sub-it: N/A/N/A Performing simulation 039 from 100 [===============                         ] 39.0%
    It/Sub-it: N/A/N/A Performing simulation 040 from 100 [================                        ] 40.0%
    It/Sub-it: N/A/N/A Performing simulation 041 from 100 [================                        ] 41.0%
    It/Sub-it: N/A/N/A Performing simulation 042 from 100 [================                        ] 42.0%
    It/Sub-it: N/A/N/A Performing simulation 043 from 100 [=================                       ] 43.0%
    It/Sub-it: N/A/N/A Performing simulation 044 from 100 [=================                       ] 44.0%
    It/Sub-it: N/A/N/A Performing simulation 045 from 100 [==================                      ] 45.0%
    It/Sub-it: N/A/N/A Performing simulation 046 from 100 [==================                      ] 46.0%
    It/Sub-it: N/A/N/A Performing simulation 047 from 100 [==================                      ] 47.0%
    It/Sub-it: N/A/N/A Performing simulation 048 from 100 [===================                     ] 48.0%
    It/Sub-it: N/A/N/A Performing simulation 049 from 100 [===================                     ] 49.0%
    It/Sub-it: N/A/N/A Performing simulation 050 from 100 [====================                    ] 50.0%
    It/Sub-it: N/A/N/A Performing simulation 051 from 100 [====================                    ] 51.0%
    It/Sub-it: N/A/N/A Performing simulation 052 from 100 [====================                    ] 52.0%
    It/Sub-it: N/A/N/A Performing simulation 053 from 100 [=====================                   ] 53.0%
    It/Sub-it: N/A/N/A Performing simulation 054 from 100 [=====================                   ] 54.0%
    It/Sub-it: N/A/N/A Performing simulation 055 from 100 [======================                  ] 55.0%
    It/Sub-it: N/A/N/A Performing simulation 056 from 100 [======================                  ] 56.0%
    It/Sub-it: N/A/N/A Performing simulation 057 from 100 [======================                  ] 57.0%
    It/Sub-it: N/A/N/A Performing simulation 058 from 100 [=======================                 ] 58.0%
    It/Sub-it: N/A/N/A Performing simulation 059 from 100 [=======================                 ] 59.0%
    It/Sub-it: N/A/N/A Performing simulation 060 from 100 [========================                ] 60.0%
    It/Sub-it: N/A/N/A Performing simulation 061 from 100 [========================                ] 61.0%
    It/Sub-it: N/A/N/A Performing simulation 062 from 100 [========================                ] 62.0%
    It/Sub-it: N/A/N/A Performing simulation 063 from 100 [=========================               ] 63.0%
    It/Sub-it: N/A/N/A Performing simulation 064 from 100 [=========================               ] 64.0%
    It/Sub-it: N/A/N/A Performing simulation 065 from 100 [==========================              ] 65.0%
    It/Sub-it: N/A/N/A Performing simulation 066 from 100 [==========================              ] 66.0%
    It/Sub-it: N/A/N/A Performing simulation 067 from 100 [==========================              ] 67.0%
    It/Sub-it: N/A/N/A Performing simulation 068 from 100 [===========================             ] 68.0%
    It/Sub-it: N/A/N/A Performing simulation 069 from 100 [===========================             ] 69.0%
    It/Sub-it: N/A/N/A Performing simulation 070 from 100 [============================            ] 70.0%
    It/Sub-it: N/A/N/A Performing simulation 071 from 100 [============================            ] 71.0%
    It/Sub-it: N/A/N/A Performing simulation 072 from 100 [============================            ] 72.0%
    It/Sub-it: N/A/N/A Performing simulation 073 from 100 [=============================           ] 73.0%
    It/Sub-it: N/A/N/A Performing simulation 074 from 100 [=============================           ] 74.0%
    It/Sub-it: N/A/N/A Performing simulation 075 from 100 [==============================          ] 75.0%
    It/Sub-it: N/A/N/A Performing simulation 076 from 100 [==============================          ] 76.0%
    It/Sub-it: N/A/N/A Performing simulation 077 from 100 [==============================          ] 77.0%
    It/Sub-it: N/A/N/A Performing simulation 078 from 100 [===============================         ] 78.0%
    It/Sub-it: N/A/N/A Performing simulation 079 from 100 [===============================         ] 79.0%
    It/Sub-it: N/A/N/A Performing simulation 080 from 100 [================================        ] 80.0%
    It/Sub-it: N/A/N/A Performing simulation 081 from 100 [================================        ] 81.0%
    It/Sub-it: N/A/N/A Performing simulation 082 from 100 [================================        ] 82.0%
    It/Sub-it: N/A/N/A Performing simulation 083 from 100 [=================================       ] 83.0%
    It/Sub-it: N/A/N/A Performing simulation 084 from 100 [=================================       ] 84.0%
    It/Sub-it: N/A/N/A Performing simulation 085 from 100 [==================================      ] 85.0%
    It/Sub-it: N/A/N/A Performing simulation 086 from 100 [==================================      ] 86.0%
    It/Sub-it: N/A/N/A Performing simulation 087 from 100 [==================================      ] 87.0%
    It/Sub-it: N/A/N/A Performing simulation 088 from 100 [===================================     ] 88.0%
    It/Sub-it: N/A/N/A Performing simulation 089 from 100 [===================================     ] 89.0%
    It/Sub-it: N/A/N/A Performing simulation 090 from 100 [====================================    ] 90.0%
    It/Sub-it: N/A/N/A Performing simulation 091 from 100 [====================================    ] 91.0%
    It/Sub-it: N/A/N/A Performing simulation 092 from 100 [====================================    ] 92.0%
    It/Sub-it: N/A/N/A Performing simulation 093 from 100 [=====================================   ] 93.0%
    It/Sub-it: N/A/N/A Performing simulation 094 from 100 [=====================================   ] 94.0%
    It/Sub-it: N/A/N/A Performing simulation 095 from 100 [======================================  ] 95.0%
    It/Sub-it: N/A/N/A Performing simulation 096 from 100 [======================================  ] 96.0%
    It/Sub-it: N/A/N/A Performing simulation 097 from 100 [======================================  ] 97.0%
    It/Sub-it: N/A/N/A Performing simulation 098 from 100 [======================================= ] 98.0%
    It/Sub-it: N/A/N/A Performing simulation 099 from 100 [======================================= ] 99.0%
    It/Sub-it: N/A/N/A Performing simulation 100 from 100 [========================================] 100.0%





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.523 seconds)


.. _sphx_glr_download_auto_features_plot_parallelization.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_parallelization.py <plot_parallelization.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_parallelization.ipynb <plot_parallelization.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
