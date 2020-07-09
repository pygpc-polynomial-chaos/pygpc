.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_gpc_plot_problem.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_gpc_plot_problem.py:


How to define a gPC problem
===========================
Random parameters
-----------------
The :math:`d` parameters of interest, which are assumed to underlie a distinct level of uncertainty, 
are modeled as a :math:`d`-variate random vector denoted by :math:`\mathbf{\xi} = (\xi_1, \xi_2, ... \xi_d)`.
It is defined in the probability space :math:`(\Theta, \Sigma, P)`. The event or random space :math:`\Theta`
contains all possible events. :math:`\Sigma` is a :math:`\sigma`-Algebra over :math:`\Theta`,
containing sets of events, and :math:`P` is a function assigning the probabilities of occurrence to the events.
The number of random variables :math:`d` determines the *dimension* of the uncertainty problem.
It is assumed that the parameters are statistically mutually independent from each other.
In order to perform a gPC expansion, the random variables must have a finite variance, which defines
the problem in the :math:`L_2`-Hilbert space.

The probability density function (pdf) :math:`p_i(\xi_i)`, with :math:`i=1,...,d`, has to be defined
for each random variable :math:`\xi_i`.

Currently, **pygpc** supports:

Beta distributed random variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Probability density function:

.. math::

    p(x) = \left(\frac{\Gamma(p)\Gamma(q)}{\Gamma(p+q)}(b-a)^{(p+q-1)}\right)^{-1} (x-a)^{(p-1)} (b-x)^{(q-1)}

The shape parameters of beta distributed random variable are defined with the parameter pdf_shape :math:`=[p, q]`
and the limits with pdf_limits :math:`=[a, b]`.


.. code-block:: default


    import pygpc
    from collections import OrderedDict

    parameters = OrderedDict()
    parameters["x1"] = pygpc.Beta(pdf_shape=[5, 5], pdf_limits=[0, 1])
    parameters["x2"] = pygpc.Beta(pdf_shape=[5, 2], pdf_limits=[0, 1])
    parameters["x3"] = pygpc.Beta(pdf_shape=[2, 10], pdf_limits=[0, 1])
    parameters["x4"] = pygpc.Beta(pdf_shape=[0.75, 0.75], pdf_limits=[0, 1])
    parameters["x5"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
    ax = parameters["x1"].plot_pdf()
    ax = parameters["x2"].plot_pdf()
    ax = parameters["x3"].plot_pdf()
    ax = parameters["x4"].plot_pdf()
    ax = parameters["x5"].plot_pdf()
    _ = ax.legend(["x1", "x2", "x3", "x4", "x5"])




.. image:: /auto_gpc/images/sphx_glr_plot_problem_001.png
    :class: sphx-glr-single-img





Normal distributed random variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Probability density function:

.. math::

    p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp{-\frac{(x-\mu)^2}{2\sigma^2}}

The mean and the standard deviation are defined with the parameter pdf_shape :math:`=[\mu, \sigma]`.


.. code-block:: default


    parameters = OrderedDict()
    parameters["x1"] = pygpc.Norm(pdf_shape=[5, 1])
    parameters["x2"] = pygpc.Norm(pdf_shape=[3, 2])
    parameters["x3"] = pygpc.Norm(pdf_shape=[1, 3])
    ax = parameters["x1"].plot_pdf()
    ax = parameters["x2"].plot_pdf()
    ax = parameters["x3"].plot_pdf()
    _ = ax.legend(["x1", "x2", "x3"])




.. image:: /auto_gpc/images/sphx_glr_plot_problem_002.png
    :class: sphx-glr-single-img





Gamma distributed random variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Probability density function:

.. math::

    p(x) = \frac{\beta^{\alpha}}{\Gamma(\alpha)}x^{\alpha-1}e^{\beta x}

The shape, rate and the location of the gamma distributed random variable is defined with
the parameter pdf_shape :math:`=[\alpha, \beta, loc]`


.. code-block:: default


    parameters = OrderedDict()
    parameters["x1"] = pygpc.Gamma(pdf_shape=[1, 1, 0])
    parameters["x2"] = pygpc.Gamma(pdf_shape=[5, 5, 0])
    parameters["x3"] = pygpc.Gamma(pdf_shape=[5, 2, 1.5])
    parameters["x4"] = pygpc.Gamma(pdf_shape=[2, 1, 1])

    ax = parameters["x1"].plot_pdf()
    ax = parameters["x2"].plot_pdf()
    ax = parameters["x3"].plot_pdf()
    ax = parameters["x4"].plot_pdf()
    _ = ax.legend(["x1", "x2", "x3", "x4"])




.. image:: /auto_gpc/images/sphx_glr_plot_problem_003.png
    :class: sphx-glr-single-img





Problem definition
^^^^^^^^^^^^^^^^^^
The gPC problem is initialized with the model and the parameters defined before:


.. code-block:: default


    # define model
    model = pygpc.testfunctions.Peaks()

    # define problem
    problem = pygpc.Problem(model, parameters)








.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.300 seconds)


.. _sphx_glr_download_auto_gpc_plot_problem.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_problem.py <plot_problem.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_problem.ipynb <plot_problem.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
