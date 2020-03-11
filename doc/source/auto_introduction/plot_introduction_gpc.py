"""
Introduction to generalized Polynomial Chaos (gPC)
==================================================

The primary focus of this tutorial rests on spectral methods, which are based on the determination of a 
functional dependence between the probabilistic in- and output of a system by means of a series of suitable 
selected functionals. The practical realization of spectral methods can be further subdivided into *intrusive* 
and **non-intrusive** approaches. Intrusive approaches are based on Galerkin methods, where the governing 
equations have to be modified to incorporate the probabilistic character of the model parameters. 
This includes the determination of the stochastic weak form of the problem according to the given 
uncertainties (Le Maitre, 2010). On the contrary,  non-intrusive approaches are based on a reduced
sampling of the probability space without any modification of the deterministic solvers. Those methods 
are more flexible and thus more suitable for universal application. Typical applications can be found 
in the fields of computational fluid dynamics (Knio and Le Maitre, 2006; Xiu, 2003; Hosder et al., 2006), heat transfer
[Wan et al., 2004; Xiu and Karniadakis, 2003), multibody dynamics (Sandu et al., 2006a, Sandu et al. 2006b),
robust design optimization (Zein, 2013) or in biomedical engineering (Saturnino et al., 2019; Weise et al. 2015;
Codecasa et al., 2016).
During the last years, spectral approaches are becoming increasingly popular. However, those are not a reference
tool yet and still unknown for many people. For that reason, particular emphasis is placed to 
describe the method and to further elucidate the principle by means of examples.

The gPC expansion
^^^^^^^^^^^^^^^^^
The basic concept of the gPC is to find a functional dependence between the random variables :math:`{\\xi}` 
and the solutions :math:`y(\\mathbf{r},{\\xi})` by means of an orthogonal polynomial basis :math:`\\Psi({\\xi})`. 
In its general form, it is given by: 

.. math::
    y(\\mathbf{r},{\\xi}) = \\sum_{\\mathbf{\\alpha}\\in\\mathcal{A}(\\mathbf{p})}
    u_{\\mathbf{\\alpha}}(\\mathbf{r}) \\Psi_{\\mathbf{\\alpha}}({\\xi}).


The terms are indexed by the multi-index  :math:`\\mathbf{\\alpha}=(\\alpha_0,...,\\alpha_{d-1})`, which is a 
`d`-tuple of non-negative integers :math:`\\mathbf{\\alpha}\\in\\mathbb{N}_0^d`. The sum is carried out over 
the multi-indices, contained in the set :math:`\\mathcal{A}(\\mathbf{p})`. The composition of the set depends 
on the type of expansion and is parameterized by a parameter vector :math:`\\mathbf{p}`, which will be 
explained in a later part of this section.

The function :math:`\\Psi_{\\mathbf{\\alpha}}({\\xi})` are the joint polynomial basis functions of the gPC. 
They are composed of polynomials :math:`\\psi_{\\alpha_i}(\\xi_i)`.

.. math::
    \\Psi_{\\mathbf{\\alpha}}({\\xi}) = \\prod_{i=1}^{d} \\psi_{\\alpha_i}(\\xi_i)


The polynomials :math:`\\psi_{\\alpha_i}(\\xi_i)` are defined for each random variable separately according 
to the corresponding pdf :math:`p_i(\\xi_i)`. They have to be chosen to ensure orthogonality. The set of 
polynomials for an optimal basis of continuous probability distributions is derived from the Askey 
scheme (Askey and Wilson, 1985). The index of the polynomials denotes its order (or degree). In this way, the
multi-index :math:`\\mathbf{\\alpha}` corresponds to the order of the individual basis functions forming 
the joint basis function.

+-----------+--------------+------------------------+-----------------------------+
| Type      | Distribution | Orthogonal polynomials | Range                       |
+===========+==============+========================+=============================+
|continuous | uniform      | Legendre               | :math:`(a,b)`               |
+-----------+--------------+------------------------+-----------------------------+
|continuous | beta         | Jacobi                 | :math:`(a,b)`               |
+-----------+--------------+------------------------+-----------------------------+
|continuous | gaussian     | Hermite                | :math:`(-\\infty,+\\infty)`   |
+-----------+--------------+------------------------+-----------------------------+
|continuous | gamma        | Laguerre               | :math:`(0,+\\infty)`         |
+-----------+--------------+------------------------+-----------------------------+
| discrete  | poisson      | Charlier               | :math:`(0,1,...)`           |
+-----------+--------------+------------------------+-----------------------------+

References
^^^^^^^^^^
.. [1] Le Maitre, O., and Knio, O. M. (2010). Spectral methods for uncertainty quantification: with applications
   to computational fluid dynamics. Springer Science & Business Media.

.. [2] Knio, O. M., & Le Maitre, O. P. (2006). Uncertainty propagation in CFD using polynomial chaos decomposition.
   Fluid dynamics research, 38(9), 616.

.. [3] Xiu, D., & Karniadakis, G. E. (2003). Modeling uncertainty in flow simulations via generalized polynomial chaos.
   Journal of computational physics, 187(1), 137-167.

.. [4] Hosder, S., Walters, R., & Perez, R. (2006). A non-intrusive polynomial chaos method for uncertainty
   propagation in CFD simulations. In 44th AIAA aerospace sciences meeting and exhibit (p. 891).

.. [5] Wan, X., Xiu, D., & Karniadakis, G. E. (2004). Modeling uncertainty in three-dimensional heat transfer problems.
   WIT Transactions on Engineering Sciences, 46.

.. [6] Xiu, D., & Karniadakis, G. E. (2003). A new stochastic approach to transient heat conduction modeling
   with uncertainty. International Journal of Heat and Mass Transfer, 46(24), 4681-4693.

.. [7] Sandu, A., Sandu, C., & Ahmadian, M. (2006). Modeling multibody systems with uncertainties.
   Part I: Theoretical and computational aspects. Multibody System Dynamics, 15(4), 369-391.

.. [8] Sandu, C., Sandu, A., & Ahmadian, M. (2006). Modeling multibody systems with uncertainties.
   Part II: Numerical applications. Multibody System Dynamics, 15(3), 241-262.

.. [9] Zein, S. (2013). A polynomial chaos expansion trust region method for robust optimization.
   Communications in Computational Physics, 14(2), 412-424.

.. [10] Saturnino, G. B., Thielscher, A., Madsen, K. H., Knösche, T. R., & Weise, K. (2019). A principled approach to
   conductivity uncertainty analysis in electric field calculations. Neuroimage, 188, 821-834.

.. [11] Weise, K., Di Rienzo, L., Brauer, H., Haueisen, J., & Toepfer, H. (2015). Uncertainty analysis in
   transcranial magnetic stimulation using nonintrusive polynomial chaos expansion.
   IEEE Transactions on Magnetics, 51(7), 1-8.

.. [12] Codecasa, L., Di Rienzo, L., Weise, K., Gross, S., & Haueisen, J. (2015). Fast MOR-based approach to
   uncertainty quantification in transcranial magnetic stimulation. IEEE Transactions on Magnetics, 52(3), 1-4.

.. [13] Weise, K., Numssen, O., Thielscher, A., Hartwigsen, G., & Knösche, T. R. (2020).
   A novel approach to localize cortical TMS effects. NeuroImage, 209, 116486.

.. [14] Askey, R., & Wilson, J. A. (1985). Some basic hypergeometric orthogonal polynomials
   that generalize Jacobi polynomials (Vol. 319). American Mathematical Soc..
"""