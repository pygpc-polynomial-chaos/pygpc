import numpy as np
import scipy.special
import scipy.stats
import matplotlib.pyplot as plt

order = np.arange(10)
alpha_pdf = np.linspace(1, 5, 10)
alpha_poly = alpha_pdf - 1
scale_int = np.zeros((len(order), len(alpha_pdf)))
scale_ana = np.zeros((len(order), len(alpha_pdf)))
i_n = 0
i_a = 0

for n in order:

    i_a = 0

    for a_pdf, a_poly in zip(alpha_pdf, alpha_poly):
        xmax = scipy.stats.gamma.ppf(0.9999999999999999, a=a_pdf, loc=0., scale=1.)
        x = np.linspace(0, xmax, int(1e5))
        poly = scipy.special.genlaguerre(n=n,alpha=a_poly,monic=False)
        y_poly = poly(x)
        y_pdf = scipy.stats.gamma.pdf(x, a=a_pdf, loc=0., scale=1.)
        y = y_poly**2 * y_pdf

        scale_int[i_n, i_a] = np.trapz(x=x,y=y)
        scale_ana[i_n, i_a] = scipy.special.factorial(n + a_poly) / scipy.special.factorial(n) / scipy.special.gamma(a_poly + 1)
        i_a += 1

    i_n += 1

X, Y = np.meshgrid(alpha_pdf, order)
fig = plt.figure(figsize=(16, 4.5))

ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.plot_surface(X, Y, scale_int, cmap="jet")
ax.set_xlabel("alpha_pdf")
ax.set_ylabel("order")
ax.set_zlim([0, np.max(scale_int)])
ax.set_title("integrated")

ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.plot_surface(X, Y, scale_ana, cmap="jet")
ax.set_xlabel("alpha_pdf")
ax.set_ylabel("order")
ax.set_zlim([0, np.max(scale_int)])
ax.set_title("analytic")

ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.plot_surface(X, Y, np.abs(scale_ana-scale_int), cmap="jet")
ax.set_xlabel("alpha_pdf")
ax.set_ylabel("order")
# ax.set_zlim([0, np.max(scale_int)])
ax.set_title("analytic vs integrated")
