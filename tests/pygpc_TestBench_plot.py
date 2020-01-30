import os
import pickle
import pygpc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib

folder = "/data/pt_01756/studies/pygpc/TestBenchContinuous/"
algorithms = os.listdir(folder)
algorithms.sort()
testbench_objs = []

order = [2, 3, 4, 6, 8, 10, 12]

for a in algorithms:
    with open(os.path.join(folder, a, "testbench.pkl"), 'rb') as f:
        testbench_objs.append(pickle.load(f))

# te = pygpc.read_session_pkl("/NOBACKUP2/tmp/TestBench/TestBenchContinuous/RegAdaptive_MP_q_1.0/Peaks_0001.pkl")

n_basis = dict()
n_grid = dict()
nrmsd = dict()
loocv = dict()
std = dict()

for testbench in testbench_objs:

    # concatenate results for static gpc
    if testbench.algorithm_type == pygpc.Static or testbench.algorithm_type == pygpc.StaticProjection:
        n_grid[os.path.split(testbench.fn_results)[1]] = dict()
        n_basis[os.path.split(testbench.fn_results)[1]] = dict()
        nrmsd[os.path.split(testbench.fn_results)[1]] = dict()
        loocv[os.path.split(testbench.fn_results)[1]] = dict()
        std[os.path.split(testbench.fn_results)[1]] = dict()

        # ["algorithm"]["testfunction"][repetitions x order]
        for pkey in testbench.problem_keys:
            n_grid[os.path.split(testbench.fn_results)[1]][pkey] = [[] for _ in range(testbench.repetitions)]
            n_basis[os.path.split(testbench.fn_results)[1]][pkey] = [[] for _ in range(testbench.repetitions)]
            nrmsd[os.path.split(testbench.fn_results)[1]][pkey] = [[] for _ in range(testbench.repetitions)]
            loocv[os.path.split(testbench.fn_results)[1]][pkey] = [[] for _ in range(testbench.repetitions)]
            std[os.path.split(testbench.fn_results)[1]][pkey] = [[] for _, o in enumerate(order)]

            for i_o, o in enumerate(order):

                for rep in range(testbench.repetitions):
                    fn = os.path.join(testbench.fn_results, pkey + "_p_" + str(o) + "_" + str(rep).zfill(4) + ".pkl")
                    session = pygpc.read_session_pkl(fn)
                    n_grid[os.path.split(testbench.fn_results)[1]][pkey][rep].append(session.gpc[0].n_grid[0])
                    n_basis[os.path.split(testbench.fn_results)[1]][pkey][rep].append(session.gpc[0].n_basis[0])
                    nrmsd[os.path.split(testbench.fn_results)[1]][pkey][rep].append(session.gpc[0].relative_error_nrmsd[0])
                    if session.gpc[0].relative_error_loocv:
                        loocv[os.path.split(testbench.fn_results)[1]][pkey][rep].append(session.gpc[0].relative_error_loocv[0])
                std_over_reps = np.std([nrmsd[os.path.split(testbench.fn_results)[1]][pkey][i][i_o] for i in range(testbench.repetitions)])
                std[os.path.split(testbench.fn_results)[1]][pkey][i_o] = std_over_reps

            n_grid[os.path.split(testbench.fn_results)[1]][pkey][rep] = np.array(n_grid[os.path.split(testbench.fn_results)[1]][pkey][rep])
            n_basis[os.path.split(testbench.fn_results)[1]][pkey][rep] = np.array(n_basis[os.path.split(testbench.fn_results)[1]][pkey][rep])
            nrmsd[os.path.split(testbench.fn_results)[1]][pkey][rep] = np.array(nrmsd[os.path.split(testbench.fn_results)[1]][pkey][rep])
            if session.gpc[0].relative_error_loocv:
                loocv[os.path.split(testbench.fn_results)[1]][pkey][rep] = np.array(loocv[os.path.split(testbench.fn_results)[1]][pkey][rep])
    else:
        n_grid[os.path.split(testbench.fn_results)[1]] = dict()
        n_basis[os.path.split(testbench.fn_results)[1]] = dict()
        nrmsd[os.path.split(testbench.fn_results)[1]] = dict()
        loocv[os.path.split(testbench.fn_results)[1]] = dict()

        # ["algorithm"]["testfunction"][repetitions x order]
        for pkey in testbench.problem_keys:

            n_grid[os.path.split(testbench.fn_results)[1]][pkey] = []
            n_basis[os.path.split(testbench.fn_results)[1]][pkey] = []
            nrmsd[os.path.split(testbench.fn_results)[1]][pkey] = []
            loocv[os.path.split(testbench.fn_results)[1]][pkey] = []

            for rep in range(testbench.repetitions):
                fn = os.path.join(testbench.fn_results, pkey + "_" + str(rep).zfill(4) + ".pkl")
                session = pygpc.read_session_pkl(fn)

                n_grid[os.path.split(testbench.fn_results)[1]][pkey].append(np.array(session.gpc[0].n_grid))
                n_basis[os.path.split(testbench.fn_results)[1]][pkey].append(np.array(session.gpc[0].n_basis))
                nrmsd[os.path.split(testbench.fn_results)[1]][pkey].append(np.array(session.gpc[0].relative_error_nrmsd))

                # if "Projection" in os.path.split(testbench.fn_results)[1]:
                #     n_grid[os.path.split(testbench.fn_results)[1]][pkey][rep] = n_grid[os.path.split(testbench.fn_results)[1]][pkey][rep][1:]

                if session.gpc[0].relative_error_loocv:
                    loocv[os.path.split(testbench.fn_results)[1]][pkey].append(np.array(session.gpc[0].relative_error_nrmsd))

dims = testbench.dims

# plot results
plot_lines = True

if dims:
    dimension_analysis = True
    figsize = [13. / 5 * len(testbench.problem_keys)/len(dims), 3.*len(dims) + len(testbench_objs)*0.25]
    n_fig_y = len(dims)
    n_fig_x = len(testbench.problem_keys)/len(dims)
else:
    figsize = [13. / 5 * len(testbench.problem_keys), 3. + len(testbench_objs)*0.25]
    n_fig_y = 1
    n_fig_x = len(testbench.problem_keys)

fontsize=12
markersequence = ["o", "X", "^", "D", "p", "s"]
markersequence = markersequence * int(np.ceil(1.0*len(testbench_objs)/len(markersequence)))
f, ax = plt.subplots(n_fig_y, n_fig_x, figsize=figsize, sharey=True)

if ax.ndim == 1:
    if n_fig_y == 1:
        ax = ax[np.newaxis, :]
    elif n_fig_x == 1:
        ax = ax[:, np.newaxis]

cmap = matplotlib.cm.get_cmap('jet')
legend = [os.path.split(t.fn_results)[1] for t in testbench_objs]

# loop over problems (testfunctions and dimension)
i_fun = 0
for i_pkey, pkey in enumerate(testbench.problem_keys):

    if dims:
        i_dim = dims.index(testbench.problem[pkey].dim)
    else:
        i_dim = 0

    if len(dims) > 0:
        if i_fun >= len(testbench.problem_keys)/len(dims):
            i_fun = 0

    n_grid_max = []
    # loop over algorithms
    for i_alg, testbench in enumerate(testbench_objs):
        # loop over repetitions

        n_grid_line = np.array([])
        nrmsd_line = np.array([])
        std_line = np.array([])

        for rep in range(testbench.repetitions):

            if not plot_lines:
                ax[i_dim, i_fun].scatter(n_grid[os.path.split(testbench.fn_results)[1]][pkey][rep],
                                         nrmsd[os.path.split(testbench.fn_results)[1]][pkey][rep],
                                         color=cmap(float(i_alg)/len(testbench_objs)),
                                         s=15,
                                         edgecolors='k',
                                         linewidths=0.5,
                                         marker=markersequence[i_alg])
            else:
                n_grid_line = np.append(n_grid_line, n_grid[os.path.split(testbench.fn_results)[1]][pkey][rep])
                nrmsd_line = np.append(nrmsd_line, nrmsd[os.path.split(testbench.fn_results)[1]][pkey][rep])
                std_line = np.append(std_line, std[os.path.split(testbench.fn_results)[1]][pkey])

            n_grid_max.append(np.max(n_grid[os.path.split(testbench.fn_results)[1]][pkey]))

        if plot_lines:
            sort_idx = np.argsort(n_grid_line)
            n_grid_line = n_grid_line[sort_idx]
            nrmsd_line = nrmsd_line[sort_idx]
            std_line = std_line[sort_idx]

            for i in n_grid_line:
                nrmsd_line[n_grid_line == i] = np.mean(nrmsd_line[n_grid_line == i])
                std_line[n_grid_line == i] = std_line[n_grid_line == i]

            ax[i_dim, i_fun].errorbar(n_grid_line,
                                      nrmsd_line,
                                      yerr=std_line,
                                      # ecolor='g',
                                      elinewidth=0.1,
                                      color=cmap(float(i_alg) / len(testbench_objs)))

            # ,
            # marker = markersequence[i_alg],
            # markersize = 5,
            # markeredgecolor = 'k'

    n_grid_max = np.max(n_grid_max)
    ax[i_dim, i_fun].set_title(pkey, fontsize=10)
    if i_dim == 0 and i_fun == 0:
        ax[i_dim, i_fun].set_ylabel("$\\epsilon_{NRMSD}$", fontsize=fontsize)
        ax[i_dim, i_fun].set_xlabel("$N_g$", fontsize=fontsize)
    ax[i_dim, i_fun].set_yscale("log")
    # ax[i_dim, i_fun].set_xscale("log")
    ax[i_dim, i_fun].set_xlim([0, n_grid_max*1.1])
    # ax[i_dim, i_fun].set_ylim([0.00001, 5])
    ax[i_dim, i_fun].grid(True)
    # ax[i_dim, i_fun].errorbar([0, n_grid_max * 1.1], [1e-2, 1e-2], 'r', xerr=0, yerr=0, linewidth=0.5)

    i_fun += 1

# legend
handles = []
for i_alg in range(len(testbench_objs)):
    if plot_lines:
        handles.append(mlines.Line2D([], [],
                                     color=cmap(float(i_alg) / len(testbench_objs)),
                                     label=legend[i_alg]))
    else:
        handles.append(mlines.Line2D([], [],
                                     color=cmap(float(i_alg)/len(testbench_objs)),
                                     marker=markersequence[i_alg],
                                     markersize=5,
                                     linestyle='None',
                                     label=legend[i_alg],
                                     markeredgecolor="k",
                                     markeredgewidth=0.5))

ax[-1, 0].legend(handles=handles, fontsize=8, loc=9, bbox_to_anchor=(0.5, -0.32))

leg = ax[-1, 0].get_legend()

for i_alg in range(len(testbench_objs)):
    leg.legendHandles[i_alg].set_markeredgecolor("k")
    leg.legendHandles[i_alg].set_markeredgewidth(1)

plt.tight_layout()
plt.show()


# ax[i_fun].tight_layout()
