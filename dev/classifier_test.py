import pygpc
import numpy as np
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import svm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import scipy as sp
import h5py
from sklearn.neural_network import MLPClassifier
import hdbscan
import random

# def dbscan_predict(model, X):
#
#     nr_samples = X.shape[0]
#
#     y_new = np.ones(shape=nr_samples, dtype=int) * -1
#
#     for i in range(nr_samples):
#         diff = model.components_ - X[i, :]  # NumPy broadcasting
#
#         dist = np.linalg.norm(diff, axis=1)  # Euclidean distance
#
#         shortest_dist_idx = np.argmin(dist)
#
#         if dist[shortest_dist_idx] < model.eps:
#             y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]
#
#     return y_new


n_grid_learn_cluster = 50
n_grid_test = 50
model = pygpc.testfunctions.BinaryDiscontinuousSphere

parameters = OrderedDict()
parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
parameters["x2"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 1])
problem = pygpc.Problem(model, parameters)

# grids
grid_learn_cluster = pygpc.RandomGrid(parameters_random=problem.parameters_random,
                                      options={"n_grid": n_grid_learn_cluster, "seed": 1})
grid_test = pygpc.RandomGrid(parameters_random=problem.parameters_random,
                             options={"n_grid": n_grid_test, "seed": 1})

# load test data
with h5py.File("/data/pt_01756/software/git/pygpc/dev/test_case_1.h5", "r") as f:
    grid_learn_cluster.coords = f["x"][:]
    y_learn_cluster = f["y"][:]

# p = OrderedDict()
# for i, key in enumerate(parameters.keys()):
#     p[key] = grid_learn_cluster.coords[:, i]
#
# model_sim = model(p)
# y_learn_cluster = model_sim.simulate()

# unsupervised learning (kmeans)
model_kmeans = KMeans(n_clusters=2, random_state=0)
model_kmeans.fit(y_learn_cluster)

# unsupervised learning (hdbscan)
# model_hdbscan = hdbscan.HDBSCAN(min_cluster_size=2, prediction_data=True).fit(y_learn_cluster)

# predict classes
classes_learn = model_kmeans.labels_
# classes_learn, strengths = hdbscan.approximate_predict(model_hdbscan, y_learn_cluster)

# classification (SVM)
# clf = svm.SVC(gamma='auto', degree=10, kernel="rbf")
# clf = QuadraticDiscriminantAnalysis(reg_param=0, tol=1e-6)
# clf = GaussianNB()
clf = MLPClassifier(alpha=0.01, max_iter=1000, activation="relu", solver="lbfgs")
clf.fit(grid_learn_cluster.coords, classes_learn)

coords_border_downsampled = pygpc.get_coords_discontinuity(classifier=clf,
                                                           x_min=np.min(grid_learn_cluster.coords, axis=0),
                                                           x_max=np.max(grid_learn_cluster.coords, axis=0),
                                                           n_coords_disc=10,
                                                           border_sampling="structured")

# determine class border
# border_sampling = "random"
# border_sampling = "structured"
#
# if border_sampling == "random":
#     coords_border_det = grid_learn_cluster.coords
#     domains = model_kmeans.labels_
#
# elif border_sampling == "structured":
#     coords_border_det = np.array(np.meshgrid(np.linspace(np.min(grid_learn_cluster.coords[:, 0]),
#                                                          np.max(grid_learn_cluster.coords[:, 0]), 1000),
#                                              np.linspace(np.min(grid_learn_cluster.coords[:, 1]),
#                                                          np.max(grid_learn_cluster.coords[:, 1]), 1000)
#                                              )).T.reshape(-1, 2)
#     domains = np.array(clf.predict(coords_border_det), dtype=np.uint8)
#
# import time
#
# start = time.time()
# dom_mat1 = np.outer(domains, np.ones(len(domains), dtype=np.uint8))
# end = time.time()
# print("outer: {}".format(end-start))
#
# start = time.time()
# dom_mat2 = np.broadcast_to(domains, (len(domains), len(domains))).T
# end = time.time()
# print("broadcast: {}".format(end-start))
#
# dom_mat = np.tile(domains[:, np.newaxis], (1, domains.shape[0]))
# mask = dom_mat != dom_mat.transpose()
# mask = np.tril(mask)*False + np.triu(mask)
#
# distance_matrix = np.ones(mask.shape)*np.nan
# for i_c, c in enumerate(coords_border_det):
#     distance_matrix[i_c, mask[i_c, :]] = np.linalg.norm(coords_border_det[mask[i_c, :], :] - c, axis=1)
#
# np.fill_diagonal(distance_matrix, np.nan)
#
# # find N_smallest distances and determine midpoints
# N_smallest = 2000
# idx = pygpc.get_indices_of_k_smallest(distance_matrix, N_smallest)
# coords_border = (coords_border_det[idx[0], :] + coords_border_det[idx[1], :])/2
#
# N_reps = 10000
# N_resample = 10
# idx = np.zeros((N_resample, N_reps))
# distance_mean = np.zeros(N_reps)
#
# for i in range(N_reps):
#     # sample N_resample points from coords_border
#     idx[:, i] = random.sample(list(range(coords_border.shape[0])), N_resample)
#
#     # determine all to all distances
#     distance_matrix_resample = np.ones((N_resample, N_resample)) * np.nan
#
#     for i_c, c in enumerate(coords_border[idx[:, i].astype(int), :]):
#         distance_matrix_resample[i_c, :] = np.linalg.norm(coords_border[idx[:, i].astype(int), :] - c, axis=1)
#
#     np.fill_diagonal(distance_matrix_resample, 1000)
#     distance_mean[i] = np.mean(np.min(distance_matrix_resample, axis=1))
#     # distance_mean[i] = np.mean(distance_matrix_resample)
#
# coords_border_downsampled = coords_border[idx[:, np.argmax(distance_mean)].astype(int), :]

# # determine distances to reference point (first point)
# distance_ref = np.linalg.norm(coords_border - coords_border[500, :], axis=1)
#
# # fit beta pdf to distances
# distance_pdf_paras, _, _, _ = pygpc.get_beta_pdf_fit(data=distance_ref, beta_tolerance=0., uni_interval=0, fn_plot="/NOBACKUP2/tmp/distance_beta")
#
# parameters = OrderedDict()
# parameters["distance"] = pygpc.Beta(pdf_shape=[1, 1], #[distance_pdf_paras[0], distance_pdf_paras[1]]
#                                     pdf_limits=[0, distance_pdf_paras[3]])
#
# N_reps = 1000
# N_resample = 10
# distance_sample = np.zeros((N_resample, N_reps))
#
# for i in range(N_reps):
#     distance_sample[:, i] = pygpc.RandomGrid(parameters_random=parameters,
#                                              options={"n_grid": N_resample, "seed": None}).coords.flatten()
#
# idx = np.argmax(np.mean(distance_sample, axis=0))
#
# te = np.abs(distance_ref[:, np.newaxis] - distance_sample[:, idx][np.newaxis, :])
#
# coords_border_idx = np.argmin(te, axis=0)
#
# coords_border_downsampled = coords_border[coords_border_idx, :]

prediction_border = clf.predict(coords_border_downsampled)

# prediction
prediction = clf.predict(grid_learn_cluster.coords)
# prediction = clf.predict(grid_test.coords)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(grid_learn_cluster.coords[:, 0], grid_learn_cluster.coords[:, 1], y_learn_cluster)
ax.scatter(grid_learn_cluster.coords[:, 0], grid_learn_cluster.coords[:, 1], classes_learn-1, c='r')
ax.scatter(grid_learn_cluster.coords[:, 0], grid_learn_cluster.coords[:, 1], prediction+3, c='k')
ax.scatter(coords_border_downsampled[:, 0], coords_border_downsampled[:, 1], prediction_border+5, c='y')

# ax.scatter(grid_test.coords[:, 0], grid_test.coords[:, 1], prediction+3, c='k')