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


n_grid_learn_cluster = 1000
n_grid_test = 1000
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
classes_learn = model_kmeans.predict(y_learn_cluster).astype(float)
# classes_learn, strengths = hdbscan.approximate_predict(model_hdbscan, y_learn_cluster)

# classification (SVM)
# clf = svm.SVC(gamma='auto', degree=10, kernel="rbf")
# clf = QuadraticDiscriminantAnalysis(reg_param=0, tol=1e-6)
# clf = GaussianNB()
clf = MLPClassifier(alpha=0.01, max_iter=1000, activation="relu", solver="lbfgs")
clf.fit(grid_learn_cluster.coords, classes_learn)

# prediction
prediction = clf.predict(grid_learn_cluster.coords)
# prediction = clf.predict(grid_test.coords)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(grid_learn_cluster.coords[:, 0], grid_learn_cluster.coords[:, 1], y_learn_cluster)
ax.scatter(grid_learn_cluster.coords[:, 0], grid_learn_cluster.coords[:, 1], classes_learn-1, c='r')
ax.scatter(grid_learn_cluster.coords[:, 0], grid_learn_cluster.coords[:, 1], prediction+3, c='k')
# ax.scatter(grid_test.coords[:, 0], grid_test.coords[:, 1], prediction+3, c='k')