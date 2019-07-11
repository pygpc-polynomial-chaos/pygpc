# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
from sklearn.neural_network import MLPClassifier
import numpy as np
import copy


def Classifier(coords, results, algorithm="learning", options=None):
    """
    Helper function to initialize Classifier class.

    Parameters
    ----------
    coords: ndarray of float [n_grid, n_dim]
        Set of n_grid parameter combinations
    results: ndarray [n_grid x n_out]
        Results of the model evaluation
    algorithm: str, optional, default: "learning"
        Algorithm to classify grid points
        - "learning" ... 2-step procedure with unsupervised and supervised learning
        - ...
    options: dict, optional, default=None
        Classifier options

    Returns
    -------
    obj : object instance of Classifier class
        Object instance of Classifier class
    """
    if algorithm == "learning":
        return ClassifierLearning(coords=coords, results=results, options=options)
    else:
        raise AttributeError("Please specify correct classification algorithm: {""learning"", ...}")


class ClassifierLearning(object):
    """
    ClassifierLearning class

    Attributes
    ----------
    coords: ndarray of float [n_grid, n_dim]
        Grid points to train the classifier
    results: ndarray [n_grid x n_out]
        Results of the model evaluation
    options: dict, optional, default=None
        Classifier options
    clf: Classifier object
        Classifier object

    """
    def __init__(self, coords, results, options=None):
        """
        Constructor; Initializes ClassifierLearning class

        Parameters
        ----------
        coords: ndarray of float [n_grid, n_dim]
            Grid points to train the classifier
        results: ndarray [n_grid x n_out]
            Results of the model evaluation
        options: dict, optional, default=None
            Classifier options
            - options["clusterer"] ... Cluster algorithm (e.g. "KMeans")
            - options["n_clusters"] ... Number of clusters in case of "KMeans"
            - options["classifier"] ... Classification algorithm (e.g. "MLPClassifier")
            - options["classifier_solver"] ... Classification algorithm (e.g. "adam" or "lbfgs")
        """
        self.results = results
        self.coords = coords
        self.options = options

        # set defaults
        if options is None:
            options = dict()
            options["clusterer"] = "KMeans"
            options["n_clusters"] = 2
            options["classifier"] = "MLPClassifier"
            options["classifier_solver"] = "lbfgs"

        # setup clusterer to determine domains (unsupervised learning)
        if options["clusterer"] == "KMeans":
            self.clusterer = KMeans(n_clusters=options["n_clusters"],
                                    random_state=42,
                                    n_jobs=-1,
                                    n_init=100)

        elif options["clusterer"] == "spectral_clustering":
            raise NotImplementedError("spectral projection not implemented yet")
            adjacency_matrix = None
            self.clusterer = spectral_clustering(adjacency_matrix,
                                                 n_clusters=options["n_clusters"],
                                                 random_state=0,
                                                 eigen_solver='arpack',
                                                 assign_labels="discretize")

        else:
            raise AttributeError("Please specify correct clusterer: {""KMeans"", ""spectral_clustering""...}")

        self.clusterer.fit(results)
        self.domains = self.clusterer.labels_
        self.swap_idx = np.arange(len(np.unique(self.domains)))

        # setup classifier for prediction (supervised learning)
        if options["classifier"] == "MLPClassifier":
            self.clf = MLPClassifier(alpha=0.01,
                                     max_iter=1000,
                                     activation="relu",
                                     solver=options["classifier_solver"])
        else:
            raise AttributeError("Please specify correct classifier: {""MLPClassifier"", ...}")

        self.clf.fit(coords, self.domains)

    def update(self, coords, results):
        """
        Updates classifier using the previous results

        Parameters
        ----------
        coords: ndarray of float [n_grid, n_dim]
            Grid points to train the classifier
        results: ndarray [n_grid x n_out]
            Results of the model evaluation
        """
        domains_old = copy.deepcopy(self.domains)

        # rerun clusterer
        self.clusterer.fit(results)
        self.domains = self.clusterer.labels_

        # check if domain labels are swapped and change it back to initial order
        domains_new = self.domains[:len(domains_old)]
        domains_unique = np.unique(domains_old)

        self.swap_idx = np.arange(len(domains_unique))
        for d in domains_unique:
            if np.mean(domains_old[domains_old == d] == domains_new[domains_old == d]) < 0.5:
                count = np.zeros(len(domains_unique))
                for di in domains_unique:
                    count[di] = np.sum(domains_new[domains_old == d] == di)

                if np.max(count) > 0:
                    self.swap_idx[d] = np.argmax(count)
                else:
                    self.swap_idx[d] = d

        domains_temp = np.zeros(self.domains.shape)

        for d in domains_unique:
            domains_temp[self.domains == d] = self.swap_idx[d]

        self.domains = domains_temp.astype(int)

        # rerun classifier
        self.clf.fit(coords, self.domains)

    def predict(self, coords):
        """
        Predict domains from new coordinates

        Parameters
        ----------
        coords: ndarray of float [n_grid, n_dim]
            Grid points to classify (has to be a 2D array)

        Returns
        -------
        domains: ndarray of float [n_grid, n_dim]
            Domain IDs of grid-points
        """

        domains = self.clf.predict(coords)

        return domains
