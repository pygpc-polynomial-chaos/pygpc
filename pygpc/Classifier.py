from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier


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
        return ClassifierLearning(coords=coords, results=results)
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

        if options is None:
            options = dict()
            options["clusterer"] = "KMeans"
            options["n_clusters"] = 2
            options["classifier"] = "MLPClassifier"
            options["classifier_solver"] = "lbfgs"

        # determine classes of function values and associated coords
        if options["clusterer"] == "KMeans":
            clusterer = KMeans(n_clusters=options["n_clusters"], random_state=0)
        else:
            raise AttributeError("Please specify correct clusterer: {""KMeans"", ...}")

        clusterer.fit(results)
        self.domains = clusterer.labels_

        # setup classifier for prediction
        self.clf = MLPClassifier(alpha=0.01, max_iter=1000, activation="relu", solver=options["classifier_solver"])
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
