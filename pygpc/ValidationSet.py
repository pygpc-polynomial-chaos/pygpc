class ValidationSet(object):
    """
    ValidationSet object
    """

    def __init__(self, grid=None, results=None):
        """
        Initializes ValidationSet

        Parameters
        ----------
        grid : Grid object
            Grid object containing the validation points (grid.coords, grid.coords_norm)
        results: ndarray [n_grid x n_out]
            Results of the model evaluation
        """
        self.grid = grid
        self.results = results
