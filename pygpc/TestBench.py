class TestBench(object):
    """
    TestBench for gPC algorithms
    """

    def __init__(self, algorithm_label, tests):
        """
        Initializes the Testbench class object instance

        Parameters
        ----------
        algorithm_label : str
            Algorithm to benchmark
        test_labels : test objects
            Tests to perform ["2D", "ND", "discontinuous"]
        """
        self.algorithm_label = algorithm_label
        self.test = []

        # create tests
        for t in tests:
            self.test.append(t(t))


class Test(object):
    """
    Test for gPC algorithms
    """
    def __init__(self):
        """
        Initializes Test class
        """
        pass


class Test2D(TestBench):
    """
    TestBench including 2D testfunctions
    """
    def __init__(self):
        """
        Initializes Test2D class
        """
        super(Test2D, self).__init__()

        # Peaks




class TestDiscontinuous(TestBench):
    """
    Test including discontinuous testfunctions
    """
    def __init__(self):
        """
        Initializes TestDiscontinuous class
        """
        super(TestDiscontinuous, self).__init__()

        # HyperbolicTangent
        # MovingParticleFrictionForce



class TestND(TestBench):
    """
    Test including 2D testfunctions
    """
    def __init__(self):
        """
        Initializes Test2D class
        """
        super(TestND, self).__init__()

