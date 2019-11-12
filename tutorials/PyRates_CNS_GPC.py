import os
import h5py
import pygpc
from collections import OrderedDict
from tutorials.PyRates_CNS_Model import PyRates_CNS_Model
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

fn_results = os.path.join(os.path.split(pygpc.__path__[0])[0], "tutorials", "datasets", "PyRates_CNS_GPC_new")

model = PyRates_CNS_Model()

# define problem (the parameter names have to be the same as in the model)
parameters = OrderedDict()
parameters["w_ein_pc"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[5.4, 21.6])
parameters["w_iin_pc"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[11.8125, 47.25])
problem = pygpc.Problem(model, parameters)

# gPC options
options = dict()
options["solver"] = "LarsLasso"
options["settings"] = None
options["order_start"] = 3
options["order_end"] = 15
options["seed"] = 1
options["projection"] = False
options["order_max_norm"] = 1.
options["interaction_order"] = 2
options["matrix_ratio"] = 2
options["n_cpu"] = 0
options["gradient_enhanced"] = False
options["gradient_calculation"] = "standard_forward"
options["error_type"] = "loocv"
options["error_norm"] = "absolute"
options["n_samples_validation"] = 1e2
options["qoi"] = 0
options["classifier"] = "learning"
options["n_samples_discontinuity"] = 10
options["adaptive_sampling"] = True
options["eps"] = 0.03
options["n_grid_init"] = 100
options["GPU"] = False
options["fn_results"] = fn_results
options["classifier_options"] = {"clusterer": "KMeans",
                                 "n_clusters": 2,
                                 "classifier": "MLPClassifier",
                                 "classifier_solver": "lbfgs"}

# define algorithm
algorithm = pygpc.MERegAdaptiveProjection(problem=problem, options=options)

# Initialize gPC Session
session = pygpc.Session(algorithm=algorithm)

# run gPC session
session, coeffs, results = session.run()

# Post-process gPC
pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                             output_idx=None,
                             calc_sobol=True,
                             calc_global_sens=True,
                             calc_pdf=True,
                             algorithm="sampling",
                             n_samples=1e4)

with h5py.File(fn_results + "_validation.hdf5", "r") as f:
    val_coords_norm = f["grid/coords_norm"][:]
    val_coords = f["grid/coords"][:]
    val_results = f["model_evaluations/original_all_qoi"][:]

# Validate gPC vs original model function (2D-slice)
pygpc.validate_gpc_plot(session=session,
                        coeffs=coeffs,
                        random_vars=["w_ein_pc", "w_iin_pc"],
                        coords=val_coords,
                        data_original=val_results,
                        output_idx=0,
                        fn_out=session.fn_results + '_val',
                        n_cpu=session.n_cpu)

# Validate gPC vs original model function (Monte Carlo)
nrmsd = pygpc.validate_gpc_mc(session=session,
                              coeffs=coeffs,
                              n_samples=int(1e3),
                              n_cpu=session.n_cpu,
                              output_idx=0,
                              fn_out=session.fn_results + '_mc')
