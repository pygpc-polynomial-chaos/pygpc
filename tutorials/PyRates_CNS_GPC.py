import h5py
import pygpc
from collections import OrderedDict
from tutorials.PyRates_CNS_Model import PyRates_CNS_Model

fn_results = f'/NOBACKUP2/tmp/PyRates_CNS_GPC/PyRates_CNS_GPC'
model = PyRates_CNS_Model

# define problem (the parameter names have to be the same as in the model)
parameters = OrderedDict()
parameters["w_ein_pc"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0.5*0.8*13.5, 2.*0.8*13.5])
parameters["w_iin_pc"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0.5*1.75*13.5, 2.*1.75*13.5])
problem = pygpc.Problem(model, parameters)

# gPC options
options = dict()
options["method"] = "reg"
options["solver"] = "Moore-Penrose"
options["settings"] = None
options["order_start"] = 2
options["order_end"] = 15
# options["order"] = [5, 5]
# options["n_grid_gradient"] = 20
# options["order_max"] = 5
options["projection"] = True
options["order_max_norm"] = 1.
options["interaction_order"] = 2
options["matrix_ratio"] = 2
options["n_cpu"] = 0
options["gradient_enhanced"] = False
options["gradient_calculation"] = "standard_forward"
options["error_type"] = "nrmsd"
options["n_samples_validation"] = 1e2
options["qoi"] = "all"
options["classifier"] = "learning"
options["classifier_options"] = {"clusterer": "KMeans",
                                 "n_clusters": 2,
                                 "classifier": "MLPClassifier",
                                 "classifier_solver": "lbfgs"}
options["n_samples_discontinuity"] = 5
options["adaptive_sampling"] = False
options["eps"] = 0.05
options["n_grid_init"] = 50
options["GPU"] = False
options["fn_results"] = fn_results

# load validation set
validation = pygpc.ValidationSet().read(fname="/NOBACKUP2/tmp/PyRates_CNS_GPC/PyRates_CNS_GPC_validation_plot.hdf5",
                                        results_key="model_evaluations/original_all_qoi")

# define algorithm
# algorithm = pygpc.MEStaticProjection(problem=problem, options=options, validation=validation)
algorithm = pygpc.MERegAdaptiveProjection(problem=problem, options=options, validation=validation)

# run gPC algorithm
gpc, coeffs, results = algorithm.run()

# # plot 2D grid (only feasible for 2D problems)
# pygpc.plot_2d_grid(coords=gpc.grid.coords,
#                    fn_plot=fn_results + '_grid')

# # Post-process gPC
pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                             output_idx=None,
                             calc_sobol=True,
                             calc_global_sens=True,
                             calc_pdf=True)

# # # Validate gPC vs original model function (Monte Carlo)
nrmsd = pygpc.validate_gpc_mc(gpc=gpc,
                              coeffs=coeffs,
                              n_samples=int(1e4),
                              output_idx=0,
                              fn_out=options["fn_results"] + '_validation_mc')

with h5py.File("/NOBACKUP2/tmp/PyRates_CNS_GPC/PyRates_CNS_GPC_validation_plot.hdf5") as f:
    val_coords_norm = f["grid/coords_norm"][:]
    val_coords = f["grid/coords"][:]
    val_results = f["model_evaluations/original_all_qoi"][:]

# Validate gPC vs original model function (2D-slice)
pygpc.validate_gpc_plot(gpc=gpc,
                        coeffs=coeffs,
                        random_vars=["w_ein_pc", "w_iin_pc"],
                        coords=val_coords,
                        data_original=val_results,
                        output_idx=1,
                        fn_out=options["fn_results"] + '_validation_plot',
                        n_cpu=0)

# # Validate gPC vs original model function (2D-slice)
pygpc.validate_gpc_plot(gpc=gpc,
                        coeffs=coeffs,
                        random_vars=["w_ein_pc", "w_iin_pc"],
                        n_grid=[51, 51],
                        output_idx=0,
                        fn_out=options["fn_results"] + '_validation_2d',
                        n_cpu=0)

# print("\t > Maximum NRMSD (gpc vs original): {:.2}%".format(np.max(nrmsd)))

print("done!\n")
