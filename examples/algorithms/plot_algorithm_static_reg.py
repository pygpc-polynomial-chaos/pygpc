"""
Algorithm: Static (Regression)
==============================
"""
def main():
    import pygpc
    from collections import OrderedDict

    fn_results = 'tmp/static_reg'   # filename of output
    save_session_format = ".pkl"    # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)

    #%%
    # Loading the model and defining the problem
    # ------------------------------------------

    # define model
    model = pygpc.testfunctions.Peaks()

    # define problem
    parameters = OrderedDict()
    parameters["x1"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[1.2, 2])
    parameters["x2"] = 1.25
    parameters["x3"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 0.6])
    problem = pygpc.Problem(model, parameters)

    #%%
    # Setting up the algorithm
    # ------------------------

    # gPC options
    options = dict()
    options["method"] = "reg"
    options["solver"] = "Moore-Penrose"
    options["settings"] = None
    options["order"] = [9, 9]
    options["order_max"] = 9
    options["interaction_order"] = 2
    options["matrix_ratio"] = 20
    options["error_type"] = "nrmsd"
    options["n_samples_validation"] = 1e3
    options["n_cpu"] = 0
    options["fn_results"] = fn_results
    options["save_session_format"] = save_session_format
    options["gradient_enhanced"] = True
    options["gradient_calculation"] = "FD_1st2nd"
    options["gradient_calculation_options"] = {"dx": 0.05, "distance_weight": -2}
    options["backend"] = "omp"
    options["grid"] = pygpc.Random
    options["grid_options"] = None

    # determine number of basis functions
    n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
                                           order_glob_max=options["order_max"],
                                           order_inter_max=options["interaction_order"],
                                           dim=problem.dim)
    # generate grid
    grid = pygpc.Random(parameters_random=problem.parameters_random,
                        n_grid=options["matrix_ratio"] * n_coeffs,
                        options={"seed": 1})

    # initialize algorithm
    algorithm = pygpc.Static(problem=problem, options=options, grid=grid)

    #%%
    # Running the gpc
    # ---------------

    # initialize gPC Session
    session = pygpc.Session(algorithm=algorithm)

    # run gPC algorithm
    session, coeffs, results = session.run()

    #%%
    # Postprocessing
    # --------------

    # read session
    session = pygpc.read_session(fname=session.fn_session, folder=session.fn_session_folder)

    # Post-process gPC
    pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                 output_idx=None,
                                 calc_sobol=True,
                                 calc_global_sens=True,
                                 calc_pdf=True,
                                 algorithm="standard",
                                 n_samples=1e3)

    #%%
    # Validation
    # ----------
    # Validate gPC vs original model function (2D-surface)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    pygpc.validate_gpc_plot(session=session,
                            coeffs=coeffs,
                            random_vars=list(problem.parameters_random.keys()),
                            n_grid=[51, 51],
                            output_idx=[0],
                            fn_out=None,
                            folder=None,
                            n_cpu=session.n_cpu)
    #%%
    # Validate gPC vs original model function (Monte Carlo)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    nrmsd = pygpc.validate_gpc_mc(session=session,
                                  coeffs=coeffs,
                                  n_samples=int(1e4),
                                  output_idx=[0],
                                  fn_out=None,
                                  folder=None,
                                  plot=True,
                                  n_cpu=session.n_cpu)

    print("> Maximum NRMSD (gpc vs original): {:.2}%".format(max(nrmsd)))

if __name__ == '__main__':
    main()