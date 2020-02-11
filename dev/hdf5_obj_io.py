import pygpc

fname_pkl_obj = "/data/pt_01756/software/git/pygpc/tests/tmp/pygpc_test_0_Static_gpc_quad.pkl"
fname_hdf5_obj = "/data/pt_01756/software/git/pygpc/tests/tmp/pygpc_test_0_Static_gpc_quad.hdf5"

# fname_pkl_obj = "/data/pt_01756/software/git/pygpc/tests/tmp/pygpc_test_7_MERegAdaptiveProjection_gpc.pkl"
# fname_hdf5_obj = "/data/pt_01756/software/git/pygpc/tests/tmp/pygpc_test_7_MERegAdaptiveProjection_gpc_obj.hdf5"

# fname_pkl_obj = "/home/kporzig/share/Dropbox/_share/gpc_hdf5/pygpc_test_4_MEStaticProjection_gpc.pkl"
# fname_hdf5_obj = "/home/kporzig/share/Dropbox/_share/gpc_hdf5/pygpc_test_4_MEStaticProjection_gpc.hdf5"


session_pkl = pygpc.read_session_pkl(fname_pkl_obj)

# pygpc.write_session_hdf5(gpc_pkl, fname_hdf5_obj)

session_hdf5 = pygpc.read_session_hdf5(fname=fname_hdf5_obj, folder="session")

a = 1

