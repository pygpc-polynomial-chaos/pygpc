import multiprocessing
import multiprocessing.pool
from _functools import partial
from .misc import compute_chunks

# setting up parallelization
n_cpu_available = multiprocessing.cpu_count()
n_cpu = min(n_cpu, n_cpu_available, n_grid_new)
if n_cpu > 1:
    grid_new_chunks = compute_chunks(grid_new, n_cpu)
elif n_grid_new == 1:
    grid_new_chunks = [grid_new]
elif n_cpu == 1:
    grid_new_chunks = grid_new
else:
    raise StandardError('Number of CPU cores not specified correctly!')

# define workhorse function
workhorse_partial = partial(func,
                            n_max=n_max,                # *(args)
                            c=c,
                            E=E,
                            e_qoi=e_qoi,
                            tri_area=tri_area,
                            conds=conds,
                            create_paraview_files=create_paraview_files,
                            subject=subject,
                            roi=roi,
                            roi_idx=roi_idx,
                            seg_idx=seg_idx,
                            mesh_idx=mesh_idx,
                            fn_c_results=fn_c_results,
                            fn_csv_out=fn_csv_out,
                            fn_paraview_out_prefix=fn_paraview_out_prefix)

pool = multiprocessing.Pool(n_cpu)
res_new_list = pool.map(workhorse_partial, grid_new_chunks)

pool.close()
pool.join()

# transforming from list (from zip and multiprocessing) to nparray
res_new = np.vstack(res_new_list)
