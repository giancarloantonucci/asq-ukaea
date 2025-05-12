import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import asQ
from utils.timing import SolverTimer
import time

Print = PETSc.Sys.Print

# Time parameters
T = 1.0 # end time
dt = 0.025
# time_res = 50
# dt = Constant(T / time_res) # time-step size
skip = 1

# window_duration = window_length*dt
# nwindows = T/window_duration
nwindows = 30

slice_length = 1
nslices = 8

mesh_res = 32
degree = 4

fname=f'nx{mesh_res}_degree{degree}_dt025_nt{slice_length*nslices}'

time_partition = [slice_length for _ in range(nslices)]
window_length = sum(time_partition)

global_comm = COMM_WORLD
ensemble = asQ.create_ensemble(time_partition, comm=global_comm)

# Mesh
mesh = SquareMesh(mesh_res, mesh_res, 1.0, quadrilateral=True, comm=ensemble.comm)

# Function spaces
# NB: Canonical name for DG elements on a quadrilateral is DQ
V_w = FunctionSpace(mesh, "DQ", degree) #, variant="spectral")  # for w (vorticity)
V_n = FunctionSpace(mesh, "DQ", degree) #, variant="spectral")  # for n (electron density)
V_phi = FunctionSpace(mesh, "CG", degree)  # for phi (electrostatic potential) - to be obtained separately
V = V_w * V_n * V_phi

Print(f'Element degree = {degree}')
Print(f'Total number of ranks = {global_comm.size}')
Print(f'Total number of DoFs = {window_length*V.dim()}')
Print(f'Number of ranks per timestep = {ensemble.comm.size}')
Print(f'Number of DoFs per timestep = {V.dim()}')
Print(f'Number of DoFs per rank = {V.dim()/ensemble.comm.size}')
Print(f'Nt = {window_length} | dt = {round(dt, 4)} | Nt*dt = {window_length*dt:.4f} | Nw = {nwindows}')

V_driftvel = VectorFunctionSpace(mesh, "CG", degree)  # for driftvel (drift velocity) - to be obtained separately

# Time-stepper parameters:
theta = 0.5 # actually trapezoidal rule

# Model parameters
L_par = Constant(10.0)
height = Constant(0.5)
blob_width = Constant(0.05)

# Coordinates
x, y = SpatialCoordinate(mesh)

def form_mass(w, n, phi, v_w, v_n, v_phi):
    return (
        w * v_w * dx
        +  n * v_n * dx
    )

norm = FacetNormal(mesh)
def form_function(w, n, phi, v_w, v_n, v_phi, t):
    driftvel = as_vector([grad(phi)[1], -grad(phi)[0]])
    driftvel_n = 0.5 * (dot(driftvel, norm) + abs(dot(driftvel, norm)))
    return (
        inner(grad(phi), grad(v_phi)) * dx
        + w * v_phi * dx
        - div(w * driftvel) * v_w * dx
        + Constant(20.0 / 9.0) * grad(n)[1] * v_w * dx
        - div(n * driftvel) * v_n * dx
        - phi * n * (v_w + v_n) / L_par * dx
        + driftvel_n('-') * ( w('-') - w('+') ) * v_w('-') * dS
        + driftvel_n('+') * ( w('+') - w('-') ) * v_w('+') * dS
        + driftvel_n('-') * ( n('-') - n('+') ) * v_n('-') * dS
        + driftvel_n('+') * ( n('+') - n('-') ) * v_n('+') * dS
    )

# Functions
solution = Function(V)
w_s, n_s, phi_s = solution.subfunctions

w_s.rename("vorticity")
n_s.rename("density")
phi_s.rename("potential")

# Initial conditions
w_s.interpolate(0.0)
n_s.interpolate(1 + height * exp(-((x - 0.5)**2 + (y - 0.5)**2) / (blob_width**2)))
phi_s.interpolate(0.0) # NB: actually, do one solve for both to get a better initial cond

# Boundary condition for phi
bc_phi = DirichletBC(V.sub(2), 0, 'on_boundary')

fieldsplit_params = {
    'ksp_type': 'fgmres',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'multiplicative',
    'pc_fieldsplit_fields_0': '2',
    'pc_fieldsplit_fields_1': '0,1',
    'fieldsplit_0': {
        'ksp_reuse_preconditioner': None,
        # 'ksp_type': 'preonly',
        # 'pc_type': 'lu',
        # 'pc_factor_mat_solver_type': 'mumps',
        'ksp_rtol': 1e-2,
        'ksp_type': 'chebyshev',
        'pc_type': 'hypre',
    },
    'fieldsplit_1': {
        'ksp_rtol': 1e-2,
        'ksp_type': 'gmres',
        'pc_type': 'bjacobi',
        'sub_pc_type': 'ilu',
    },
}

patch_params = {
    'mat_type': 'matfree',
    'ksp_type': 'gmres',
    'pc_type': 'python',
    'pc_python_type': 'firedrake.PatchPC',
    'patch': {
        'pc_patch': {
            'save_operators': True,
            'partition_of_unity': True,
            'sub_mat_type': 'seqdense',
            'construct_dim': 0,
            'construct_type': 'star',
            'local_type': 'additive',
            'precompute_element_tensors': True,
            'symmetrise_sweep': False
        },
        'sub': {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_shift_type': 'nonzero',
        }
    }
}

block_params = patch_params
block_params['ksp_rtol'] = 1e-5

circulant_params = {
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'circulant_alpha': 1e-4,
    'circulant_block': block_params,
    'circulant_block_0_ksp_view': ':block_ksp_view.log',
    **{f'circulant_block_{i}_ksp_converged_rate': f':block_{i}_ksp.log'
       for i in range(window_length)}
}

atol = 1e-7
newton_params = {
    'snes_linesearch_type': 'basic',
    'snes': {
        'converged_reason': None,
        'monitor_short': None,
        'rtol': 1e-10,
        'atol': atol,
        "ksp_ew": None,
        "ksp_ew_version": 1,
        "ksp_ew_rtol0": 5e-1,
        "ksp_ew_threshold": 1e-1,
    },
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor_short': None,
        'converged_rate': None,
        'rtol': 1e-2,
        'atol': atol,
    },
    **circulant_params,
}

newton_smoother = {
    'snes_linesearch_type': 'basic',
    'snes_converged_reason': None,
    'snes_monitor_short': None,
    'ksp_monitor_short': None,
    'ksp_converged_rate': None,
    **circulant_params,
}

composite_params = {
    'snes': {
        'converged_reason': None,
        'monitor_short': None,
        'rtol': 1e-10,
        'atol': atol,
    },
    'snes_type': 'composite',
    'snes_composite_type': 'multiplicative',
    'snes_composite_sneses': 'newtonls,newtonls',
    'sub_0_snes_max_it': 2,
    'sub_0': {
        **newton_smoother,
        'ksp_type': 'gmres',
        'ksp_pc_side': 'right',
        'ksp_max_it': 1,
        'ksp_converged_maxits': None,
        'snes_lag_preconditioner': -2,
    },
    'sub_1_snes_max_it': 1,
    'sub_1': {
        **newton_smoother,
        'ksp_type': 'fgmres',
        'ksp_atol': atol,
        'snes_ksp': {
            'ew': None,
            'ew_version': 1,
            'ew_rtol0': 1e-2,
            'ew_threshold': 1e-2,
        }
    },
}

solver_params = newton_params
solver_params['snes_view'] = ':snes_view.log'

pdg = asQ.Paradiag(
    ensemble=ensemble,
    form_mass=form_mass,
    form_function=form_function,
    ics=solution,
    dt=dt,
    theta=theta,
    time_partition=time_partition,
    solver_parameters=solver_params,
    bcs=[bc_phi],
    options_prefix='pdg'
)

# Main loop
is_last_slice = pdg.layout.is_local(-1)
if is_last_slice:
    output_file = VTKFile(f"output/Blob2D_{fname}.pvd", comm=ensemble.comm)
    output_file.write(*solution.subfunctions, time=0)
start_time = time.time()

timer = SolverTimer()

def window_preproc(pdg, wndw, rhs):
    Print('')
    Print(f'### === --- Calculating time-window {wndw} --- === ###')
    Print('')
    timer.start_timing()


def window_postproc(pdg, wndw, rhs):
    timer.stop_timing()
    Print('', comm=global_comm)
    Print(f'Window solution time: {timer.times[-1]:.3f}', comm=global_comm)
    Print('', comm=global_comm)

    # postprocess this timeslice
    if is_last_slice:
        nt = pdg.total_windows*pdg.ntimesteps
        time = float(nt*pdg.aaoform.dt)
        Print(f'\nTime = {time:.3f}', comm=ensemble.comm)
        solution.assign(pdg.aaofunc[-1])
        output_file.write(w_s, n_s, phi_s, time=time)


pdg.solve(nwindows=nwindows,
          preproc=window_preproc,
          postproc=window_postproc)

end_time = time.time()
Print(f"Simulation complete. Total wall time: {end_time - start_time:.2f} seconds")

# Save output at end time
if is_last_slice:
    solution.assign(pdg.aaofunc[-1])
    VTKFile(f"output/Blob2D_final_{fname}.pvd", comm=ensemble.comm).write(w_s, n_s, phi_s, time=nwindows*window_length*dt)
