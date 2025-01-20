import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import asQ
from utils.timing import SolverTimer
import time

Print = PETSc.Sys.Print

slice_length = 1
nslices = 4
time_partition = [slice_length for _ in range(nslices)]

global_comm = COMM_WORLD
ensemble = asQ.create_ensemble(time_partition, comm=global_comm)

# Mesh
mesh_res = 8
mesh = SquareMesh(mesh_res, mesh_res, 1.0, quadrilateral=True, comm=ensemble.comm)

# Function spaces
# NB: Canonical name for DG elements on a quadrilateral is DQ
degree = 1
V_w = FunctionSpace(mesh, "DQ", degree) #, variant="spectral")  # for w (vorticity)
V_n = FunctionSpace(mesh, "DQ", degree) #, variant="spectral")  # for n (electron density)
V_phi = FunctionSpace(mesh, "CG", degree)  # for phi (electrostatic potential) - to be obtained separately
V = V_w * V_n * V_phi

V_driftvel = VectorFunctionSpace(mesh, "CG", degree)  # for driftvel (drift velocity) - to be obtained separately

# Time parameters
T = 5.0 # end time
time_res = 50
skip = 1
t = Constant(0.0) # current time
dt = Constant(T / time_res) # time-step size

# Time-stepper parameters:
theta = 0.5 # actually trapezoidal rule

# Model parameters
L_par = 10.0
height = 0.5
blob_width = 0.05  # width

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

blob2d_params = {
    'ksp_rtol': 1e-5,
    'ksp_type': 'fgmres',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'multiplicative',
    'pc_fieldsplit_fields_0': '2',
    'pc_fieldsplit_fields_1': '0,1',
    'fieldsplit_0': {
        'ksp_rtol': 1e-5,
        'ksp_type': 'chebyshev',
        'pc_type': 'bjacobi',
        'sub_pc_type': 'ilu',
        'ksp_reuse_preconditioner': None,
    },
    'fieldsplit_1': {
        'ksp_rtol': 1e-5,
        'ksp_type': 'gmres',
        'pc_type': 'bjacobi',
        'sub_pc_type': 'ilu',
    },
}

patch_params = {
    'ksp_rtol': 1e-5,
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

atol = 1e-7
solver_params = {
    'snes_converged_reason': None,
    'snes_monitor': None,
    'snes_max_it': 100,
    'snes_rtol': 1e-6,
    'snes_atol': atol,
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_rate': None,
        'rtol': 1e-2,
        'atol': atol,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.CirculantPC',
    'circulant_alpha': 1e-4,
    'circulant_block': patch_params,
}

window_length = sum(time_partition)
for i in range(window_length):
    solver_params[f'circulant_block_{i}_ksp_converged_rate'] = f':block_{i}_ksp.log'

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
)

# Main loop
is_last_slice = pdg.layout.is_local(-1)
if is_last_slice:
    output_file = VTKFile("Blob2D_output.pvd", comm=ensemble.comm)
    output_file.write(*solution.subfunctions)
start_time = time.time()

window_duration = window_length*dt
nwindows = T/window_duration

timer = SolverTimer()

def window_preproc(pdg, wndw, rhs):
    Print('')
    Print(f'### === --- Calculating time-window {wndw} --- === ###')
    Print('')
    timer.start_timing()


def window_postproc(pdg, wndw, rhs):
    timer.stop_timing()
    Print('', comm=global_comm)
    Print(f'Window solution time: {timer.times[-1]}', comm=global_comm)
    Print('', comm=global_comm)

    # postprocess this timeslice
    if is_last_slice:
        solution.assign(pdg.aaofunc[-1])
        w_s, n_s, phi_s = solution.subfunctions
        output_file.write(w_s, n_s, phi_s)

        nt = pdg.total_windows*pdg.ntimesteps - 1
        time = float(nt*pdg.aaoform.dt)
        Print(f'Time = {round(time, 3)}', comm=ensemble.comm)

pdg.solve(nwindows=5, # =nwindows
          preproc=window_preproc,
          postproc=window_postproc)

end_time = time.time()
Print(f"Simulation complete. Total wall time: {end_time - start_time:.2f} seconds")

# Save output at end time
if is_last_slice:
    solution.assign(pdg.aaofunc[-1])
    VTKFile("Blob2D_output_final.pvd", comm=ensemble.comm).write(*solution.subfunctions)
