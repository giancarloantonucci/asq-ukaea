import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import asQ
import time

time_partition = [4]

ensemble = asQ.create_ensemble(time_partition)

# Mesh
mesh_res = 32
mesh = SquareMesh(mesh_res, mesh_res, 1.0, quadrilateral=True, comm=ensemble.comm)

# Function spaces
# NB: Canonical name for DG elements on a quadrilateral is DQ
V_w = FunctionSpace(mesh, "DQ", 4) #, variant="spectral")  # for w (vorticity)
V_n = FunctionSpace(mesh, "DQ", 4) #, variant="spectral")  # for n (electron density)
V_phi = FunctionSpace(mesh, "CG", 4)  # for phi (electrostatic potential) - to be obtained separately
V = V_w * V_n * V_phi

V_driftvel = VectorFunctionSpace(mesh, "CG", 4)  # for driftvel (drift velocity) - to be obtained separately

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
# w, n, phi = split(solution) # asQ is doing the symbolic stuff
w_s, n_s, phi_s = solution.subfunctions
# v_w, v_n, v_phi = TestFunctions(V)

w_s.rename("vorticity")
n_s.rename("density")
phi_s.rename("potential")

# Initial conditions
w_s.interpolate(0.0)
n_s.interpolate(1 + height * exp(-((x - 0.5)**2 + (y - 0.5)**2) / (blob_width**2)))
phi_s.interpolate(0.0) # NB: actually, do one solve for both to get a better initial cond

# Save initial conditions for verification
# VTKFile("Blob2D_output_initial.pvd").write(solution.sub(0), solution.sub(1))

# Boundary condition for phi
bc_phi = DirichletBC(V.sub(2), 0, 'on_boundary')



# F = form_mass(w, n, phi, v_w, v_n, v_phi) + form_function(w, n, phi, v_w, v_n, v_phi, None)

# Solver parameters for phi
# linparams_phi = {
#     "mat_type": "aij",
#     "snes_type": "ksponly",
#     "ksp_type": "preonly",
#     "pc_type": "lu",
# }

# Time-stepper parameters, from Cahn-Hilliard example:

blob2d_params = {
    'ksp_rtol': 1e-5,
    'ksp_type': 'fgmres',
    # 'ksp_converged_reason': None,
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'multiplicative',
    'pc_fieldsplit_fields_0': '2',
    'pc_fieldsplit_fields_1': '0,1',
    'fieldsplit_0': {
        'ksp_rtol': 1e-5,
        'ksp_type': 'gmres',
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

atol = 1e-7
solver_params = {
    'snes_monitor': None,
    'snes_max_it': 100,
    'snes_rtol': 1e-6,
    'snes_converged_reason': None,
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
    'circulant_block': blob2d_params,
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
output_file = VTKFile("Blob2D_output.pvd")
start_time = time.time()
cnt = 0

while float(t) < float(T):
    if (float(t) + float(dt)) >= T:
        dt.assign(T - float(t))
        
    pdg.aaofunc.assign(pdg.aaofunc[-1])
    
    # Output results every 5 steps
    if(cnt % skip == 0):
        print("Saving output...\n")
        solution.assign(pdg.aaofunc.initial_condition)
        w_s, n_s, phi_s = solution.subfunctions
        output_file.write(w_s, n_s, phi_s)
        
    # Advance solution in time
    cnt += 1
    pdg.solve()
    # t.assign(float(t) + 4*float(dt))
    t += 4*float(dt)
    print(f"Current time: {float(t)}, Time step: {float(dt)}")

end_time = time.time()
print(f"Simulation complete. Total wall time: {end_time - start_time:.2f} seconds")

# Save output at end time
VTKFile("Blob2D_output_final.pvd").write(solution.sub(0), solution.sub(1), phi_s)
