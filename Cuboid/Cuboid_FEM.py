from dolfin import *
import time

start = time.time()
# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
# parameters["form_compiler"]["representation"] = "quadrature"  # change quadrature to uflacs if there's problem
# parameters["form_compiler"]["quadrature_degree"] = 2
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
# mesh = UnitCubeMesh(40, 20, 20)
p0 = Point(0.0, 0.0, 0.0)
p1 = Point(1.25, 1.0, 1.0)
mesh = BoxMesh(p0, p1, 50, 40, 40)  # Number of elements
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Mark boundary subdomians
left = CompiledSubDomain("near(x[0], side) && on_boundary", side=0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side=1.25)

# Define Dirichlet boundary (x = 0 or x = 1)
c = Expression(("0.0", "0.0", "0.0"), degree=2)
r = Expression(("scale*0.0",
                "scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])",
                "scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])"),
                scale = 0.5, y0 = 0.5, z0 = 0.5, theta = pi/3, degree=2)

bcl = DirichletBC(V, c, left)
bcr = DirichletBC(V, r, right)
bcs = [bcl, bcr]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, -0.5, 0.0))  # Body force per unit volume
T  = Constant((1.0,  0.0, 0.0))  # Traction force on the boundary

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)            # Deformation gradient
C = variable(F.T*F)          # Right Cauchy-Green tensor
EE = 0.5 * (C - I)
# Invariants of deformation tensors
Ic = tr(C)
J = sqrt(det(C))
# J  = det(F)

# Elasticity parameters
E, nu = 10**6, 0.3
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)
  # Green-Lagrange strain tensor
S = 2*diff(psi, C)  # Second Piola-Kirchhoff stress tensor
# P = dot(F, S)
# Compute Jacobian of F
J = derivative(F, u, du)
problem = NonlinearVariationalProblem(F, u, bcs, J)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters
# prm['newton_solver']['absolute_tolerance'] = 1E-8
# prm['newton_solver']['relative_tolerance'] = 1E-7
# prm['newton_solver']['maximum_iterations'] = 25
# prm['newton_solver']['relaxation_parameter'] = 1.0
prm['newton_solver']['linear_solver'] = 'gmres'
# prm['newton_solver']['linear_solver'] = 'minres'
solver.solve()
# Solve variational problem
# solve(F == 0, u, bcs, J=J, form_compiler_parameters=ffc_options)
# solve(F == 0, u, bcs, J=J, solver_parameters={"linear_solver":"lu"})
# Save solution in VTK format
vtkfile = File("./output/fem/Cuboid_U.pvd")
vtkfile << u
# Project and write stress field to post-processing file
# W = TensorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
# GreenStrain = project(EE, V=W)
# File("Cuboid_E.pvd") << GreenStrain
# W = TensorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
# secondPiola = project(S, V=W)
# File("Cuboid_S.pvd") << secondPiola
# Plot and hold solution
plot(u, mode="displacement", interactive=True)
L2 = inner(u, u) * dx
H10 = inner(grad(u), grad(u)) * dx
energynorm = sqrt(assemble(psi*dx))
# H1 = inner(F, P) * dx
L2norm = sqrt(assemble(L2))
H10norm = sqrt(assemble(H10))
print("L2 norm = %.10f" % L2norm)
print("H1 norm = %.10f" % sqrt(L2norm**2 + H10norm**2))
print("H10 norm (H1 seminorm) = %.10f" % H10norm)
print("H1 norm = %.10f" % energynorm)
print("L2 norm = %.10f" % norm(u, norm_type="L2"))
print("H1 norm = %.10f" % norm(u, norm_type="H1"))
print("H10 norm = %.10f" % norm(u, norm_type="H10"))
print("Running time = %.3f" % float(time.time()-start))
# u_P1 = project(u, V)
u_nodal_values = u.vector()
u_values = u.compute_vertex_values()
# array_u = u_nodal_values.array()
# Plot solution
plot(u, title='Displacement', mode='displacement')

F = I + grad(u)
P = mu * F + (lmbda * ln(det(F)) - mu) * inv(F).T
secondPiola = inv(F) * P
Sdev = secondPiola - (1./3)*tr(secondPiola)*I # deviatoric stress
von_Mises = sqrt(3./2*inner(Sdev, Sdev))
V = FunctionSpace(mesh, "Lagrange", 1)
W = TensorFunctionSpace(mesh, "Lagrange", 1)
von_Mises = project(von_Mises, V)
Stress = project(secondPiola, W, solver_type='gmres')
plot(von_Mises, title='Stress intensity')

# Compute magnitude of displacement
u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, V)
plot(u_magnitude, 'Displacement magnitude')
# print('min/max u:',
#       u_magnitude.vector().array().min(),
#       u_magnitude.vector().array().max())
# Save solution to file in VTK format
File('./output/fem/elasticity/displacement.pvd') << u
File('./output/fem/elasticity/von_mises.pvd') << von_Mises
File('./output/fem/elasticity/magnitude.pvd') << u_magnitude
File('./output/fem/elasticity/Stress.pvd') << Stress