import numpy as np
# ------------------------------ network settings ---------------------------------------------------
iteration = 40
lr = 0.5
D_in = 3
H = 30
D_out = 3
# ------------------------------ material parameter -------------------------------------------------
model_energy = 'NeoHookean3D'
E = 10**6
nu = 0.3
param_c1 = 630
param_c2 = -1.2
param_c = 10000
# ----------------------------- define structural parameters ---------------------------------------
Length = 1.25
Height = 1.0
Depth = 1.0
known_left_ux = 0
known_left_uy = 0
known_left_uz = 0

known_right_ux = lambda x: x - x
known_right_uy = lambda y, z: 0.5 * (0.5 + (y - 0.5)*np.cos(np.pi/3) - (z - 0.5)*np.sin(np.pi/3) - y)
known_right_uz = lambda y, z: 0.5 * (0.5 + (y - 0.5)*np.sin(np.pi/3) + (z - 0.5)*np.cos(np.pi/3) - z)
body_force_x = 0.0
body_force_y = -0.5
body_force_z = 0.0

# penalty1 = 10000000
# penalty2 = 9000000

bc_left_penalty = 1  # not using in this example
bc_right_penalty = 1   # not using in this example
bc_sr1_penalty = 1   # not using in this example
bc_sr2_penalty = 1   # not using in this example
bc_sr3_penalty = 1   # not using in this example
bc_sr4_penalty = 1   # not using in this example
# ------------------------------ define domain and collocation points -------------------------------
Nx = 40  # 120  # 120
Ny = 40  # 30  # 60
Nz = 40  # 30  # 10
x_min, y_min, z_min = (0.0, 0.0, 0.0)
hx = Length / (Nx - 1)
hy = Height / (Ny - 1)
hz = Depth / (Nz - 1)

shape = [Nx, Ny, Nz]
dxdydz = [hx, hy, hz]
# ------------------------------ neural network definition ------------------------------------------

# ------------------------------ data testing -------------------------------------------------------
num_test_x = 40
num_test_y = 40
num_test_z = 40
# ------------------------------ filename output ----------------------------------------------------
# filename_out = "./NeoHook3D_TwistingCubic_v16"
filename_out = "./output/dem/result_CuboidNeoHook_train10x10x10_iter40"