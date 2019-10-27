import numpy as np
# ------------------------------ network settings ---------------------------------------------------
iteration = 70
lr = 0.5
D_in = 3
H = 30
D_out = 3
# ------------------------------ material parameter -------------------------------------------------
model_energy = 'NeoHookean3D'
K = 5.0/3.0*100
G = 1*100.0
E = (9*K*G)/(3*K+G)
nu = (3*K-2*G)/(2*(3*K+G))
param_c1 = 630
param_c2 = -1.2
param_c = 10000
# ----------------------------- define structural parameters ---------------------------------------
Length = 1.0
Height = 1.0
Depth = 5.999999

known_bt_ux = 0  # bottom ux
known_bt_uy = 0  # bottom uy
known_bt_uz = 0  # bottom uz

known_fr_tx = 0  # front tx
known_fr_ty = 24  # front ty
known_fr_tz = 0  # front tz

known_bk_tx = 0  # back tx
known_bk_ty = -24  # back ty
known_bk_tz = 0  # back tz

# bc_bt_penalty = 1000.0
bc_bt_penalty = 1.0
bc_fr_penalty = 1.0
bc_bk_penalty = 1.0
# ------------------------------ define domain and collocation points -------------------------------
Nx = 10  # 20
Ny = 10  # 20
Nz = 60  # 120
Nx_2 = 30  # 60
Ny_2 = 10  # 20
Nz_2 = 10  # 20

x_min, y_min, z_min = (0.0, 0.0, 0.0)
x_max, y_max, z_max = (Length, Height, Depth)
Length, Height, Depth = (x_max, y_max, z_max)
hx1, hy1, hz1 = Length / (Nx - 1), Height / (Ny - 1), Depth / (Nz - 1)
shape1 = [Nx, Ny, Nz]
dxdydz1 = [hx1, hy1, hz1]

x_min_2, y_min_2, z_min_2 = (-1.0, 0.0, 6.0)
x_max_2, y_max_2, z_max_2 = (2.0, 1.0, 7.0)
hx2, hy2, hz2 = (x_max_2 - x_min_2) / (Nx_2 - 1), (y_max_2 - y_min_2) / (Ny_2 - 1), (z_max_2 - z_min_2) / (Nz_2 - 1)
shape2 = [Nx_2, Ny_2, Nz_2]
dxdydz2 = [hx2, hy2, hz2]
shape3 = [int(Nx_2/2), Ny_2]
dxdy = ((x_max_2 - x_min_2)/2) / (int(Nx_2/2) - 1), (y_max_2 - y_min_2) / (Ny_2 - 1)
# ------------------------------ neural network definition ------------------------------------------

# ------------------------------ data testing -------------------------------------------------------
num_test_x = Nx
num_test_y = Ny
num_test_z = Nz
# ------------------------------ filename output ----------------------------------------------------
filename_out = "./NeoHook3D_TwistingTbar_Pressure24_v11"