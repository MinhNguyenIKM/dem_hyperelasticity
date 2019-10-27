from dem_hyperelasticity.config import *
from dem_hyperelasticity.TStructure.config import *


def setup_domain():
    x_dom = x_min, x_max, Nx
    y_dom = y_min, y_max, Ny
    z_dom = z_min, z_max, Nz
    # create points
    lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
    lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
    lin_z = np.linspace(z_dom[0], z_dom[1], z_dom[2])
    # dom = np.zeros((Nx * Ny * Nz, 3))
    xGrid, yGrid, zGrid = np.meshgrid(lin_x, lin_y, lin_z)
    dom1 = np.concatenate((np.array([xGrid.flatten()]).T, np.array([yGrid.flatten()]).T, np.array([zGrid.flatten()]).T), axis=1)


    lin_x_2 = np.linspace(x_min_2, x_max_2, Nx_2)
    lin_y_2 = np.linspace(y_min_2, y_max_2, Ny_2)
    lin_z_2 = np.linspace(z_min_2, z_max_2, Nz_2)
    xGrid2, yGrid2, zGrid2= np.meshgrid(lin_x_2, lin_y_2, lin_z_2)
    dom2 = np.concatenate((np.array([xGrid2.flatten()]).T, np.array([yGrid2.flatten()]).T, np.array([zGrid2.flatten()]).T), axis=1)
    dom = np.unique(np.concatenate((dom1, dom2), axis=0), axis=0)
    # intersection = intersect2Arrays(dom1, dom2)

    # dom1_without_common, common = union2DArray(dom1, dom2)
    # dom = np.concatenate((dom1_without_common, dom2), axis=0)
    # c = 0
    # for z in np.nditer(lin_z):
    #     for x in np.nditer(lin_x):
    #         tb = y_dom[2] * c
    #         te = tb + y_dom[2]
    #         c += 1
    #         dom[tb:te, 0] = x
    #         dom[tb:te, 1] = lin_y
    #         dom[tb:te, 2] = z
    # print(dom.shape)
    # np.meshgrid(lin_x, lin_y, lin_z)
    fig = plt.figure(figsize=(2, 1))
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(dom[:, 0], dom[:, 1], dom[:, 2], s=0.005, facecolor='black')
    # ax.scatter(dom2[:, 0], dom2[:, 1], dom2[:, 2], s=0.5, facecolor='black')
    # ax.set_xlabel('X', fontsize=3)
    # ax.set_ylabel('Y', fontsize=3)
    # ax.set_zlabel('Z', fontsize=3)
    ax.tick_params(labelsize=4)
    ax.view_init(elev=120., azim=-90)
    # ------------------------------------ BOUNDARY ----------------------------------------
    # Bottom boundary condition (Dirichlet BC 1)
    bcb_u_pts_idx = np.where(dom[:, 2] == z_min)
    bcb_u_pts = dom[bcb_u_pts_idx, :][0]
    bcb_u = np.ones(np.shape(bcb_u_pts)) * [known_bt_ux, known_bt_uy, known_bt_uz]

    # Front Top boundary condition (Neumann BC 1)
    bcfr_t_pts_idx = np.where((dom[:, 2] >= (z_max_2-1)) & (dom[:, 1] == 0) & (dom[:, 0] >= 0.5))
    bcfr_t_pts = dom[bcfr_t_pts_idx, :][0]
    bcfr_t = np.ones(np.shape(bcfr_t_pts)) * [known_fr_tx, known_fr_ty, known_fr_tz]

    # Back Top boundary condition (Neumann BC 1)
    bcbk_t_pts_idx = np.where((dom[:, 2] >= (z_max_2-1)) & (dom[:, 1] == 1) & (dom[:, 0] <= 0.5))
    bcbk_t_pts = dom[bcbk_t_pts_idx, :][0]
    bcbk_t = np.ones(np.shape(bcbk_t_pts)) * [known_bk_tx, known_bk_ty, known_bk_tz]

    ax.scatter(dom[:, 0], dom[:, 1], dom[:, 2], s=0.005, facecolor='blue')
    ax.scatter(bcb_u_pts[:, 0], bcb_u_pts[:, 1], bcb_u_pts[:, 2], s=0.5, facecolor='black')
    ax.scatter(bcfr_t_pts[:, 0], bcfr_t_pts[:, 1], bcfr_t_pts[:, 2], s=0.5, facecolor='red')
    ax.scatter(bcbk_t_pts[:, 0], bcbk_t_pts[:, 1], bcbk_t_pts[:, 2], s=0.5, facecolor='green')
    plt.show()

    boundary_neumann = {
        # condition on the front side
        "neumann_1": {
            "coord": bcfr_t_pts,
            "known_value": bcfr_t,
            "penalty": bc_fr_penalty
        },
        # condition on the back side
        "neumann_2": {
            "coord": bcbk_t_pts,
            "known_value": bcbk_t,
            "penalty": bc_bk_penalty
        }
        # adding more boundary condition here ...
    }
    boundary_dirichlet = {
        # condition on the bottom
        "dirichlet_1": {
            "coord": bcb_u_pts,
            "known_value": bcb_u,
            "penalty": bc_bt_penalty
        }
        # adding more boundary condition here ...
    }
    return dom, dom1, dom2, boundary_neumann, boundary_dirichlet


# -----------------------------------------------------------------------------------------------------
# prepare inputs for testing the model
# -----------------------------------------------------------------------------------------------------
def get_datatest():
    x_dom_test = x_min, x_max, Nx
    y_dom_test = y_min, y_max, Ny
    z_dom_test = z_min, z_max, Nz
    # create points
    x_space = np.linspace(x_dom_test[0], x_dom_test[1], x_dom_test[2])
    y_space = np.linspace(y_dom_test[0], y_dom_test[1], y_dom_test[2])
    z_space = np.linspace(z_dom_test[0], z_dom_test[1], z_dom_test[2])
    yGrid, xGrid, zGrid = np.meshgrid(x_space, y_space, z_space)
    data_test = np.concatenate(
        (np.array([xGrid.flatten()]).T, np.array([yGrid.flatten()]).T, np.array([zGrid.flatten()]).T), axis=1)
    return x_space, y_space, z_space, data_test

def get_datatest2():
    lin_x_2 = np.linspace(x_min_2, x_max_2, Nx_2)
    lin_y_2 = np.linspace(y_min_2, y_max_2, Ny_2)
    lin_z_2 = np.linspace(z_min_2, z_max_2, Nz_2)
    yGrid2, xGrid2, zGrid2 = np.meshgrid(lin_x_2, lin_y_2, lin_z_2)
    dom2 = np.concatenate(
        (np.array([xGrid2.flatten()]).T, np.array([yGrid2.flatten()]).T, np.array([zGrid2.flatten()]).T), axis=1)
    return lin_x_2, lin_y_2, lin_z_2, dom2


# ----------------------------------------------------------
# Find intersection between 2 multidimensional arrays quickly
# Solution: https://stackoverflow.com/questions/49303679/intersection-between-two-multi-dimensional-arrays-with-tolerance-numpy-pytho?noredirect=1&lq=1
# ----------------------------------------------------------
def intersect2Arrays(a, b):
    import itertools
    output = np.empty((0, 3))
    for i0, i1 in itertools.product(np.arange(a.shape[0]), np.arange(b.shape[0])):
        print(i0, i1)
        if np.all(np.isclose(a[i0], b[i1], atol=1e-10)):
            output = np.concatenate((output, [b[i1]]), axis=0)
    return output


def intersect2Arrays2(a, b):
    ax = a[:, 0]
    ay = a[:, 1]
    az = a[:, 2]
    bx = b[:, 0]
    by = b[:, 1]
    bz = b[:, 2]
    xInts = np.in1d(ax, bx)
    yInts = np.in1d(ay, by)
    zInts = np.in1d(az, bz)
    idx = np.where(np.where(xInts == yInts) == zInts)
    output = np.concatenate((ax[idx], ay[idx], az[idx]), axis=1)
    return output


if __name__ == '__main__':
    setup_domain()
