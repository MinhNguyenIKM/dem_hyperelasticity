# -*- coding: utf-8 -*-
"""
@author: sfmt4368 (Simon), texa5140 (Cosmin), minh.nguyen@ikm.uni-hannover.de

Implements the 1D Bar nonlinear
"""

import torch
from torch.autograd import grad
import numpy as np
import numpy.random as npr
from matplotlib import cm
from matplotlib import pyplot as plt
import matplotlib as mpl
import time
from mpl_toolkits.mplot3d import Axes3D
from pyevtk.hl import gridToVTK
import scipy as sp
# from graphviz import Digraph
import torch
from torch.autograd import Variable
# make_dot was moved to https://github.com/szagoruyko/pytorchviz
# from torchviz import make_dot

dev = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("CUDA not available, running on CPU")
mpl.rcParams['figure.dpi'] = 100
# fix random seeds
axes = {'labelsize' : 'large'}
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 17}
legend = {'fontsize': 'medium'}
lines = {'linewidth': 3,
         'markersize' : 7}
mpl.rc('font', **font)
mpl.rc('axes', **axes)
mpl.rc('legend', **legend)
mpl.rc('lines', **lines)
npr.seed(2019)
torch.manual_seed(2019)

# ANALYTICAL SOLUTION
# exact = lambda X: 1. / 135. * X * (3 * X ** 4 - 40 * X ** 2 + 105)
exactU = lambda X: 1/135 * (68 + 105*X - 40*X**3 + 3*X**5)
exactStrain = lambda x: 1./9. * (x**4 - 8*x**2 + 7)
exactEnergy = lambda eps: (1+eps)**(3/2) - 3/2*eps - 1

# ------------------------------ network settings ---------------------------------------------------
iteration = 30
D_in = 1
H = 10
D_out = 1
learning_rate = 1.0
# ------------------------------ material parameter -------------------------------------------------
model_energy = 'Bar1D'
# ----------------------------- define structural parameters ---------------------------------------
Length = 1.0
Height = 1.0
Depth = 1.0
known_left_ux = 0
bc_left_penalty = 1.0

known_right_tx = 0
bc_right_penalty = 1.0
# ------------------------------ define domain and collocation points -------------------------------
Nx = 1000  # 120  # 120
x_min = -1
h = (Length - x_min) / (Nx-1)
# ------------------------------ data testing -------------------------------------------------------
num_test_x = 100
# ------------------------------ filename output ----------------------------------------------------
# ------------------------------ filename output ----------------------------------------------------


# -------------------------------------------------------------------------------
# Purpose: setting domain and collect database
# -------------------------------------------------------------------------------
def setup_domain():
    x_dom = x_min, Length, Nx
    # create points
    dom = np.array([np.linspace(x_dom[0], x_dom[1], x_dom[2])]).T
    return dom


# -----------------------------------------------------------------------------------------------------
# prepare inputs for testing the model
# -----------------------------------------------------------------------------------------------------
def get_datatest():
    x_dom_test = x_min, Length, num_test_x
    # create points
    x_space = np.sort(np.random.uniform(x_min, Length, size=(num_test_x, 1)), axis=0)
    # x_space = np.array([np.linspace(x_dom_test[0], x_dom_test[1], x_dom_test[2])]).T
    return x_space


# --------------------------------------------------------------------------------------------
#               MATERIAL CLASS
# --------------------------------------------------------------------------------------------
class MaterialModel:
    # ---------------------------------------------------------------------------------------
    # Purpose: Construction method
    # ---------------------------------------------------------------------------------------
    def __init__(self):
        print("Material setup !")

    def getEnergyBar1D(self, u, x):
        dudx = grad(u, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        energy = (1 + dudx) ** (3/2) - 3/2*dudx - 1
        return energy

    def getStrongForm(self, u, x):
        dudx = grad(u, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        dWdE = 3/2 * ((1 + dudx)**0.5 - 1)
        strong = grad(dWdE, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0] + x
        return strong


# --------------------------------------------------------------------------
#           NEURAL NETWORK CLASS
# --------------------------------------------------------------------------
class MultiLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultiLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        torch.nn.init.constant_(self.linear1.bias, 0.)
        torch.nn.init.constant_(self.linear2.bias, 0.)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        y1 = torch.tanh(self.linear1(x))
        y = self.linear2(y1)
        return y


# --------------------------------------------------------------------------------
#       MAIN CLASS: Deep Energy Method
# --------------------------------------------------------------------------------
class DeepEnergyMethod:
    # Instance attributes
    def __init__(self, model):
        self.model = MultiLayerNet(model[0], model[1], model[2])
        self.model = self.model.to(dev)

    # ------------------------------------------------------------------
    # Purpose: training model
    # ------------------------------------------------------------------
    def train_model(self, data, material_model, iteration, type_energy, integration):
        x = torch.from_numpy(data).float()
        x = x.to(dev)
        x.requires_grad_(True)
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate, max_iter=20)
        start_time = time.time()
        energy_loss_array = []
        boundary_loss_array = []
        loss_array = []
        it_time = time.time()
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            def closure():
                # https://pytorch.org/docs/stable/optim.html, The closure should clear the gradients, compute the loss, and return it.
                # ----------------------------------------------------------------------------------
                # Internal Energy
                # ----------------------------------------------------------------------------------
                u_pred = self.getU(x)
                u_pred.double()
                # Strain energy equations = Internal Energy
                if type_energy == "Bar1D":
                    potential_energy = material_model.getEnergyBar1D(u_pred, x)
                else:
                    print("Error: Please specify type model !!!")
                    exit()
                f1 = u_pred*x
                if integration == 'montecarlo':
                    dom_crit = (Length-x_min) * self.loss_sum(potential_energy) - (Length-x_min) * self.loss_sum(f1)
                elif integration == 'trapezoidal':
                    dom_crit = self.trapz1D(potential_energy, x=x) - self.trapz1D(f1, x=x)
                else:
                    dom_crit = self.simps1D(potential_energy, x=x) - self.simps1D(f1, x=x)
                # ----------------------------------------------------------------------------------
                # Compute and print loss
                # ----------------------------------------------------------------------------------
                energy_loss = dom_crit
                boundary_loss = torch.tensor([0])
                loss = energy_loss
                optimizer.zero_grad()
                loss.backward()
                print('Iter: %d Loss: %.9e Energy: %.9e Boundary: %.9e Time: %.3e'
                      % (t + 1, loss.item(), energy_loss.item(), boundary_loss.item(), time.time() - it_time))
                energy_loss_array.append(energy_loss.data)
                boundary_loss_array.append(boundary_loss.data)
                loss_array.append(loss.data)
                return loss
            optimizer.step(closure)
        elapsedDEM = time.time() - start_time
        print('Training time: %.4f' % elapsedDEM)

    def getU(self, x):
        u = self.model(x)
        return (x + 1) * u

    def simps(self, y, x=None, dx=1, axis=-1, even='avg'):
        nd = len(y.shape)
        N = y.shape[axis]
        last_dx = dx
        first_dx = dx
        returnshape = 0
        if x is not None:
            if len(x.shape) == 1:
                shapex = [1] * nd
                shapex[axis] = x.shape[0]
                saveshape = x.shape
                returnshape = 1
                x = x.reshape(tuple(shapex))
            elif len(x.shape) != len(y.shape):
                raise ValueError("If given, shape of x must be 1-d or the "
                                 "same as y.")
            if x.shape[axis] != N:
                raise ValueError("If given, length of x along axis must be the "
                                 "same as y.")
        if N % 2 == 0:
            val = 0.0
            result = 0.0
            slice1 = (slice(None),) * nd
            slice2 = (slice(None),) * nd
            if even not in ['avg', 'last', 'first']:
                raise ValueError("Parameter 'even' must be "
                                 "'avg', 'last', or 'first'.")
            # Compute using Simpson's rule on first intervals
            if even in ['avg', 'first']:
                slice1 = self.tupleset(slice1, axis, -1)
                slice2 = self.tupleset(slice2, axis, -2)
                if x is not None:
                    last_dx = x[slice1] - x[slice2]
                val += 0.5 * last_dx * (y[slice1] + y[slice2])
                result = self._basic_simps(y, 0, N - 3, x, dx, axis)
            # Compute using Simpson's rule on last set of intervals
            if even in ['avg', 'last']:
                slice1 = self.tupleset(slice1, axis, 0)
                slice2 = self.tupleset(slice2, axis, 1)
                if x is not None:
                    first_dx = x[tuple(slice2)] - x[tuple(slice1)]
                val += 0.5 * first_dx * (y[slice2] + y[slice1])
                result += self._basic_simps(y, 1, N - 2, x, dx, axis)
            if even == 'avg':
                val /= 2.0
                result /= 2.0
            result = result + val
        else:
            result = self._basic_simps(y, 0, N - 2, x, dx, axis)
        if returnshape:
            x = x.reshape(saveshape)
        return result

    def tupleset(self, t, i, value):
        l = list(t)
        l[i] = value
        return tuple(l)

    def _basic_simps(self, y, start, stop, x, dx, axis):
        nd = len(y.shape)
        if start is None:
            start = 0
        step = 2
        slice_all = (slice(None),) * nd
        slice0 = self.tupleset(slice_all, axis, slice(start, stop, step))
        slice1 = self.tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
        slice2 = self.tupleset(slice_all, axis, slice(start + 2, stop + 2, step))

        if x is None:  # Even spaced Simpson's rule.
            result = torch.sum(dx / 3.0 * (y[slice0] + 4 * y[slice1] + y[slice2]), axis)
        else:
            # Account for possibly different spacings.
            #    Simpson's rule changes a bit.
            # h = np.diff(x, axis=axis)
            h = self.torch_diff_axis_0(x, axis=axis)
            sl0 = self.tupleset(slice_all, axis, slice(start, stop, step))
            sl1 = self.tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
            h0 = h[sl0]
            h1 = h[sl1]
            hsum = h0 + h1
            hprod = h0 * h1
            h0divh1 = h0 / h1
            tmp = hsum / 6.0 * (y[slice0] * (2 - 1.0 / h0divh1) +
                                y[slice1] * hsum * hsum / hprod +
                                y[slice2] * (2 - h0divh1))
            result = torch.sum(tmp, dim=axis)
        return result

    def torch_diff_axis_0(self, a, axis):
        if axis == 0:
            return a[1:, 0:1] - a[:-1, 0:1]
        elif axis == -1:
            return a[1:] - a[:-1]
        else:
            print("Not implemented yet !!! function: torch_diff_axis_0 error !!!")
            exit()

    def simps1D(self, f, x=None, dx=1.0, axis=-1):
        f1D = f.flatten()
        if x is not None:
            x1D = x.flatten()
            return self.simps(f1D, x1D, dx=dx, axis=axis)
        else:
            return self.simps(f1D, dx=dx, axis=axis)

    def __trapz(self, y, x=None, dx=1.0, axis=-1):
        # y = np.asanyarray(y)
        if x is None:
            d = dx
        else:
            d = x[1:] - x[0:-1]
            # reshape to correct shape
            shape = [1] * y.ndimension()
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        nd = y.ndimension()
        slice1 = [slice(None)] * nd
        slice2 = [slice(None)] * nd
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)
        ret = torch.sum(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis)
        return ret

    def trapz1D(self, y, x=None, dx=1.0, axis=-1):
        y1D = y.flatten()
        if x is not None:
            x1D = x.flatten()
            return self.__trapz(y1D, x1D, dx=dx, axis=axis)
        else:
            return self.__trapz(y1D, dx=dx)

    def get_gradu(self, u, x):
        dudx = grad(u, x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        return dudx

    def evaluate_model(self, x_space):
        t_tensor = torch.from_numpy(x_space).float()
        t_tensor = t_tensor.to(dev)
        t_tensor.requires_grad_(True)
        u_pred_torch = self.getU(t_tensor)
        u_pred = u_pred_torch.detach().cpu().numpy()
        dudx_torch = grad(u_pred_torch, t_tensor, torch.ones(t_tensor.shape[0], 1, device=dev))[0]
        dudx = dudx_torch.detach().cpu().numpy()
        return u_pred, dudx

    # --------------------------------------------------------------------------------
    # Purpose: loss sum for the energy part
    # --------------------------------------------------------------------------------
    @staticmethod
    def loss_sum(tinput):
        return torch.sum(tinput) / tinput.data.nelement()

    # --------------------------------------------------------------------------------
    # purpose: loss square sum for the boundary part
    # --------------------------------------------------------------------------------
    @staticmethod
    def loss_squared_sum(tinput, target):
        row, column = tinput.shape
        loss = 0
        for j in range(column):
            loss += torch.sum((tinput[:, j] - target[:, j]) ** 2) / tinput[:, j].data.nelement()
        return loss


# --------------------------------------------------------------------------------
# purpose: doing something in post processing for visualization
# --------------------------------------------------------------------------------
def write_vtk(filename, x_space, y_space, z_space, Ux, Uy, Uz):
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    displacement = (Ux, Uy, Uz)
    gridToVTK(filename, xx, yy, zz, pointData={"displacement": displacement})


# ----------------------------------------------------------------------
#                   EXECUTE PROGRAMME
# ----------------------------------------------------------------------
if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(10, 10))
    fig2, bx = plt.subplots(figsize=(10, 10))
    fig3, cx = plt.subplots(figsize=(10, 10))
    fig4, dx = plt.subplots(figsize=(10, 10))
    # ----------------------------------------------------------------------
    #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
    # ----------------------------------------------------------------------
    dom = setup_domain()
    x_predict = get_datatest()
    exact_solution = exactU(x_predict)
    exact_eps = exactStrain(x_predict)
    # ----------------------------------------------------------------------
    #                   STEP 2.1: SETUP DEM MODEL
    # ----------------------------------------------------------------------
    mat = MaterialModel()
    dems = DeepEnergyMethod([D_in, H, D_out])
    time_dems = time.time()
    dems.train_model(dom, mat, iteration, model_energy, 'simpson')
    time_dems = time.time() - time_dems
    u_pred_dems, eps_pred_dems = dems.evaluate_model(x_predict)
    error_L2_DEMS = np.linalg.norm(exact_solution - u_pred_dems, 2) / np.linalg.norm(exact_solution, 2)
    error_H1_DEMS = np.linalg.norm(exact_eps - eps_pred_dems, 2) / np.linalg.norm(exact_eps, 2)

    demt = DeepEnergyMethod([D_in, H, D_out])
    time_demt = time.time()
    demt.train_model(dom, mat, iteration, model_energy, 'trapezoidal')
    time_demt = time.time() - time_demt
    u_pred_demt, eps_pred_demt = demt.evaluate_model(x_predict)
    error_L2_DEMT = np.linalg.norm(exact_solution - u_pred_demt, 2) / np.linalg.norm(exact_solution, 2)
    error_H1_DEMT = np.linalg.norm(exact_eps - eps_pred_demt, 2) / np.linalg.norm(exact_eps, 2)

    demm = DeepEnergyMethod([D_in, H, D_out])
    time_demm = time.time()
    demm.train_model(dom, mat, iteration, model_energy, 'montecarlo')
    time_demm = time.time() - time_demm
    u_pred_demm, eps_pred_demm = demm.evaluate_model(x_predict)
    error_L2_DEMM = np.linalg.norm(exact_solution - u_pred_demm, 2) / np.linalg.norm(exact_solution, 2)
    error_H1_DEMM = np.linalg.norm(exact_eps - eps_pred_demm, 2) / np.linalg.norm(exact_eps, 2)

    ax.plot(x_predict, exact_solution, label="Exact", linestyle='dashed', color='black')
    ax.plot(x_predict, u_pred_dems, 'rx', label='DEMS')
    ax.plot(x_predict, u_pred_demt, 'g+', label='DEMT')
    ax.plot(x_predict, u_pred_demm, 'b2', label='DEMM')
    ax.legend(ncol=4, fancybox=True, shadow=True, loc='upper left', bbox_to_anchor=(0.0, 1.14))
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('u(X)')
    bx.plot(x_predict, exactStrain(x_predict), label='Exact', linestyle='dashed', color='black')
    bx.plot(x_predict, eps_pred_dems, 'rx', label='DEMS')
    bx.plot(x_predict, eps_pred_demt, 'g+', label='DEMT')
    bx.plot(x_predict, eps_pred_demm, 'b2', label='DEMM')
    legend = bx.legend(loc='upper left', shadow=True)
    bx.legend(ncol=4, fancybox=True, shadow=True, loc='upper left', bbox_to_anchor=(0.0, 1.14))
    bx.grid(True)
    bx.set_xlabel('X')
    bx.set_ylabel('du(X)/dX')
    cx.semilogy(x_predict, np.abs(exact_solution - u_pred_dems), 'r-.', label=r"$|u_{Exact} - u_{{DEMS}}|$")
    cx.semilogy(x_predict, np.abs(exact_solution - u_pred_demt), 'g-.', label='$|u_{Exact} - u_{DEMT}|$')
    cx.semilogy(x_predict, np.abs(exact_solution - u_pred_demm), 'b-.', label='$|u_{Exact} - u_{DEMM}|$')
    cx.grid(True)
    cx.set_xlabel('X')
    cx.set_ylabel('error')
    cx.legend(ncol=3, fancybox=True, shadow=True, loc='upper left', bbox_to_anchor=(-0.03, 1.14))
    dx.semilogy(x_predict, np.abs(exact_eps - eps_pred_dems), 'r-.', label='$|\epsilon_{Exact} - \epsilon_{DEMS}|$')
    dx.semilogy(x_predict, np.abs(exact_eps - eps_pred_demt), 'g-.', label='$|\epsilon_{Exact} - \epsilon_{DEMT}|$')
    dx.semilogy(x_predict, np.abs(exact_eps - eps_pred_demm), 'b-.', label='$|\epsilon_{Exact} - \epsilon_{DEMM}|$')
    dx.grid(True)
    dx.set_xlabel('X')
    dx.set_ylabel('error')
    dx.legend(ncol=3, fancybox=True, shadow=True, loc='upper left', bbox_to_anchor=(-0.03, 1.14))
    print("DEMS ||e||L2 : %.2e" % error_L2_DEMS)
    print("DEMT ||e||L2 : %.2e" % error_L2_DEMT)
    print("DEMM ||e||L2 : %.2e" % error_L2_DEMM)
    print("DEMS ||e||H1 : %.2e" % error_H1_DEMS)
    print("DEMT ||e||H1 : %.2e" % error_H1_DEMT)
    print("DEMM ||e||H1 : %.2e" % error_H1_DEMM)
    print("DEMS time : %.2f" % time_dems)
    print("DEMT time : %.2f" % time_demt)
    print("DEMM time : %.2f" % time_demm)
    plt.show()
