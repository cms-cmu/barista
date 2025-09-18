import collections
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from coffea4bees.classifier.algorithm.statistics.variance import TensorVariance
from rich import box as BoxStyle
from rich.table import Table


### newly added features (needs to move to base class):
# transvserse mass of two two-dimensional vectors
def transverse_mass(v1, v2):
    pt_idx1 = 0 if (v1.shape[-1] == 2) else 1 # works with both 2d and 4d vectors
    pt_idx2 = 0 if (v1.shape[-1] == 2) else 1
    mT = np.sqrt(2*v1[:, pt_idx1:pt_idx1+1] * v2[:, pt_idx2:pt_idx2+1]*(1-np.cos(calcDeltaPhi(v1, v2))))
    return mT
            
            
class layerOrganizer:
    def __init__(self):
        self.layers = collections.OrderedDict()
        self.nTrainableParameters = 0

    def addLayer(self, newLayer, inputLayers=None, startIndex=1):
        if inputLayers:
            try:
                inputIndicies = inputLayers  # [layer.index for layer in inputLayers]
                newLayer.index = max(inputIndicies) + 1
            except TypeError:
                inputIndicies = [layer.index for layer in inputLayers]
                newLayer.index = max(inputIndicies) + 1
        else:
            try:
                newLayer.index = startIndex.index
            except:
                newLayer.index = startIndex

        try:
            self.layers[newLayer.index].append(newLayer)
        except (KeyError, AttributeError):
            self.layers[newLayer.index] = [newLayer]

    def countTrainableParameters(self):
        self.nTrainableParameters = 0
        for index in self.layers:
            for layer in self.layers[index]:
                for param in layer.parameters():
                    self.nTrainableParameters += (
                        param.numel() if param.requires_grad else 0
                    )

    def setLayerRequiresGrad(self, index=None, requires_grad=True, debug=False):
        if debug:
            self.countTrainableParameters()
            print(
                "\nChange trainable parameters from", self.nTrainableParameters, end=" "
            )
        try:  # treat index as list of indices
            if index is None:  # apply to all layers
                index = self.layers.keys()
            if index == -1:
                index = [sorted(self.layers.keys())[-1]]
            for i in index:
                for layer in self.layers[i]:
                    for param in layer.parameters():
                        param.requires_grad = requires_grad
        except TypeError:  # index is just an int
            for layer in self.layers[index]:
                for param in layer.parameters():
                    param.requires_grad = requires_grad
        if debug:
            self.countTrainableParameters()
            print("to", self.nTrainableParameters)

    def initLayer(self, index):
        try:  # treat index as list of indices
            print("Rerandomize layer indicies", index)
            for i in index:
                for layer in self.layers[i]:
                    layer.randomize()
        except TypeError:  # index is just an int
            print("Rerandomize layer index", index)
            for layer in self.layers[index]:
                layer.randomize()

    def computeStats(self):
        for index in self.layers:
            for layer in self.layers[index]:
                layer.gradStats.compute()

    def resetStats(self):
        for index in self.layers:
            for layer in self.layers[index]:
                layer.gradStats.reset()

    def print(self, batchNorm=False):
        for index in self.layers:
            width = len(self.layers[index])
            if index > 1:
                width = max(width, len(self.layers[index - 1]))
            print(
                " %s Layer %2d %s" % ("-" * 20, index, "-" * (50 * width - 31 + width))
            )
            for layer in self.layers[index]:
                print("|", layer.name.ljust(49), end="")
            print("|")
            if batchNorm:
                for layer in self.layers[index]:
                    if layer.batchNorm:
                        layer.batchNorm.print()
                    else:
                        print("|", " " * 49, end="")
                print("")
        print(" %s" % ("-" * (50 * width)))
        # for layer in self.layers[index]:
        #     if layer.gradStats:
        #         print('|',layer.gradStats.summary.ljust(50), end='')
        #     else:
        #         print('|',' '*50, end='')
        # print('')
        # for layer in self.layers[index]:
        #     print('|',str(layer.module).ljust(45), end=' ')
        # print('|')
def vec2str(v, f="{:.4g}"):
    return map(f.format, v)

# some basic four-vector operations
def PxPyPzE(v):  # need this to be able to add four-vectors
    pt = v[:, 0:1]
    eta = v[:, 1:2]
    phi = v[:, 2:3]
    m = v[:, 3:4]

    Px, Py, Pz = (
        pt * phi.cos(),
        pt * phi.sin(),
        pt * sinh(eta),
    )  # sinh unsupported by ONNX
    E = (pt**2 + Pz**2 + m**2).sqrt()

    return torch.cat((Px, Py, Pz, E), 1)


def PxPyPzM(v):
    pt = v[:, 0:1]
    eta = v[:, 1:2]
    phi = v[:, 2:3]
    m = v[:, 3:4]

    Px, Py, Pz = (
        pt * phi.cos(),
        pt * phi.sin(),
        pt * sinh(eta),
    )  # sinh unsupported by ONNX
    # E = (pt**2 + Pz**2 + m**2).sqrt()

    return torch.cat((Px, Py, Pz, m), 1)


def PtEtaPhiM(v):
    px = v[:, 0:1]
    py = v[:, 1:2]
    pz = v[:, 2:3]
    e = v[:, 3:4]

    Pt = (px**2 + py**2).sqrt()
    ysign = py.sign()
    ysign = (
        ysign + (ysign == 0.0).float()
    )  # if py==0, px==Pt and acos(1)=pi/2 so we need zero protection on py.sign()
    Phi = (px / (Pt + 0.00001)).acos() * ysign
    # try:
    #     Eta = (pz/(Pt+0.00001)).asinh()
    # except:
    Eta = asinh(pz / (Pt + 0.00001))  # asinh not supported by ONNX

    M = F.relu(e**2 - px**2 - py**2 - pz**2).sqrt()

    return torch.cat((Pt, Eta, Phi, M), 1)

## New
def calcDeltaPhi(v1, v2):  # expects eta, phi representation
    phi_idx1 = 1 if (v1.shape[-1] == 2) else 2 # works with both 2d and 4d vectors
    phi_idx2 = 1 if (v1.shape[-1] == 2) else 2 

    dPhi12 = (v1[:, phi_idx1:phi_idx1+1] - v2[:, phi_idx2:phi_idx2+1]) % math.tau
    dPhi21 = (v2[:, phi_idx2:phi_idx2+1] - v1[:, phi_idx1:phi_idx1+1]) % math.tau
    dPhi = torch.min(dPhi12, dPhi21)
    return dPhi


def calcDeltaR(v1, v2):  # expects eta, phi representation
    dPhi = calcDeltaPhi(v1, v2)
    dR = ((v1[:, 1:2] - v2[:, 1:2]) ** 2 + dPhi**2).sqrt()
    return dR


def addFourVectors(
    v1, v2, v1PxPyPzE=None, v2PxPyPzE=None
):  # output added four-vectors in pt,eta,phi,m coordinates and opening angle between constituents
    # vX[batch index, (pt,eta,phi,m), object index]
    # dR  = calcDeltaR(v1, v2)

    if v1PxPyPzE is None:
        v1PxPyPzE = PxPyPzE(v1)
    if v2PxPyPzE is None:
        v2PxPyPzE = PxPyPzE(v2)

    v12PxPyPzE = v1PxPyPzE + v2PxPyPzE
    v12 = PtEtaPhiM(v12PxPyPzE)

    return v12, v12PxPyPzE

def matrixMdR(
    v1, v2, v1PxPyPzE=None, v2PxPyPzE=None
):  # output matrix M.shape = (batch size, 2, n v1 objects, m v2 objects)
    if v1PxPyPzE is None:
        v1PxPyPzE = PxPyPzE(v1)
    if v2PxPyPzE is None:
        v2PxPyPzE = PxPyPzE(v2)

    b = v1PxPyPzE.shape[0]
    n, m = v1PxPyPzE.shape[2], v2PxPyPzE.shape[2]

    # use PxPyPzE representation to compute M
    v1PxPyPzE = v1PxPyPzE.view(b, 4, n, 1)
    v2PxPyPzE = v2PxPyPzE.view(b, 4, 1, m)

    M = diObjectMass(v1PxPyPzE, v2PxPyPzE)
    M = torch.log(1 + M)

    # use PtEtaPhiM representation to compute dR
    v1 = v1.view(b, -1, n, 1)
    v2 = v2.view(b, -1, 1, m)

    dR = calcDeltaR(v1, v2)
    return torch.cat( (M, dR), 1 )
    #dPhi = calcDeltaPhi(v1, v2)
    #return torch.cat((M, dPhi), 1)

# pytorch 0.4 does not have inverse hyperbolic trig functions
def asinh(x):
    xsign = x.sign()
    xunsigned = x * xsign
    loggand = (
        xunsigned + (xunsigned.pow(2) + 1).sqrt()
    )  # numerically unstable if you do x+(x.pow(2)+1).sqrt() because is very close to zero when x is very negative
    return (
        torch.log(loggand) * xsign
    )  # if x is zero then asinh(x) is also zero so don't need zero protection on xsign


def acosh(x):
    return torch.log(x + (x**2 - 1).sqrt())


def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


def sinh(x):
    return (torch.exp(x) - torch.exp(-x)) / 2.0

class stats:
    def __init__(self):
        self.grad = collections.OrderedDict()
        self.mean = collections.OrderedDict()
        self.std = collections.OrderedDict()
        self.summary = ""

    def update(self, attr, grad):
        try:
            self.grad[attr] = torch.cat((self.grad[attr], grad), dim=0)
        except (KeyError, TypeError):
            self.grad[attr] = grad.clone()

    def compute(self):
        self.summary = ""
        self.grad["combined"] = None
        for attr, grad in self.grad.items():
            try:
                self.grad["combined"] = torch.cat((self.grad["combined"], grad), dim=1)
            except TypeError:
                self.grad["combined"] = grad.clone()

            self.mean[attr] = grad.mean(dim=0).norm()
            self.std[attr] = grad.std()
            # self.summary += attr+': <%1.1E> +/- %1.1E r=%1.1E'%(self.mean[attr],self.std[attr],self.mean[attr]/self.std[attr])
        self.summary = "grad: <%1.1E> +/- %1.1E SNR=%1.1f" % (
            self.mean["combined"],
            self.std["combined"],
            (self.mean["combined"] / self.std["combined"]).log10(),
        )

    def dump(self):
        for attr, grad in self.grad.items():
            print(attr, grad.shape, grad.mean(dim=0).norm(2), grad.std())

    def reset(self):
        for attr in self.grad:
            self.grad[attr] = None

def make_hook(gradStats, module, attr):
    def hook(grad):
        gradStats.update(attr, grad / getattr(module, attr).norm(2))

    return hook

def deltaM(v1, v2):
    # PxPyPzE1 = PxPyPzE(v1)
    # PxPyPzE2 = PxPyPzE(v2)
    # print('x')
    # print(PxPyPzE1[0])
    # print('x_reg')
    # print(PxPyPzE2[0])
    delta = v1 - v2  # PxPyPzE(v1) - PxPyPzE(v2)
    # print('delta')
    # print(delta[0])
    M = (
        (delta[:, 3] ** 2 - delta[:, 0] ** 2 - delta[:, 1] ** 2 - delta[:, 2] ** 2)
        .abs()
        .sqrt()
    )
    return M


def ReLU(x):
    return F.relu(x)


def CReLU(x, dim=1):
    x = torch.cat((x, -x), dim)
    return F.relu(x)


def LeLU(x):
    return F.leaky_relu(x, 0.5)


def SiLU(
    x,
):  # SiLU https://arxiv.org/pdf/1702.03118.pdf   Swish https://arxiv.org/pdf/1710.05941.pdf
    return x * torch.sigmoid(x)

def isinf(x):
    return (x == float("inf")) | (x == float("-inf")) | (x == float("nan"))

# def NonLU(x, training=False): # Non-Linear Unit
def NonLU(x):  # Non-Linear Unit
    # return ReLU(x)
    # return F.rrelu(x, training=training)
    # return F.leaky_relu(x, negative_slope=0.1)
    return SiLU(x)
    # return F.elu(x)

def diObjectMass(v1PxPyPzE, v2PxPyPzE):
    v12PxPyPzE = v1PxPyPzE + v2PxPyPzE
    M = F.relu(
        v12PxPyPzE[:, 3:4] ** 2
        - v12PxPyPzE[:, 0:1] ** 2
        - v12PxPyPzE[:, 1:2] ** 2
        - v12PxPyPzE[:, 2:3] ** 2
    ).sqrt()
    # precision issues can in rare cases causes a negative value in above ReLU argument. Replace these with zero using ReLU before sqrt
    return M

class GhostBatchNorm1d(
    nn.Module
):  # https://arxiv.org/pdf/1705.08741v2.pdf has what seem like typos in GBN definition.
    def __init__(
        self,
        features,
        ghost_batch_size=32,
        number_of_ghost_batches=64,
        nAveraging=1,
        stride=1,
        eta=0.9,
        bias=True,
        device="cuda",
        name="",
        conv=False,
        features_out=None,
        phase_symmetric=False,
        PCC=False,
    ):
        super(GhostBatchNorm1d, self).__init__()
        self.name = name
        self.index = None
        self.stride = stride
        self.device = device
        self.features = features
        self.features_out = features_out if features_out is not None else self.features
        self.register_buffer(
            "ghost_batch_size", torch.tensor(ghost_batch_size, dtype=torch.long)
        )
        self.register_buffer(
            "nGhostBatches",
            torch.tensor(number_of_ghost_batches * nAveraging, dtype=torch.long),
        )
        self.conv = False
        self.gamma = None
        self.bias = None
        self.PCC = PCC
        self.noPullCount = 0
        self.updates = 0
        if conv:
            self.conv = conv1d(
                self.features,
                self.features_out,
                self.stride,
                self.stride,
                name="%s conv" % name,
                bias=bias,
                phase_symmetric=phase_symmetric,
                PCC=self.PCC,
            )
        else:
            self.gamma = nn.Parameter(torch.ones(self.features))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.features))
        self._mean_var = TensorVariance()
        self.runningStats = True
        self.initialized = False

        self.register_buffer("eps", torch.tensor(1e-5, dtype=torch.float))
        self.register_buffer("eta", torch.tensor(eta, dtype=torch.float))
        self.register_buffer(
            "m", torch.zeros((1, 1, self.stride, self.features), dtype=torch.float)
        )
        self.register_buffer(
            "s", torch.ones((1, 1, self.stride, self.features), dtype=torch.float)
        )
        self.register_buffer("zero", torch.tensor(0.0, dtype=torch.float))
        self.register_buffer("one", torch.tensor(1.0, dtype=torch.float))
        self.register_buffer("two", torch.tensor(2.0, dtype=torch.float))

    def print(self):
        table = Table(box=BoxStyle.HORIZONTALS, show_header=False)
        for i in range(self.stride):
            table.add_row("mean", *vec2str(self.m[0, 0, i, :]))
        for i in range(self.stride):
            table.add_row("std", *vec2str(self.s[0, 0, i, :]))
        if self.gamma is not None:
            table.add_row("gamma", *vec2str(self.gamma.data))
            if self.bias is not None:
                table.add_row("bias", *vec2str(self.bias.data))
        logging.info(f"Ghost Batch {self.name}", table)

    @torch.no_grad()
    def updateMeanStd(self, x, mask=None):
        batch_size = x.shape[0]
        pixels = x.shape[2]
        x = (
            x.detach()
            .transpose(1, 2)
            .contiguous()
            .view(batch_size * pixels, 1, self.features)
        )
        if mask is not None:
            mask = mask.detach().view(batch_size * pixels)
            x = x[mask == 0, :, :]
        # this won't work for any layers with stride!=1
        x = x.view(-1, 1, self.stride, self.features)
        self._mean_var += TensorVariance(x)

    @torch.no_grad()
    def initMeanStd(self):
        self.m = self._mean_var.mean.to(self.device)
        self.s = self._mean_var.variance_unbiased.sqrt().to(self.device)
        self.initialized = True
        self.runningStats = False
        self.print()

    # @torch.no_grad()
    # def setMeanStd(self, x, mask=None):
    #     batch_size = x.shape[0]
    #     pixels = x.shape[2]
    #     x = (
    #         x.detach()
    #         .transpose(1, 2)
    #         .contiguous()
    #         .view(batch_size * pixels, 1, self.features)
    #     )
    #     if mask is not None:
    #         mask = mask.detach().view(batch_size * pixels)
    #         x = x[mask == 0, :, :]
    #     # this won't work for any layers with stride!=1
    #     x = x.view(-1, 1, self.stride, self.features)
    #     m64 = x.mean(dim=0, keepdim=True, dtype=torch.float64)  # .to(self.device)
    #     self.m = m64.type(torch.float32).to(self.device)
    #     self.s = x.std(dim=0, keepdim=True).to(self.device)
    #     # if x.shape[0]>16777216: # too big for quantile???
    #     self.initialized = True
    #     # self.setGhostBatches(0)
    #     self.runningStats = False
    #     self.print()

    def setGhostBatches(self, nGhostBatches):
        # if nGhostBatches==0 and self.nGhostBatches>0:
        #     print('Set # of ghost batches to zero: %s'%self.name)
        self.nGhostBatches = torch.tensor(nGhostBatches, dtype=torch.long).to(
            self.device
        )

    def forward(self, x, mask=None, debug=False):
        batch_size = x.shape[0]
        pixels = x.shape[2]
        # pixel_groups = pixels//self.stride
        pixel_groups = torch.div(pixels, self.stride, rounding_mode="trunc")

        if self.training and self.nGhostBatches != 0 and not self.PCC:
            self.ghost_batch_size = batch_size // self.nGhostBatches.abs()

            #
            # Apply batch normalization with Ghost Batch statistics
            #
            x = (
                x.transpose(1, 2)
                .contiguous()
                .view(
                    self.nGhostBatches.abs(),
                    self.ghost_batch_size * pixel_groups,
                    self.stride,
                    self.features,
                )
            )

            if mask is None:
                gbm = x.mean(dim=1, keepdim=True)
                gbs = (x.var(dim=1, keepdim=True) + self.eps).sqrt()

            else:
                # Compute masked mean and std for each ghost batch
                mask = mask.view(
                    self.nGhostBatches.abs(),
                    self.ghost_batch_size * pixel_groups,
                    self.stride,
                    1,
                )
                nUnmasked = (
                    (mask == 0).sum(dim=1, keepdim=True).float()
                )  # .to(self.device)
                unmasked0 = (nUnmasked == self.zero).float()  # .to(self.device)
                unmasked1 = (nUnmasked == self.one).float()  # .to(self.device)
                denomMean = nUnmasked + unmasked0  # prevent divide by zero
                denomVar = (
                    nUnmasked + unmasked0 * self.two + unmasked1 - self.one
                )  # prevent divide by zero with bessel correction
                gbm = x.masked_fill(mask, 0).sum(dim=1, keepdim=True) / denomMean
                gbs = (
                    ((x - gbm).masked_fill(mask, 0) ** 2).sum(dim=1, keepdim=True)
                    / denomVar
                    + self.eps
                ).sqrt()

            #
            # Keep track of running mean and standard deviation.
            #
            if self.runningStats or debug:
                # Use mean over ghost batches for running mean and std
                bm = gbm.detach().mean(dim=0, keepdim=True)
                bs = gbs.detach().mean(dim=0, keepdim=True)

                # self.updates += 1
                # if m_pulls.abs().max() < 1 and s_pulls.abs().max() < 1:
                #     self.noPullCount += m_pulls.shape[-1]*2
                # else:
                #     self.noPullCount = 0
                # if self.noPullCount > 10 and self.updates > 100:
                #     print()
                #     self.setGhostBatches(0)
                #     print(m_pulls)
                #     print(s_pulls)
                #     print()

                if debug and self.initialized:
                    gbms = gbm.detach().std(dim=0, keepdim=True)
                    gbss = gbs.detach().std(dim=0, keepdim=True)
                    m_pulls = (bm - self.m) / gbms
                    s_pulls = (bs - self.s) / gbss
                    # s_ratio = bs/self.s
                    # if (m_pulls.abs()>5).any() or (s_pulls.abs()>5).any():
                    print()
                    print(self.name)
                    print("self.m\n", self.m)
                    print("    bm\n", bm)
                    print("  gbms\n", gbms)
                    print(
                        "m_pulls\n", m_pulls, m_pulls.abs().mean(), m_pulls.abs().max()
                    )
                    print("-------------------------")
                    print("self.s\n", self.s)
                    print("    bs\n", bs)
                    print("  gbss\n", gbss)
                    print(
                        "s_pulls\n", s_pulls, s_pulls.abs().mean(), s_pulls.abs().max()
                    )
                    # print('s_ratio\n',s_ratio)
                    print()
                    # input()

            if self.runningStats:
                # Simplest possible method
                if self.initialized:
                    self.m = self.eta * self.m + (self.one - self.eta) * bm
                    self.s = self.eta * self.s + (self.one - self.eta) * bs
                else:
                    self.m = self.zero * self.m + bm
                    self.s = self.zero * self.s + bs
                    self.initialized = True

            if self.nGhostBatches > 0:
                x = x - gbm
                x = x / gbs
            else:
                x = x.view(batch_size, pixel_groups, self.stride, self.features)
                x = x - self.m
                x = x / self.s

        else:
            # Use mean and standard deviation buffers rather than batch statistics
            # .view(self.nGhostBatches, self.ghost_batch_size*pixel_groups, self.stride, self.features)
            x = x.transpose(1, 2).view(
                batch_size, pixel_groups, self.stride, self.features
            )
            x = x - self.m
            x = x / self.s

        if self.conv:
            # back to standard indexing for convolutions: [batch, feature, pixel]
            x = x.view(batch_size, pixels, self.features).transpose(1, 2).contiguous()
            x = self.conv(x)
        else:
            x = x * self.gamma
            if self.bias is not None:
                x = x + self.bias
            # back to standard indexing for convolutions: [batch, feature, pixel]
            x = x.view(batch_size, pixels, self.features).transpose(1, 2).contiguous()
        return x

class conv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=1,
        stride=1,
        bias=True,
        groups=1,
        name="",
        index=None,
        doGradStats=False,
        hiddenIn=False,
        hiddenOut=False,
        batchNorm=False,
        batchNormMomentum=0.9,
        nAveraging=1,
        phase_symmetric=False,
        PCC=False,
    ):
        super(conv1d, self).__init__()
        self.bias = (
            bias and not batchNorm and not phase_symmetric
        )  # if doing batch norm, bias is in BN layer, not convolution
        self.phase_symmetric = phase_symmetric
        self.in_channels = in_channels  # *2 if self.phase_symmetric else in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        if phase_symmetric:
            self.out_channels = self.out_channels // 2
        self.register_buffer("eps", torch.tensor(1e-5, dtype=torch.float))
        self.PCC = PCC
        self.module = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            stride=stride,
            bias=self.bias,
            groups=groups,
        )
        if self.phase_symmetric and bias:
            self.bias = nn.Parameter(torch.zeros(1, self.out_channels * 2, 1))
        if batchNorm:
            self.batchNorm = GhostBatchNorm1d(
                self.out_channels,
                nAveraging=nAveraging,
                eta=batchNormMomentum,
                bias=bias,
                name="%s GBN" % name,
            )  # nn.BatchNorm1d(out_channels)
        else:
            self.batchNorm = False

        # self.hiddenIn=hiddenIn
        # if self.hiddenIn:
        #     self.moduleHiddenIn = nn.Conv1d(in_channels,in_channels,1)
        # self.hiddenOut=hiddenOut
        # if self.hiddenOut:
        #     self.moduleHiddenOut = nn.Conv1d(out_channels,out_channels,1)
        self.name = name
        self.index = index
        self.gradStats = None
        self.k = 1.0 / (in_channels * kernel_size)
        if doGradStats:
            self.gradStats = stats()
            self.module.weight.register_hook(
                make_hook(self.gradStats, self.module, "weight")
            )
            # self.module.bias  .register_hook( make_hook(self.gradStats, self.module, 'bias'  ) )

    def randomize(self):
        # if self.hiddenIn:
        #     nn.init.uniform_(self.moduleHiddenIn.weight, -(self.k**0.5), self.k**0.5)
        #     nn.init.uniform_(self.moduleHiddenIn.bias,   -(self.k**0.5), self.k**0.5)
        # if self.hiddenOut:
        #     nn.outit.uniform_(self.moduleHiddenOut.weight, -(self.k**0.5), self.k**0.5)
        #     nn.outit.uniform_(self.moduleHiddenOut.bias,   -(self.k**0.5), self.k**0.5)
        nn.init.uniform_(self.module.weight, -(self.k**0.5), self.k**0.5)
        if self.bias:
            nn.init.uniform_(self.module.bias, -(self.k**0.5), self.k**0.5)

    def forward(self, x, mask=None, debug=False):
        # if self.hiddenIn:
        #     x = NonLU(self.moduleHiddenIn(x), self.moduleHiddenIn.training)
        # if self.hiddenOut:
        #     x = NonLU(self.module(x), self.module.training)
        #     return self.moduleHiddenOut(x)
        if self.PCC:
            x = x - x.mean(dim=1, keepdim=True)
            x = x / (x.norm(dim=1, keepdim=True) + self.eps)
            self.module.weight.data = (
                self.module.weight.data
                - self.module.weight.data.mean(dim=1, keepdim=True)
            )
        x = self.module(x)
        if self.batchNorm:
            x = self.batchNorm(x, mask, debug)
        if self.phase_symmetric:  # https://arxiv.org/pdf/1603.05201v2.pdf
            x = torch.cat((x, -x), 1)
            if type(self.bias) is nn.Parameter:
                x = x + self.bias
        return x


class linear(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        name=None,
        index=None,
        doGradStats=False,
        bias=True,
    ):
        super(linear, self).__init__()
        self.module = nn.Linear(in_channels, out_channels, bias=bias)
        self.bias = bias
        self.name = name
        self.index = index
        self.gradStats = None
        self.k = 1.0 / in_channels
        if doGradStats:
            self.gradStats = stats()
            self.module.weight.register_hook(
                make_hook(self.gradStats, self.module, "weight")
            )
            # self.module.bias  .register_hook( make_hook(self.gradStats, self.module, 'bias'  ) )

    def randomize(self):
        nn.init.uniform_(self.module.weight, -(self.k**0.5), self.k**0.5)
        if self.bias:
            nn.init.uniform_(self.module.bias, -(self.k**0.5), self.k**0.5)

    def forward(self, x):
        return self.module(x)

class dijetReinforceLayer(nn.Module):
    def __init__(self, dijetFeatures, batchNorm=False, phase_symmetric=False):
        super(dijetReinforceLayer, self).__init__()
        self.dD = dijetFeatures
        self.index = None
        self.name = "jet, jet, dijet convolution"
        # # make fixed convolution to compute average of jet pixel pairs (symmetric bilinear)
        # self.sym = nn.Conv1d(self.dD, self.dD, 2, stride=2, bias=False, groups=self.dD)
        # self.sym.weight.data.fill_(0.5)
        # self.sym.weight.requires_grad = False

        # # make fixed convolution to compute difference of jet pixel pairs (antisymmetric bilinear)
        # self.antisym = nn.Conv1d(self.dD, self.dD, 2, stride=2, bias=False, groups=self.dD)
        # self.antisym.weight.data.fill_(0.5)
        # self.antisym.weight.data[:,:,1] *= -1
        # self.antisym.weight.requires_grad = False

        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|  ##stride=3 kernel=3 reinforce dijet features
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|
        # self.conv = conv1d(self.dD, self.dD, 3, stride=3, name='dijet reinforce convolution', batchNorm=batchNorm)
        self.conv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            stride=3,
            conv=True,
            name=self.name,
            PCC=False,
        )

    def forward(self, j, d):
        d = torch.cat(
            (
                j[:, :, 0:2],
                d[:, :, 0:1]
            ),
            2,
        )
        d = self.conv(d)
        return d
    
class lepWReinforceLayer(nn.Module):
    def __init__(self, features, batchNorm=False, phase_symmetric=False):
        super().__init__()
        self.index = None
        self.features = features
        self.name = "lepton, met convolution"

        self.conv = GhostBatchNorm1d(
            self.features,
            phase_symmetric=phase_symmetric,
            stride=7,
            conv=True,
            name=self.name,
        )

    def forward(self, lepW, x): # use x here to maintain structure of resnet block code
        lepW = self.conv(lepW)
        return lepW

class ResNetBlock(nn.Module):
    def __init__(
        self,
        nFeatures,
        phase_symmetric=True,
        device="cuda",
        layers=None,
        inputLayers=None,
        prefix="",
        nLayers=2,
        xx0Update=True,
    ):
        super(ResNetBlock, self).__init__()
        self.d = nFeatures  # dimension of feature space
        self.device = device
        self.xx0Update = xx0Update
        self.reinforce = []
        self.conv = []
        for i in range(1, nLayers + 1):
            previousLayers = inputLayers + self.reinforce + self.conv

            if (
                i != nLayers
            ):  # we don't output updated x array so we don't need a final x convolution
                self.conv.append(
                    GhostBatchNorm1d(
                        self.d,
                        phase_symmetric=phase_symmetric,
                        conv=True,
                        name="%sjet convolution" % prefix,
                        PCC=False,
                    )
                )
                layers.addLayer(self.conv[-1], previousLayers)

            if prefix == "":
                self.reinforce.append(
                    dijetReinforceLayer(self.d, phase_symmetric=phase_symmetric)
                )
            else:
                self.reinforce.append(
                    lepWReinforceLayer(self.d, phase_symmetric=phase_symmetric)
                )
                pass
            layers.addLayer(self.reinforce[-1], previousLayers)

        self.reinforce = nn.ModuleList(self.reinforce)
        self.conv = nn.ModuleList(self.conv)

    def setGhostBatches(self, nGhostBatches, subset=False):
        for i, reinforce in enumerate(self.reinforce):
            if subset and i % 2:
                continue
            reinforce.conv.setGhostBatches(nGhostBatches)
        for i, conv in enumerate(self.conv):
            if subset and i % 2:
                continue
            conv.setGhostBatches(nGhostBatches)

    def forward(self, x, xx, x0, xx0, debug=False):

        for i, conv in enumerate(self.conv):
            xx = self.reinforce[i](x, xx)
            x = conv(x)
            xx = xx + xx0
            x = x + x0
            xx = NonLU(xx)
            x = NonLU(x)

        xx = self.reinforce[-1](x, xx)
        xx = xx + xx0
        if self.xx0Update:
            xx0 = xx.clone()
            xx = NonLU(xx)
            return xx, xx0
        xx = NonLU(xx)
        return xx

class MinimalAttention(
    nn.Module
):  # https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec https://arxiv.org/pdf/1706.03762.pdf
    def __init__(
        self,
        dim=8,
        heads=1,
        layers=None,
        inputLayers=None,
        # iterations=2,
        phase_symmetric=True,
        do_qv=True,
        device="cuda",
    ):
        super().__init__()

        self.debug = False
        self.device = device
        self.d = dim
        # if phase_symmetric: self.d = self.d//2
        self.phase_symmetric = phase_symmetric
        self.h = heads
        self.dh = self.d // self.h
        self.do_qv = do_qv
        # self.iter = iterations

        self.q_GBN = GhostBatchNorm1d(self.d, name="attention q GBN")
        self.v_GBN = GhostBatchNorm1d(self.d, name="attention v GBN")
        if self.do_qv:
            self.qv_GBN = GhostBatchNorm1d(self.d, name="attention qv GBN")
        # self.origin = nn.Parameter(torch.zeros(1,self.h, self.dh,1,1))
        # self.qv_ref = nn.Parameter(torch. ones(1,self.h, self.dh,1,1))
        self.score_GBN = GhostBatchNorm1d(self.h, name="attention score GBN")
        self.conv = GhostBatchNorm1d(
            dim,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="attention out convolution",
        )

        self.negativeInfinity = torch.tensor(-1e9, dtype=torch.float).to(device)

        if layers:
            layers.addLayer(self.q_GBN, inputLayers)
            layers.addLayer(self.v_GBN, inputLayers)
            layers.addLayer(self.qv_GBN, inputLayers)
            layers.addLayer(self.score_GBN, inputLayers + [self.v_GBN])
            layers.addLayer(self.conv, inputLayers + [self.score_GBN])

    def attention(self, q, v, mask, qv=None, debug=False):
        bs, qsl, vsl = q.shape[0], q.shape[3], v.shape[4]

        # q,qv,v are (bs,h,dh,qsl,1),(bs,h,dh,qsl,vsl),(bs,h,dh,1,vsl)
        score = (q * v).sum(dim=2)  # + (qv).sum(dim=2) # sum over feature space
        if qv is not None:
            score = score + (qv).sum(dim=2)
        # score is (bs,h,qsl,vsl)

        # masked ghost batch normalization of score
        score = score.view(bs, self.h, qsl * vsl)
        score = self.score_GBN(
            score, mask.view(bs, qsl * vsl) if mask is not None else None
        )
        score = score.view(
            bs, self.h, 1, qsl, vsl
        )  # extra dim for broadcasting over features
        # mask fill with negative infinity to make sure masked items do not contribute to softmax
        if mask is not None:
            score = score.masked_fill(mask, self.negativeInfinity)

        v_weights = F.softmax(
            score, dim=4
        )  # compute joint probability distribution for which values  best correspond to each query

        # scale down v's using sigmoid of score, ie, don't want to force the attention block to pick a v if none of them match well.
        score = torch.sigmoid(score)
        v_weights = v_weights * score

        # if mask is not None:
        #     v_weights = v_weights.masked_fill(mask, 0) # make sure masked weights are zero (should be already because score was set to -infinity before softmax and sigmoid)

        if debug or self.debug:
            if mask is not None:
                print("     mask\n", mask[0])
            print("    score\n", score[0])
            print("v_weights\n", v_weights[0])

        # qv         is (bs, h, dh, qsl, vsl)
        #  v         is (bs, h, dh,   1, vsl)
        #  v_weights is (bs, h,  1, qsl, vsl)
        if qv is not None:
            v = v + qv
        # mask is (bs, 1, 1, qsl, vsl)
        if mask is not None:
            v_weights = v_weights.masked_fill(
                mask, 0
            )  # make sure masked weights are zero (should be already because score was set to -infinity before softmax and sigmoid)
            v = v.masked_fill(mask, 0)
        q_res = (v * v_weights).sum(
            dim=4
        )  # query residual features come from weighted sum of values
        # q_res is (bs, h, dh, qsl)

        if debug or self.debug:
            print("q_res\n", q_res[0])
        return q_res, v_weights  # , v_res

    def setGhostBatches(self, nGhostBatches, subset=False):
        self.score_GBN.setGhostBatches(nGhostBatches)
        self.q_GBN.setGhostBatches(nGhostBatches)
        self.v_GBN.setGhostBatches(nGhostBatches)
        if self.do_qv:
            self.qv_GBN.setGhostBatches(nGhostBatches)
        if subset:
            return
        self.conv.setGhostBatches(nGhostBatches)

    def forward(self, q=None, v=None, mask=None, q0=None, qv=None, debug=False):
        bs = q.shape[0]
        qsl = q.shape[2]
        vsl = v.shape[2]

        q_mask, v_mask = None, None
        if mask is not None:
            # check if all items are going to be masked. mask is (bs, qsl, vsl)
            q_mask = mask.all(2).view(bs, 1, qsl)
            v_mask = mask.all(1).view(bs, 1, vsl)
            mask = mask.view(bs, qsl * vsl)

        # print('q before GBN\n',q[0])
        # print('v before GBN\n',v[0])

        if qv is not None:
            qv = qv.view(bs, self.d, qsl * vsl)
            qv = self.qv_GBN(qv, mask)
        q = self.q_GBN(q, q_mask)
        v = self.v_GBN(v, v_mask)

        # print('q after GBN\n',q[0])
        # print('v after GBN\n',v[0])

        # broadcast mask over heads and features
        if mask is not None:
            mask = mask.view(bs, 1, 1, qsl, vsl)
        q = q.view(
            bs, self.h, self.dh, qsl, 1
        )  # extra dim for broadcasting over values
        v = v.view(
            bs, self.h, self.dh, 1, vsl
        )  # extra dim for broadcasting over queries
        if qv is not None:
            qv = qv.view(bs, self.h, self.dh, qsl, vsl)

        # q  = q -self.origin
        # v  =  v-self.origin
        # qv = qv*self.qv_ref

        # calculate attention
        q, v_weights = self.attention(
            q, v, mask, qv, debug
        )  # outputs a linear combination of values (v) given the overlap of the queries (q)
        # q is (bs, h, dh, qsl), v_weights is (bs, h,  1, qsl, vsl)
        q, v_weights = q.view(bs, self.d, qsl), v_weights.view(bs, self.h, qsl, vsl)
        # print('q after attention\n',q[0])
        # if self.phase_symmetric:
        #     q = torch.cat([q, -q], 1)
        q = NonLU(q)
        q = self.conv(q, q_mask)
        if q_mask is not None:
            q = q.masked_fill(q_mask, 0)
        q = q0 + q  # add residual features to q0
        q0 = q.clone()
        q = NonLU(q)

        if self.debug:
            print("q out\n", q[0])
            print("delta q\n", (q[0] - q_in))
            input_val = input("continue debug? [y]/n: ")
            self.debug = input_val == "" or input_val == "y"
        return q, q0, v_weights

class InputEmbed(nn.Module):
    def __init__(
        self,
        dijetFeatures,
        quadjetFeatures,
        ancillaryFeatures=[],
        layers=None,
        device="cuda",
        phase_symmetric=False,
        store=None,
        storeData=None,
    ):
        super(InputEmbed, self).__init__()
        self.layers = layers
        self.debug = False
        self.dD = dijetFeatures
        self.dQ = quadjetFeatures
        self.dA = len(ancillaryFeatures)
        self.ancillaryFeatures = ancillaryFeatures
        self.device = device

        self.store = None
        self.storeData = None

        if self.dA:
            self.ancillaryEmbed = GhostBatchNorm1d(
                self.dA,
                features_out=self.dD,
                phase_symmetric=phase_symmetric,
                conv=True,
                bias=False,
                name="ancillary embedder",
            )
            # self.ancillaryConv  = GhostBatchNorm1d(self.dD, phase_symmetric=phase_symmetric, conv=True, name='Ancillary Convolution')
        ## to do section
        # embed inputs to dijetResNetBlock in target feature space
        self.bJetEmbed = GhostBatchNorm1d(
            4,
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="jet embedder",
        )  # phi is relative to dijet
        self.bJetConv = GhostBatchNorm1d(
            self.dD, phase_symmetric=phase_symmetric, conv=True, name="jet convolution"
        )
        self.nonbJetEmbed = GhostBatchNorm1d(
            4,
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="attention jet embedder",
        )  # phi is removed but isbjet label is added
        self.nonbJetConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="attention jet convolution",
        )
        self.lepEmbed = GhostBatchNorm1d(
            4,
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="lepton embedder",
        )
        self.lepConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="lepton convolution",
        )
        self.nuEmbed = GhostBatchNorm1d(
            2,
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="MET embedder",
        )
        self.nuConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="lepton convolution",
        )
        self.lnuEmbed = GhostBatchNorm1d(
            4,
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="leptonic W embedder",
        )
        self.lnuConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="leptonic W convolution",
        )
        self.WWEmbed = GhostBatchNorm1d(
            4,
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="WW embedder",
        )
        self.WWConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="H -> WW convolution",
        )
        self.MdREmbed = GhostBatchNorm1d(
            3,
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="M(a,b), dR(a,b) embedder",
        )
        self.MdRConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="M(a,b), dR(a,b) convolution",
        )

        self.bsl, self.wsl = 2, 2

        self.mask_bb_same = torch.zeros(
            (1, self.bsl, self.wsl), dtype=torch.bool
        ).to(self.device)
        for i in range(self.bsl):
            self.mask_bb_same[:, i, i] = (
                1  # mask diagonal, don't want mass, dR of jet with itself. (we do want duplicates for i,j and j,i because query and value are treated differently in attention block)
            )

        self.mask_qq_same = torch.zeros(
            (1, self.wsl, self.bsl), dtype=torch.bool
        ).to(self.device)
        for i in range(self.wsl):
            self.mask_qq_same[:, i, i] = 1  # mask diagonal

        self.bbDiJetEmbed = GhostBatchNorm1d(
            4,
            features_out=self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="dijet embedder",
        )  # phi is relative to quadjet
        self.nonbDiJetEmbed = GhostBatchNorm1d(
            3,
            features_out=self.dQ,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="quadjet embedder",
        )  # phi is removed
        self.bbDiJetConv = GhostBatchNorm1d(
            self.dD,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="dijet convolution",
        )
        self.nonbDiJetConv = GhostBatchNorm1d(
            self.dQ,
            phase_symmetric=phase_symmetric,
            conv=True,
            name="quadjet convolution",
        )

        self.layers.addLayer(self.bJetEmbed)
        self.layers.addLayer(self.ancillaryEmbed)
        self.layers.addLayer(self.nonbJetEmbed)
        self.layers.addLayer(self.nonbDiJetEmbed)
        self.layers.addLayer(self.MdREmbed)
        # self.layers.addLayer(self. diMdPhi_embed)
        # self.layers.addLayer(self.triMdPhi_embed)
        self.layers.addLayer(self.bJetConv, [self.bJetEmbed])
        self.layers.addLayer(self.bbDiJetConv, [self.bbDiJetEmbed])
        # self.layers.addLayer(self.ancillaryConv, [self.ancillaryEmbed])
        self.layers.addLayer(self.nonbDiJetConv, [self.nonbDiJetEmbed])
        self.layers.addLayer(self.MdRConv, [self.MdREmbed])
        ### to do section

    def dataPrep(self, b, nb, l, nu, a):  # , device='cuda'):
        device = b.get_device() if b.get_device() >= 0 else "cpu"
        # # if device=='cpu': # prevent overwritting data from dataloader when doing operations directly from RAM rather than copying to VRAM
        # j = j.clone()
        # o = o.clone()
        # a = a.clone()

        n = b.shape[0]
        b = b.view(n, 2, 4)
        nb = nb.view(n, 2, 4)
        l = l.view(n, 1, 4)
        nu = nu.view(n, 1, 2)
        a = a.view(n, self.dA, 2)

        a[:, 1, :] = torch.log(
            a[:, 1, :] - 3
        )  # TODO: find index based on the feature name, check if relevant

        if self.store:
            self.storeData["bjets"] = b.detach().to("cpu").numpy()
            self.storeData["non-bjets"] = nb.detach().to("cpu").numpy()

        ## bb: H->bb dijet candidates, qq: W->qq dijet candidates"
        bb, bbPxPyPzE = addFourVectors(
            b[:, :, (0)], b[:, :, (1)]
        )

        qq, qqPxPyPzE = addFourVectors(
            nb[:, :, (0)],
            nb[:, :, (1)],
            v1PxPyPzE= bbPxPyPzE[:, :, (0)],
            v2PxPyPzE= bbPxPyPzE[:, :, (1)],
        )

        mask, bbMdR, qqMdR, bbnMdR, mask_bb, mask_qq, mask_bbn = None, None, None, None, None, None, None, None
        j_isbJet = torch.cat(
            [b, 2 * torch.ones((n, 1, 2), dtype=torch.float).to(device)], 1
        )  # label bJets with 2 (-1 for mask, 0 for not preselected, 1 for preselected jet)
        nb = torch.cat([j_isbJet, nb], 2)
        mask = (nb[:, 4, :] == -1).to(device)
        bPxPyPzE = PxPyPzE(b)
        nbPxPyPzE = PxPyPzE(nb)

        # For b-jets: compute matrix of dijet masses and opening angles between other jets
        n = bb.shape[0]
        bbMdR = matrixMdR(b, b, v1PxPyPzE=bPxPyPzE, v2PxPyPzE=bPxPyPzE)
        bbMdR = torch.cat(
            [
                bbMdR,
                torch.zeros((n, 1, self.bsl, self.bsl), dtype=torch.float).to(
                    device
                ),
            ],
            1,
        )  # flag with zeros to signify dijet quantities

        mask_bb = mask.view(n, 1, self.bsl) | mask.view(
            n, self.bsl, 1
        )  # mask of 2d matrix of b-jets (i,j) is True if mask[i] | mask[j]
        mask_bb = mask_bb.masked_fill(self.mask_bb_same.to(device), 1)

        # For nonb-jets: compute matrix of dijet masses and opening angles between other jets
        n = qq.shape[0]
        qqMdR = matrixMdR(nb, nb, v1PxPyPzE=nbPxPyPzE, v2PxPyPzE=nbPxPyPzE)
        qqMdR = torch.cat(
            [
                qqMdR,
                torch.zeros((n, 1, self.bsl, self.bsl), dtype=torch.float).to(
                    device
                ),
            ],
            1,
        )  # flag with zeros to signify dijet quantities

        # For lepton and MET, compute transverse mass
        lnu_mT = transverse_mass(l, nu)

        mask_qq = mask.view(n, 1, self.bsl) | mask.view(
            n, self.bsl, 1
        )  # mask of 2d matrix of b-jets (i,j) is True if mask[i] | mask[j]
        mask_qq = mask_qq.masked_fill(self.mask_qq_same.to(device), 1)

        # compute matrix of trijet masses and opening angles between b-dijets and non-bjets
        bbnMdR = matrixMdR(bb, nb, v1PxPyPzE=bbPxPyPzE, v2PxPyPzE=nbPxPyPzE)
        bbnMdR = torch.cat(
            [
                bbnMdR,
                torch.ones((n, 1, self.wsl, self.bsl), dtype=torch.float).to(
                    device
                ),
            ],
            1,
        )  # flag with ones to signify trijet quantities
        lepQQdR = calcDeltaR(l, qq)

        mask_bbn = mask.view(n, 1, self.bsl)

        nb[:, (0, 3), :] = torch.log(1 + nb[:, (0, 3), :])
        nb[isinf(nb)] = -1  # isinf not supported by ONNX

        nb = torch.cat(
            (nb[:, :2, :], nb[:, 3:, :]), 1
        )  # remove phi from othJet features

        b[:, (0, 3), :] = torch.log(1 + b[:, (0, 3), :])
        bb[:, (0, 3), :] = torch.log(1 + bb[:, (0, 3), :])
        qq[:, (0, 3), :] = torch.log(1 + qq[:, (0, 3), :])

        b = torch.cat([b, b[:, :, (1,0)]] , 2)
        # only keep relative angular information so that learned features are invariant under global phi rotations and eta/phi flips
        b[:, 2:3, :] = calcDeltaPhi(bb, b[:, :, :]) # replace jet phi with deltaPhi between dijet and jet

        bb[:, 2:3, :] = calcDeltaPhi(qq, bb[:, :, :])

        return b, bb, qq, a, nb , l, nu, lnu_mT, lepQQdR, bbMdR, qqMdR, bbnMdR, mask, mask_bb, mask_qq, mask_bbn

    def updateMeanStd(self,  b, nb, l, nu, a):
        b, bb, qq, a, nb , l, nu, lnu_mT,lepQQdR, bbMdR, qqMdR, bbnMdR, mask, mask_bb, mask_qq, mask_bbn = self.dataPrep(
            b, nb, l, nu, a)
          # , device='cpu')
        self.ancillaryEmbed.updateMeanStd(a)
        self.nonbJetEmbed.updateMeanStd(qq, mask)

        n, self.bsl, self.wsl = bb.shape[0], bb.shape[2], nb.shape[2]

        MdR = torch.cat(
            (
                bbMdR.view(n, 3, self.bsl * self.bsl),
                qqMdR.view(n, 3, self.wsl * self.wsl),
                bbnMdR.view(n, 3, self.bsl * self.wsl)
            ),
            dim=2,
        )
        mask_MdR = torch.cat(
            (
                mask_bb.view(n, self.bsl * self.bsl),
                mask_qq.view(n, self.wsl * self.wsl),
                mask_bbn.view(n, self.bsl * self.wsl),
            ),
            dim=1,
        )
        self.MdREmbed.updateMeanStd(MdR, mask_MdR)
        # self. diMdPhi_embed.setMeanStd(ooMdPhi.view(n, 2, self.bsl*self.bsl), mask_oo.view(n, self.bsl*self.bsl))
        # self.triMdPhi_embed.setMeanStd(doMdPhi.view(n, 2, self.wsl*self.bsl), mask_do.view(n, self.wsl*self.bsl))

        self.bJetEmbed.updateMeanStd(b)
        self.bbDiJetEmbed.updateMeanStd(bb)
        self.nonbJetEmbed.updateMeanStd(nb)
        self.nonbDiJetEmbed.updateMeanStd(qq)
        self.lepEmbed.updateMeanStd(l)
        self.nuEmbed.updateMeanStd(nu)
        self.lnuEmbed.updateMeanStd(lnu_mT)
        self.WWEmbed.updateMeanStd(lepQQdR)

    def initMeanStd(self):
        self.ancillaryEmbed.initMeanStd()
        self.bJetEmbed.initMeanStd()
        self.bbDiJetEmbed.initMeanStd()
        self.nonbJetEmbed.initMeanStd()
        self.nonbDiJetEmbed.initMeanStd()
        self.lepEmbed.initMeanStd()
        self.nuEmbed.initMeanStd()
        self.lnuEmbed.initMeanStd()
        self.WWEmbed.initMeanStd()

    def setGhostBatches(self, nGhostBatches, subset=False):
        self.ancillaryEmbed.setGhostBatches(nGhostBatches)
        self.bJetEmbed.setGhostBatches(nGhostBatches)
        self.bbDiJetEmbed.setGhostBatches(nGhostBatches)
        self.nonbJetEmbed.setGhostBatches(nGhostBatches)
        self.nonbDiJetEmbed.setGhostBatches(nGhostBatches)

        self.lepEmbed.setGhostBatches(nGhostBatches)
        self.nuEmbed.setGhostBatches(nGhostBatches)
        self.lnuEmbed.setGhostBatches(nGhostBatches)
        self.WWEmbed.setGhostBatches(nGhostBatches)
        if subset:
            return
        self.MdRConv.setGhostBatches(nGhostBatches)
        self.bJetConv.setGhostBatches(nGhostBatches)
        self.bbDiJetConv.setGhostBatches(nGhostBatches)
        self.nonbJetConv.setGhostBatches(nGhostBatches)
        self.nonbDiJetConv.setGhostBatches(nGhostBatches)
        self.lepConv.setGhostBatches(nGhostBatches)
        self.nuConv.setGhostBatches(nGhostBatches)
        self.lnuConv.setGhostBatches(nGhostBatches)
        self.WWConv.setGhostBatches(nGhostBatches)

    def forward(self, b, nb, l, nu, a):
        b, bb, qq, a, nb , l, nu, lnu_mT, lepQQdR, bbMdR, qqMdR, bbnMdR, mask, mask_bb, mask_qq, mask_bbn = self.dataPrep(b, nb, l, nu, a)
        a = self.ancillaryEmbed(a)
        # a = self.ancillaryConv(NonLU(a))
        nb = self.nonbJetEmbed(nb, mask)
        qq = self.nonbDiJetEmbed(qq, mask)
        nb = nb + a
        # print('o after add a\n',o[0])
        # o0 = o.clone()
        nb = self.nonbJetConv(NonLU(nb), mask)
        # print('o after conv a\n',o[0])
        # o = o+o0

        n = bb.shape[0]

        # bbMdR is (n, 3, bsl, wsl)
        # flatten the matrices for passing through convolution
        bbMdR = bbMdR.view(n, 3, self.bsl * self.bsl)
        qqMdR = qqMdR.view(n, 3, self.wsl * self.wsl)
        bbnMdR = bbnMdR.view(n, 3, self.bsl * self.wsl)
        mask_bb = mask_bb.view(n, self.bsl * self.bsl)
        mask_qq = mask_qq.view(n, self.wsl * self.wsl)
        mask_bbn = mask_qq.view(n, self.bsl * self.wsl)
        MdR = torch.cat((bbMdR, qqMdR, bbnMdR), dim=2)
        mask_MdR = torch.cat((mask_bb, mask_qq, mask_bbn), dim=1)
        # MdPhi is (n, 3, osl*osl + dsl*osl)
        MdR = self.MdREmbed(MdR, mask_MdR)
        MdR = self.MdRConv(NonLU(MdR), mask_MdR)

        # get back original shape (equivalent to unflatten)
        bbMdR = MdR[:, :, : self.bsl * self.bsl].view(
            n, self.dD, self.bsl, self.bsl
        )
        qqMdR = MdR[:, :, self.wsl * self.wsl :].view(
            n, self.dD, self.wsl, self.wsl
        )
        bbnMdR = MdR[:, :, self.wsl * self.wsl :].view(
            n, self.dD, self.bsl, self.wsl
        )
        mask_bb = mask_bb.view(n, self.bsl, self.bsl)
        mask_qq = mask_qq.view(n, self.wsl, self.wsl)
        mask_bbn = mask_qq.view(n, self.bsl, self.wsl)

        b = self.bJetEmbed(b)
        bb = self.bbDiJetEmbed(bb)
        b = b + a
        # d = d+a
        # q = q+a
        # j0 = j.clone()
        # d0 = d.clone()
        # q0 = q.clone()
        b = self.bJetConv(NonLU(b))
        bb = self.bJetConv(NonLU(bb))
        # j = j+j0
        # d = d+d0
        # q = q+q0
        l = self.lepEmbed(l)
        nu = self.nuEmbed(nu)
        lnu_mT = self.lnuEmbed(lnu_mT)
        lepQQdR = self.WWEmbed(lepQQdR)
        l = self.lepConv(NonLU(l))
        nu = self.nuConv(NonLU(nu))
        lnu_mT = self.lnuConv(NonLU(lnu_mT))
        lepQQdR = self.WWConv(NonLU(lepQQdR))

        return b, bb, nb, qq, l, nu, lnu_mT, lepQQdR, bbMdR, mask_bb, qqMdR, mask_qq, bbnMdR, mask_bbn 
    
class HCR(nn.Module):
    def __init__(
        self,
        dijetFeatures,
        quadjetFeatures,
        ancillaryFeatures,
        useOthJets="",
        device="cuda",
        nClasses=1,
        architecture="HCR",
    ):
        super(HCR, self).__init__()
        self.debug = False
        self.dA = len(ancillaryFeatures)
        self.dD = dijetFeatures  # dimension of embeded   dijet feature space
        self.dQ = quadjetFeatures  # dimension of embeded quadjet feature space
        self.device = device
        dijetBottleneck = None
        self.name = (
            architecture
            + ("+" + useOthJets if useOthJets else "")
            + "_%d" % (dijetFeatures)
        )
        self.nC = nClasses
        self.store = None
        self.storeData = {}
        self.onnx = False
        self.nGhostBatches = 64
        self.phase_symmetric = True

        self.layers = layerOrganizer()

        # this module handles input shifting scaling and learns the optimal scale and shift for the appropriate inputs
        self.inputEmbed = InputEmbed(
            self.dD,
            self.dQ,
            ancillaryFeatures,
            useOthJets=self.useOthJets,
            layers=self.layers,
            device=self.device,
            phase_symmetric=self.phase_symmetric,
        )

        # Stride=3 Kernel=3 reinforce dijet features, in parallel update jet features for next reinforce layer
        # |1|2|1,2|3|4|3,4|1|3|1,3|2|4|2,4|1|4|1,4|2|3|2,3|
        #     |1,2|   |3,4|   |1,3|   |2,4|   |1,4|   |2,3|
        self.bbDiJetResNetBlock = ResNetBlock(
            self.dD,
            prefix="",
            nLayers=2,
            phase_symmetric=self.phase_symmetric,
            device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.bJetConv, self.inputEmbed.bbDiJetConv],
        )
        self.nonbDiJetResNetBlock = ResNetBlock(
            self.dD,
            prefix="",
            nLayers=2,
            phase_symmetric=self.phase_symmetric,
            device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.nonbJetConv, self.inputEmbed.nonbDiJetConv],
        )
        self.lepWResNetBlock = ResNetBlock(
            self.dD,
            prefix="leptonic W",
            nLayers=2,
            phase_symmetric=self.phase_symmetric,
            device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.lepConv, self.inputEmbed.nuEmbed, self.inputEmbed.lnuEmbed],
        )

        self.attention_WW = MinimalAttention(
            self.dD,
            heads=2,
            phase_symmetric=self.phase_symmetric,
            layers=self.layers,
            inputLayers=[self.lepWResNetBlock.conv[-1], self.nonbDiJetResNetBlock.reinforce[-1]],
            device=self.device,
        )

        self.attention_hh = MinimalAttention(
            self.dD,
            heads=2,
            phase_symmetric=self.phase_symmetric,
            layers=self.layers,
            inputLayers=[self.inputEmbed.bJetConv, self.attention_WW],
            device=self.device,
        )
        previousLayer = self.attention_hh.conv

        # # embed inputs to quadjetResNetBlock in target feature space
        # self.dijetEmbedInQuadjetSpace = GhostBatchNorm1d(self.dQ, phase_symmetric=self.phase_symmetric, conv=True, name='dijet embed in quadjet space')

        # Embed enhanced WW representation
        self.WW_final_embed = GhostBatchNorm1d(
            self.dD, 
            features_out=self.dD,  # or some common dimension
            conv=True, 
            name="WW final embed"
        )

        # Embed enhanced HH representation  
        self.HH_final_embed = GhostBatchNorm1d(
            self.dD,
            features_out=self.dD, 
            conv=True,
            name="HH final embed"
        )

        self.layers.addLayer(self.WW_final_embed, [self.attention_WW])
        self.layers.addLayer(self.HH_final_embed, [self.attention_hh])

        self.final_combine = GhostBatchNorm1d(
            self.dD * 2,  # Input from concatenated WW + HH 
            features_out=self.nC, 
            conv=True, 
            name="combine WW and HH"
        )
        # self.layers.addLayer(self.dijetEmbedInQuadjetSpace, [previousLayer])
        self.layers.addLayer(self.final_combine, [self.WW_final_embed, self.HH_final_embed])


    def embedding_layers(self):
        return sorted(set(self.layers.layers).difference(self.output_layers()))

    def output_layers(self):
        return [self.out.index]

    def updateMeanStd(self, j, o, a):
        self.inputEmbed.updateMeanStd(j, o, a)

    def initMeanStd(self):
        self.inputEmbed.initMeanStd()

    def setGhostBatches(self, nGhostBatches, subset=False):
        self.inputEmbed.setGhostBatches(nGhostBatches, subset)
        self.dijetResNetBlock.setGhostBatches(nGhostBatches, subset)
        if self.useOthJets:
            self.attention_oo.setGhostBatches(nGhostBatches, subset)
            self.attention_do.setGhostBatches(nGhostBatches, subset)
        self.quadjetResNetBlock.setGhostBatches(nGhostBatches, subset)
        # self.combine_q.setGhostBatches(nGhostBatches)
        self.select_q.setGhostBatches(nGhostBatches)
        self.out.setGhostBatches(nGhostBatches)
        self.nGhostBatches = nGhostBatches

    def forward(self, b, nb, l, nu, a):
        self.forwardCalls += 1
        # print('\n-------------------------------\n')
        b, bb, nb, qq, l, nu, lnu_mT, lepQQdR,  bbMdR, mask_bb, qqMdR, mask_qq, bbnMdR, mask_bbn  = self.inputEmbed(
            b, nb, l, nu, a
        )  # format inputs to array of objects and apply scalers and GBNs
        # print('o after inputEmbed\n',o[0])
        n = b.shape[0]
        #
        # Build up dijet pixels with jet pixels and initial dijet pixels
        #

        # Embed the jet 4-vectors and dijet ancillary features into the target feature space
        b0 = b.clone()
        bb0 = bb.clone()
        nb0 = nb.clone()
        qq0 = qq.clone()
        l0 = l.clone()
        nu0 = nu.clone()

        b = NonLU(b)
        bb = NonLU(bb)
        nb = NonLU(nb)
        qq = NonLU(qq)
        l = NonLU(l)
        nu = NonLU(nu)
        lnu_mT = NonLU(lnu_mT)

        # bb: H -> bb candidates, qq: W -> qq candidates 
        bb, bb0 = self.bbDiJetResNetBlock(b, bb, b0, bb0, debug=self.debug)
        qq, qq0 = self.nonbDiJetResNetBlock(nb, qq, nb0, qq0, debug=self.debug)

        # Create unified W candidate with all information available
        lep_W = torch.cat([l, nu, lnu_mT], dim=-1)  # 7D vector
        lep_W0 = lep_W.clone()
        lep_W = NonLU(lep_W)
        lep_W, lep_W0 = self.lepWResNetBlock(lep_W, lep_W, lep_W0, lep_W0) # pass same inputs twice for resnet function

        bbMdR = NonLU(bbMdR)
        qqMdR = NonLU(qqMdR)
        bbnMdR = NonLU(bbnMdR)

        WW, WW0, WW_weights = self.attention_WW(
            lep_W,        # queries: leptonic W candidate
            qq,           # values: hadronic W candidate (non-bjet dijets) 
            mask_qq,      # mask for non-bjets
            lep_W0,       # residual for leptonic W
            lepQQdR,       # physics relationships between leptonic W and hadronic W
            self.debug
        )
        HH, HH0, HH_weights = self.attention_hh(
            bb,           # queries: HH candidate (bb dijets)
            WW,  # values: WW system from previous attention
            mask_bbn,
            bb0,          # residual for bb
            bbnMdR,    # physics relationships between H-> bb and W -> qq
            self.debug
        )

        if self.store:
            self.storeData["WW"] = WW.detach().to("cpu").numpy()
            self.storeData["HH"] = HH.detach().to("cpu").numpy()

        WW_final = self.WW_final_embed(WW)
        HH_final = self.HH_final_embed(HH)
        
        HH_score = WW_final + HH_final
        HH_logits = self.final_combine(HH_score)
        HH_logits = HH_logits.view(n, self.nC)

        # Convert to probabilities for output/storage
        if self.store:
            HH_score = F.softmax(HH_logits, dim=1)
            if self.store:
                self.storeData["HH_score"] = HH_score.detach().to("cpu").numpy()
                self.storeData["WW_weights"] = WW_weights.detach().to("cpu").numpy() # see how much H->WW contributes to discrimination
                self.storeData["HH_weights"] = HH_weights.detach().to("cpu").numpy()


        return HH_logits

    def setStore(self, store):
        self.store = store
        self.inputEmbed.store = store
        self.inputEmbed.storeData = self.storeData

    def writeStore(self):
        # print(self.storeData)
        print(self.store)
        np.save(self.store, self.storeData)