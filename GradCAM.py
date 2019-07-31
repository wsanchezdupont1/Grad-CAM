"""

Willis Sanchez-duPont

Grad-CAM implementation (ARXIV LINK HERE).

Notes:

- Currently only works on torch.nn.Sequentials or torch.nn.ModuleLists.

TODO:
- Generalize to modules with multiple outputs (hooks, etc.)

"""

import torch
import torch.nn as nn

class GradCAM(nn.Module):
    """
    GradCAM

    Class that performs grad-CAM algorithm on a given model (ARXIV LINK HERE).
    """
    def __init__(self,model,device='cuda'):
        """
        Constructor.

        inputs:
            model - (torch.nn.Module) Indexable module to perform grad-cam on (e.g. torch.nn.Sequential)
            device - (str) device to perform computations on
        """
        super(GradCAM,self).__init__()

        self.device = device
        self.input = torch.empty(0)
        self.model = model.to(self.device) # push params to device
        self.activation = torch.empty(0) # initialize activation maps
        self.grad = torch.empty(0) # initialize gradient maps
        self.result = torch.empty(0) # network output holder
        self.gradCAMs = torch.empty(0) # output maps
        self.fh = [] # module forward hook
        self.bh = [] # module backward hook

    def forward(self,x,layerNum=None):
        """
        forward

        Compute class activation maps for a given input and layer.

        inputs:
            x - (torch.Tensor) input image to compute CAMs for.
            layerNum - (int) index of element within model to compute CAMs for.
                       Used as index of self.model (i.e. the hook is applied to
                       self.model[layerNum]. If None, use inputs
        """
        self.input = x.to(self.device)

        if layerNum is None:
            self.activation = self.input # treat inputs as activation maps

        # hook registration
        self.sethooks(layerNum)

        self.result = self.model(self.input) # store result (already on device)
        self.gradCAMs = torch.empty(0) # NOTE: leave on cpu


        # for each batch element + class, compute and store maps
        # for n in range(self.result.size(0)):
        #     for c in range(self.result.size(1)):
        #         self.result[n][c].backward(retain_graph=True)
        #         print('self.grad.shape =',self.grad.shape)
        #         coeffs = self.grad[n].sum(-1).sum(-1) / (self.grad.size(2) * self.grad.size(3)) # coeffs has size = self.activations.size(1)
        #         print('coeffs.shape =',coeffs.shape)
        #         prods = coeffs.unsqueeze(-1).unsqueeze(-1)*self.activation[n] # align dims to get appropriate linear combination of feature maps (which reside in dim=1 of self.activations)
        #         cam = torch.nn.ReLU()(prods.sum(dim=1)) # sum along activation dimensions (result size = batchsize x U x V)
        #         self.gradCAMs = torch.cat([self.gradCAMs, cam.unsqueeze(0).to('cpu')],dim=0) # add CAMs to function output variable
        #         self.model.zero_grad() # clear gradients for next backprop


        summed = self.result.sum(dim=0) # sum out batch
        for c in range(self.result.size(1)):
            if c == self.result.size(1)-1:
                summed[c].backward()
            else:
                summed[c].backward(retain_graph=True)

            coeffs = self.grad.sum(-1).sum(-1) / (self.grad.size(2) * self.grad.size(3)) # coeffs has size = self.activations.size(1)
            prods = coeffs.unsqueeze(-1).unsqueeze(-1)*self.activation # align dims to get appropriate linear combination of feature maps (which reside in dim=1 of self.activations)
            cam = torch.nn.ReLU()(prods.sum(dim=1)) # sum along activation dimensions (result size = batchsize x U x V)
            print('cam.shape =',cam.shape)
            self.gradCAMs = torch.cat([self.gradCAMs, cam.unsqueeze(0).to('cpu')],dim=0) # add CAMs to function output variable
            self.model.zero_grad() # clear gradients for next backprop


        self.bh.remove()
        if layerNum is None:
            self.fh.remove()

        return self.gradCAMs.permute(1,0,2,3)

    def sethooks(self,layerNum=None):
        """
        sethooks

        Set hooks for each layer.

        inputs:
            layerNm - (int) index value of self.model to compute CAMs for (i.e. self.model[layerNum])
        """
        # define hook functions
        def getactivation(mod,input,output):
            """
            getactivation

            Copy activations to self.activation on forward pass.

            inputs:
                mod - (torch.nn.Module) module being hooked
                input - (tuple) inputs to layer being hooked
                output - (tensor) output of layer being hooked (NOTE: if module has multiple outputs this may be a tuple.)
            """
            self.activation = output

        def getgrad(mod,gradin,gradout):
            """
            getgrad

            Get gradients during hook.

            inputs:
                mod - (torch.nn.Module) module being hooked
                gradin - (tuple) inputs to last operation in mod
                gradout - (tuple) tuple of accumulated gradients w.r.t. outputs of mod
            """
            self.grad = gradout[0]

        def getgrad_input(g):
            """
            getgrad_input

            Get gradient on backward pass when using inputs for CAMs.

            inputs:
                g - (torch.Tensor) tensor to set/store as gradient on backward pass
            """
            self.grad = g


        if layerNum is None: # if using inputs
            self.activation.requires_grad=True # make sure grads are available
            self.bh = self.activation.register_hook(getgrad_input) # save gradient
        else:
            # fwd + back hooks on layerNum module
            self.fh = self.model[layerNum].register_forward_hook(getactivation)
            self.bh = self.model[layerNum].register_backward_hook(getgrad)


class Flatten(torch.nn.Module):
    """
    Flatten

    torch.view as a layer.
    """
    def __init__(self,start_dim=1):
        """
        __init__

        Constructor.

        inputs:
        start_idm - (bool) dimension to begin flattening at.
        """
        super(Flatten,self).__init__()
        self.start_dim = start_dim

    def forward(self,x):
        """
        forward

        Forward pass.
        """
        return x.flatten(self.start_dim)

"""
MODULE TESTING
"""
if __name__ == '__main__':
    '''
    Imports
    '''
    from matplotlib import pyplot as plt
    import numpy as np


    '''
    Basic test with dummy inputs/model.
    '''
    x = torch.rand(6,3,4,4)
    m = torch.nn.Sequential(torch.nn.Conv2d(3,4,2), torch.nn.ReLU(), Flatten(), torch.nn.Linear(4*3*3,4))
    GC = GradCAM(m)
    maps = GC(x,layerNum=0)

    plt.subplot(1,2,1)
    plt.imshow(np.array(x[0].permute(1,2,0))) # plot input exapmle
    plt.subplot(1,2,2)
    plt.imshow(np.array(maps[0][0].detach())) # plot map example
    plt.show()
