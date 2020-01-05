"""

Willis Sanchez-duPont
wsanchezdupont@g.hmc.edu

Implementation of Grad-CAM (https://arxiv.org/abs/1610.02391).

Notes:

- (batch summing) In order to be able to process a batch of examples at once,
  we can sum out the batch dimension of the model output and then call .backward()
  on each class, rather than call the .backward() function  B samples * C classes
  number of times. The gradient of the class output w.r.t each activation map pixel
  is unchanged by this operation (i.e. in backprop we take each output and pass
  it thru a sum operator, and the local gradient of the sum operation w.r.t. its
  inputs is one, so the chain rule leaves dC_i/da_abcd intact where a,b,c, and d
  are indices for batch, channel, row, and column respectively and C_i is class i)

- (dog example) This repo uses the same examples as https://github.com/jacobgil/pytorch-grad-cam
  Note that the generated CAM for the bull mastiff class (243) produces a result different from
  "dog.jpg". Not totally sure why this is the case, however cloning the previously mentioned
  repo and running grad-cam.py results in the same image as the one produced by the code in this
  file (perhaps that image was generated from a previous version of their code?).
  Also note that "cat.jpg" is identical to the results from this repo.
  - (argument parsing) For some reason, argparse insists that the -g flag comes after --classes.
    This problem doesn't arise if you use the full flag (i.e. --guided)

"""

import numpy as np
import torch
import torch.nn as nn

class GradCAM():
    """
    GradCAM

    Class that performs grad-CAM or guided grad-CAM on a given model (https://arxiv.org/abs/1610.02391).
    """
    def __init__(self,model,device='cuda',verbose=False):
        """
        Constructor.

        inputs:
            model - (torch.nn.Module) Indexable module to perform grad-cam on (e.g. torch.nn.Sequential)
            device - (str) device to perform computations on
            verbose - (bool) enables printouts for e.g. debugging
        """
        self.device = device
        self.model = model
        model.to(self.device) # push params to device
        self.model.eval()

        self.activations = torch.empty(0) # initialize activation maps
        self.grads = torch.empty(0) # initialize gradient maps
        self.results = torch.empty(0) # network output holder
        self.gradCAMs = torch.empty(0) # output maps
        self.fh = [] # module forward hook
        self.bh = [] # module backward hook

        self.guidehooks = [] # list of hooks on conv layers for guided backprop
        self.guidedGrads = torch.empty(0) # grads of class activations w.r.t. inputs
        self.deconv = None

        self.verbose = verbose

    def __call__(self,x,submodule=None,classes='all',guided=False,deconv=False):
        """
        __call__

        Compute class activation maps for a given input and submodule.

        inputs:
            x - (torch.Tensor) input image to compute CAMs for.
            submodule - (str or torch.nn.Module) name of module to get CAMs from OR submodule object to hook directly
            classes - (list) list of indices specifying which classes to compute grad-CAM for (MUST CONTAIN UNIQUE ELEMENTS). CAMs are returned in same order as specified in classes.
            guided - (None or True or list) Guided grad-CAM enabler. If None, no guided gradcam. If True,
                     apply guided backprop to all modules in self._modules.items() that have a
                     __class__.__name__ of ReLU. If list, apply hooks to all modules in the given list
            deconv - (bool) use 'deconvolution' backprop operation from https://arxiv.org/abs/1412.6806

        outputs:
            self.gradCAMs - (numpy.ndarray) numpy.ndarray of shape (batch,classes,u,v) where u and v are the activation map dimensions
        """
        x = x.to(self.device)
        x.retain_grad()

        if self.verbose:
            print('samples pushed to device')

        if submodule is None:
            self.activations = x # treat inputs as activation maps

        # register grad hook for vanilla grad-CAM
        self.set_gradhook(submodule)
        if self.verbose:
            print('CAM hooks set')

        # register hooks on ReLUs for guided backprop or 'deconvolution'
        self.deconv = deconv
        self.set_guidehooks(guided)

        # forward pass
        if guided and x.requires_grad==False:
            x.requires_grad = True

        self.results = self.model(x) # store result (already on device)
        self.gradCAMs = torch.empty(0)
        if self.verbose:
            print('model outputs computed')

        # grad and CAM computations
        summed = self.results.sum(dim=0) # sum out batch results. See note at top of file

        if classes == 'all':
            classes = [x for x in range(self.results.size(1))] # list of all classes
        for c in classes: # loop thru classes
            if c == classes[-1]:
                summed[c].backward()
            else:
                summed[c].backward(retain_graph=True) # retain graph to be able to backprop without calling forward again

            # see paper for details on math
            coeffs = self.grads.sum(-1).sum(-1) / (self.grads.size(2) * self.grads.size(3)) # coeffs has size = (batchsize, activations.shape(1)
            prods = coeffs.unsqueeze(-1).unsqueeze(-1)*self.activations # align dims to get appropriate linear combination of feature maps (which reside in dim=1 of self.activations)
            cam = torch.nn.ReLU()(prods.sum(dim=1)) # sum along activation dimensions (result size = batchsize x U x V)
            self.gradCAMs = torch.cat([self.gradCAMs, cam.unsqueeze(0).to('cpu')],dim=0) # add CAMs to function output variable
            self.model.zero_grad() # clear gradients for next backprop

            self.guidedGrads = torch.cat([self.guidedGrads, x.grad.cpu().unsqueeze(0)],dim=0) # save grad of class w.r.t. inputs
            x.grad.data = torch.zeros_like(x.grad.data) # clear out input grads as well

            if self.verbose:
                print('class {} gradCAMs computed'.format(c))

        # remove hooks
        self.bh.remove()
        if submodule is not None:
            self.fh.remove()

        if len(self.guidehooks) == 0:
            for hook in self.guidehooks:
                hook.remove()
        if self.verbose:
            print('all hooks removed')

        self.gradCAMs = self.gradCAMs.permute(1,0,2,3).detach().numpy() # batch first, requires_grad=False, and convert to numpy array
        self.guidedGrads = self.guidedGrads.permute(1,0,2,3,4).numpy() # batch,class,channel,height,width

        if self.verbose:
            print('self.gradCAMs.shape =',self.gradCAMs.shape)
            if guided:
                print('self.guidedGrads.shape =',self.guidedGrads.shape)

        if guided:
            resizedGradCAMs = np.zeros((self.gradCAMs.shape[0],self.gradCAMs.shape[1],3,x.shape[-2],x.shape[-1]))
        else:
            resizedGradCAMs = np.zeros((self.gradCAMs.shape[0],self.gradCAMs.shape[1],x.shape[-2],x.shape[-1]))
        for i in range(self.gradCAMs.shape[0]):
            for j in range(self.gradCAMs.shape[1]):
                resizedcam = cv2.resize(self.gradCAMs[i][j],(x.shape[-2],x.shape[-1]))
                if guided:
                    resizedGradCAMs[i][j] = resizedcam.reshape(1,*resizedcam.shape) * self.guidedGrads[i][j]
                else:
                    resizedGradCAMs[i][j] = resizedcam

                if self.verbose:
                    print('resizedGradCAMs.shape =',resizedGradCAMs.shape)
                    print('resizedGradCAMs.min() =',resizedGradCAMs.min())
                    print('resizedGradCAMs.max() =',resizedGradCAMs.max())

                resizedGradCAMs[i][j] = resizedGradCAMs[i][j] - np.min(resizedGradCAMs[i][j])
                resizedGradCAMs[i][j] = resizedGradCAMs[i][j] / np.max(resizedGradCAMs[i][j])

        self.gradCAMs = resizedGradCAMs
        if self.verbose:
            print('self.gradCAMs.shape =',self.gradCAMs.shape)

        return self.gradCAMs

    def set_gradhook(self,submodule=None):
        """
        set_gradhook

        Set hook on submodule for grad-CAM computation.

        inputs:
            submodule - (str or torch.nn.Module) name of submodule within self.model to hook OR submodule to hook directly
        """
        # define hook functions
        def getactivation(mod,input,output):
            """
            getactivation

            Copy activations to self.activations on forward pass.

            inputs:
                mod - (torch.nn.Module) module being hooked
                input - (tuple) inputs to submodule being hooked (ignore)
                output - (tensor) output of submodule being hooked (NOTE: untested for modules with multiple outputs)
            """
            self.activations = output

        def getgrad(mod,gradin,gradout):
            """
            getgrad

            Get gradients during hook.

            inputs:
                mod - (torch.nn.Module) module being hooked
                gradin - (tuple) inputs to last operation in mod (ignore)
                gradout - (tuple) gradients of class activation w.r.t. outputs of mod
            """
            self.grads = gradout[0]

        def getgrad_input(g):
            """
            getgrad_input

            Get gradient on backward pass when using inputs for CAMs.

            inputs:
                g - (torch.Tensor) tensor to set/store as gradient on backward pass
            """
            self.grads = g


        if submodule is None: # if using inputs as activation maps
            self.activations.requires_grad=True # make sure input grads are available
            self.bh = self.activations.register_hook(getgrad_input) # save gradient
        elif submodule.__class__.__name__ == 'str': # if using string name of submodule...
            for name,module in self.model._modules.items():
                if name == submodule:
                    self.fh = module.register_forward_hook(getactivation) # forward hook
                    self.bh = module.register_backward_hook(getgrad) # backward hook
        else: # if using the submodule itself
            self.fh = submodule.register_forward_hook(getactivation) # forward hook
            self.bh = submodule.register_backward_hook(getgrad) # backward hook

    def guidedhook(self,mod,gradin,gradout):
        """
        guidedhook

        Hook that applies a clamping operation to gradients for guided backpropagation.
        Modifies gradient during backprop by returning an altered gradin.

        inputs:
            mod - (torch.nn.Module) module being hooked
            gradin - (tuple) tuple of gradients of loss w.r.t inputs and parameters of mod.
                     Different for various layer types. For Conv2d layers, it's
                     (input.grad, mod.weight.grad, mod.bias.grad)
            gradout - (tuple) tuple of gradients of loss w.r.t. outputs of mod
        """
        if len(gradin) != 1:
            raise Exception('len(gradin) != 1. It should be equal to 1 for ReLU layers. Verify which modules you are hooking.')

        return (torch.clamp(gradin[0],min=0),)

    def deconvhook(self,mod,gradin,gradout):
        """
        deconvhook

        Hook that applies 'deconvolution' operation from https://arxiv.org/abs/1412.6806

        inputs:
            mod - (torch.nn.Module) module being hooked
            gradin - (tuple) tuple of gradients of loss w.r.t inputs and parameters of mod.
                     Different for various layer types. For Conv2d layers, it's
                     (input.grad, mod.weight.grad, mod.bias.grad)
            gradout - (tuple) tuple of gradients of loss w.r.t. outputs of mod
        """
        if len(gradin) != 1:
            raise Exception('len(gradin) != 1. It should be equal to 1 for ReLU layers. Verify which modules you are hooking.')

        return (torch.clamp(gradout[0],min=0),)

    def set_guidehooks(self,guided=False,deconv=False):
        """
        set_guidehooks

        Set hooks on ReLU modules for guided backprop.

        inputs:
            guided - (bool or torch.nn.Module list) use guided grad-CAM. Hooks all ReLU submodules if True, hooks given modules in list if list
            deconv - (bool) optionally use 'deconvolution' operation instead of guided backpropagation
        """
        hooktype = self.guidedhook
        if deconv == True:
            hooktype = self.deconvhook

        if guided is True:
            for name,module in self.model._modules.items():
                if module.__class__.__name__ == 'ReLU': # TODO: make this more elegant instead of creating a dummy Conv2d
                    h = module.register_backward_hook(hooktype)
                    self.guidehooks.append(h)
        elif guided.__class__.__name__ == 'list': # manually provided list of modules to hook
            for module in guided:
                h = module.register_backward_hook(hooktype)
                self.guidehooks.append(h)

        if self.verbose:
            print('len(self.guidehooks) =',len(self.guidehooks))
            print('guided backprop hooks set')

class Flatten(torch.nn.Module):
    """
    Flatten

    torch.flatten as a layer.
    """
    def __init__(self,start_dim=1):
        """
        __init__

        Constructor.

        inputs:
        start_dim - (bool) dimension to begin flattening at.
        """
        super(Flatten,self).__init__()
        self.start_dim = start_dim

    def forward(self,x):
        """
        forward

        Forward pass.
        """
        return x.flatten(self.start_dim)

def create_masked_image(x,cam,filename='examples/testimage.jpg'):
    """
    similar to show_cam_on_image from https://github.com/jacobgil/pytorch-grad-cam
    """
    mask = cv2.applyColorMap(np.uint8(cam*255),cv2.COLORMAP_JET)
    mask = np.float32(mask) / 255
    mask = mask + np.float32(x)
    mask = mask / np.max(mask)
    if filename is not None:
        cv2.imwrite(filename,np.uint8(mask*255))

def preprocess_image(img):
    """
    Copied from https://github.com/jacobgil/pytorch-grad-cam grad-cam.py
    """
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input

"""
MODULE TESTING
"""
if __name__ == '__main__':
    '''
    Imports
    '''
    from matplotlib import pyplot as plt
    import numpy as np
    from torch.autograd import Variable,Function

    # TODO: add argparse
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument('-d','--device',type=str,default='cuda',help="Device to use for computations. | 'cuda' (default) or 'cpu'")
    parser.add_argument('-g','--guided',action='store_true',help="Flag to use guided grad-CAM")
    parser.add_argument('--deconv',action='store_true',help="Flag to enable 'deconv' operation during backprop instead of guided grad-CAM (must have --guided flag enabled as well) (see https://arxiv.org/abs/1412.6806)")
    parser.add_argument('-v','--verbose',action='store_true',help="Enable verbose mode")
    parser.add_argument('-c','--classes',type=int,nargs='+',default=281,help="Variable number of arguments giving list of vgg class indices to compute CAMs for (e.g. '281' or '243 281' etc.) | default: 281 (tabby cat)")
    parser.add_argument('-i','--imname',type=str,default='./examples/both.png',help="Filename of desired input image")
    parser.add_argument('-b','--batchsize',type=int,default=2,help="Number of samples to process as a batch (copies of same image). Can be used for timing")
    parser.add_argument('-t','--timer',action='store_true',help="Flag to enable timing of grad-CAM computation. Enables timing of grad-CAM __call__ only")

    opts = parser.parse_args()
    '''
    VGG19 test using examples from https://github.com/jacobgil/pytorch-grad-cam for comparison
    '''
    import cv2
    from torchvision.models import vgg19
    if opts.timer:
        import time

    # this block is mostly copied/adapted from the repo linked above
    x = cv2.imread(opts.imname,1) # 1 for color image
    x = np.float32(cv2.resize(x, (224,224))) / 255 # move to range [0,1]
    im = preprocess_image(x)
    im = torch.cat(opts.batchsize*[im],dim=0)

    # create network and set to eval mode
    vgg = vgg19(pretrained=True)

    # create GradCAM object and generae grad-CAMs
    GC = GradCAM(model=vgg, device=opts.device, verbose=opts.verbose)
    if opts.classes.__class__.__name__ == 'list':
        classes = opts.classes
    elif opts.classes.__class__.__name__ == 'int':
        classes = [opts.classes]
    else:
        raise Exception("Invalid list of classes provided!")

    # place hooks on ReLU modules inside the vgg.features submodule
    if opts.guided:
        hookmods = []
        for name,module in vgg.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                hookmods.append(module)
        g = hookmods
    else:
        g = False

    if opts.timer:
        start = time.time()
    cam = GC(im,submodule=GC.model.features._modules["35"],classes=classes,guided=g,deconv=opts.deconv)
    if opts.timer:
        end = time.time()
        print('grad-CAM computation time = {}'.format(end-start))

    # reshape grad-CAMs + create heatmaps + save images
    # Note: only first image from batch is used because this test script processes a batch of the same sample
    for i in range(len(classes)):
        mask = cam[0][i]
        if not g:
            mask = (mask - np.min(mask)) / np.max(mask)
        if g:
            mask = mask.transpose(1,2,0)
            ggrads = GC.guidedGrads[0][i].transpose(1,2,0)
            ggrads = ggrads - ggrads.min()
            ggrads = ggrads / ggrads.max()
            cv2.imwrite(filename='examples/guided_backprop_class_{}.jpg'.format(classes[i]),img=np.uint8(ggrads*255))
            cv2.imwrite(filename='examples/guided_gradcam_class_{}.jpg'.format(classes[i]),img=np.uint8(mask*255))
        else:
            create_masked_image(x,mask,filename='examples/cam_class_{}.jpg'.format(classes[i]))
