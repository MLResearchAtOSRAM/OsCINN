""" Copyright Nerrror (Alexander Luce)
    Copyright (c) 2021 AMS-Osram"""

from model.cond_resnet import *
import warnings

from tqdm import tqdm
import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import FrEIA as fr
# from FrEIA.modules import GLOWCouplingBlock as glow
# from FrEIA.modules import NICECouplingBlock as nice
from FrEIA.modules import RNVPCouplingBlock as rnvp
from FrEIA.modules import PermuteRandom as permute
from FrEIA.modules.all_in_one_block import *
from FrEIA.framework.reversible_sequential_net import *

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(torch.cat([init_dim * torch.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

class OsCinn1D():

    def __init__(self, input_dim, cond_dim, num_of_blocks=8, cuda=True):
        '''
        Creates a cINN for 1D input and conditional data.

        OsCINN1D is a wrapper class for the FrEIA Sequential cINN from the FrEIA package which was
        used in the Master Thesis of Alexander Luce for prediction of multilayer thin-films. 

        Parameters
        ----------
        input_dim : int 
            Number of values of the data which should be inverted
        cond_dim : int
            Number of values of the conditional data
        num_of_blocks : int, optional 
            Number of invertible coupling blocks
        cuda : bool
            specifies if the network is processed on the GPU 

        Warnings
        --------
        if cuda is set True but no Cuda drivers are available a warning is displayed
            
        '''
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.num_of_blocks = num_of_blocks

        self.optimizer = None
        self.optimizer_kwargs = {}
        self.scheduler = None
        self.scheduler_kwargs = {}

        if torch.cuda.is_available() and cuda:
            self.cuda = cuda
        elif not cuda:
            self.cuda = cuda
        else:
            self.cuda = False
            warnings.warn(f'Cuda not available - Move Network to cpu instead\nMake sure to only pass Tensors on cpu to the Network!')


        self.cond_net = ResNet18_1D_dense_output(channels = [20,20], levels = 2) # TODO: automatic shape
        self.cinn = Ff.SequenceINN(self.input_dim)


        for block in range(self.num_of_blocks):
            self.cinn.append(Fm.AllInOneBlock,
                            cond=block, 
                            cond_shape=[15], # TODO: automatic shape
                            subnet_constructor=self.subnet_fc, 
                            permute_soft=True, 
                            affine_clamping=3., 
                            global_affine_init=0.8)
            # affine-clamping höher drehen
            # global_affine_init etwas unter 1 (0.8zb) falls net am anfang instabil ist

        if self.cuda:
            self.cond_net = self.cond_net.to('cuda')
            self.cinn = self.cinn.to('cuda')
            print('Successfully moved Network to GPU')
        else:
            self.cond_net = self.cond_net.to('cpu')
            self.cinn = self.cinn.to('cpu')
            
        self.trainable_params = (list(self.cond_net.parameters()) + list(self.cinn.parameters()))

    @staticmethod
    def subnet_fc(dims_in, dims_out):
        '''
        Addes a fully connected subnetwork to a conditional coupling block. The parameters
        should be automatically set by the AllInOneBlock

        Parameters
        ----------
        dims_in : int
            input dimension 
        dims_out : int
            output dimension
        '''
        return nn.Sequential(nn.Linear(dims_in, 512), nn.ReLU(), nn.BatchNorm1d(512),# vielleicht mal ohne bn probieren
                            nn.Linear(512, 512), nn.ReLU(),
                            nn.Linear(512,  dims_out))

    @property
    def optimizer(self):
        '''
        Sets the PyTorch optimizer (eg. Adam) for the cINN.
        '''
        return self._optimizer 
    
    @optimizer.setter
    def optimizer(self, torch_optimizer):
        self._optimizer = torch_optimizer

    @optimizer.getter
    def optimizer(self):
        if self._optimizer is None:
            return
        return self._optimizer(self.trainable_params, **self.optimizer_kwargs)

    @property
    def scheduler(self):
        '''
        Optional, sets the PyTorch scheduler (eg. MultiStepLR) for the cINN.
        '''
        return self._scheduler
    
    @scheduler.setter
    def scheduler(self, torch_scheduler):
        self._scheduler = torch_scheduler

    @scheduler.getter
    def scheduler(self):
        if self._scheduler is None:
            return
        return self._scheduler(self.scheduler, **self.scheduler_kwargs)

    # TODO: implement easy custom starting points
    def train(self, data_loader, epochs):
        '''
        Starts the training of the cINN for a given number of epochs on a given DataLoader.

        Parameters
        ----------
        data_loader : DataLoader
            PyTorch DataLoader which contains the 1D training data in the first axis 
            and the corresponding 1D conditional data in the second axis.
        epochs : int
            Number of epochs the network should be trained.

        '''
        if not isinstance(data_loader, torch.utils.data.DataLoader):
            raise TypeError('Please load data via a PyTorch DataLoader\n"https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler"')
        if self.optimizer is None:
            print('No optimizer set - you must specify an optimizer before training')
            return 

        device = 'cuda' if torch.cuda.is_available() and self.cuda else 'cpu'
        self.cond_net = self.cond_net.to(device)
        self.cinn = self.cinn.to(device)
        
        # adapt the weights of the BatchNorm layers during training.
        self.cond_net.train()
        self.cinn.train()

        try:
            for epoch in tqdm(range(epochs)):
                for i, (x, c) in tqdm(enumerate(data_loader), leave=False):
                    self.optimizer.zero_grad()
                    c = c.unsqueeze(dim=1)
                    # sample data from the moons distribution
                    x, c = x.to(device), c.to(device)
                    c = self.cond_net(c)
                    c = [c.squeeze() for _ in range(self.num_of_blocks)]
                    # pass to INN and get transformed variable z and log Jacobian determinant
                    z, log_jac_det = self.cinn(x, c=c)
                    # calculate the negative log-likelihood of the model with a standard normal prior
                    loss = 0.5*torch.sum(z**2, 1) - log_jac_det
                    loss = loss.mean() / self.input_dim
                    # losses.append(loss.item())
                    # backpropagate and update the weights
                    loss.backward()
                    self.optimizer.step()
                if self.scheduler:    
                    self.scheduler.step()
        except KeyboardInterrupt:
            return

    def eval_forward(self, data):
        '''
        Evaluates the OsCINN in forward direction. 

        During forward evaluation, no gradients or BatchNorm moments are updated. 

        Parameters
        ----------
        data : Tensor, Dataset
            PyTorch Tensor or Dataset which contains the 1D input data in the 
            first axis and the corresponding 1D conditional data in the second axis.
            (Shape [x, c])

        Returns
        -------
        z : Tensor
            latent space values of the inputs
        log_jac_det : Tensor
            log Jacobian Determinant of the transformed latent space values. 

        Notes
        -----
        Due to the implementation of the cond_resnet, a single input 
        (shape [input_dim]) can't be processed. The input is therefore tiled to 
        shape [2,1,input_dim].
        '''
        # TODO: automatically deal with only one sample - done
        # TODO: test implementation
        x, c = data
        device = 'cuda' if torch.cuda.is_available() and self.cuda else 'cpu'
        self.cond_net = self.cond_net.to(device)
        self.cinn = self.cinn.to(device)
        x, c = x.to(device), c.to(device)
        
        # Set the running mean of BatchNorm layers to eval mode
        self.cond_net.eval()
        self.cinn.eval()

        if x.dim() > 1:
            is_batch = True
        else:
            is_batch = False
        
        with torch.no_grad():
            c = c.unsqueeze(dim=1) if is_batch else tile(c.unsqueeze(dim=0).unsqueeze(dim=0),0,2)
            c = self.cond_net(c)
            c = [c.squeeze() for i in range(self.num_of_blocks)]
            # pass to INN and get transformed variable z and log Jacobian determinant
            x = x if is_batch else tile(x.unsqueeze(dim=0),0,2)
            z, log_jac_det = self.cinn(x, c=c, rev=False)
        return (z, log_jac_det) if is_batch else (z[0], log_jac_det[0])

    def eval_inverse(self, data):
        '''
        Evaluates the OsCINN in inverse direction. 

        During inverse evaluation, no gradients or BatchNorm moments are updated. 

        Parameters
        ----------
        data : Tensor, Dataset
            PyTorch Tensor or Dataset which contains the 1D latent space values in the 
            first axis and the corresponding 1D conditional data in the second axis.
            (Shape [z, c])

        Returns
        -------
        x_hat : Tensor
            Predicted values for the input
        log_jac_det : Tensor
            log Jacobian Determinant of the transformed latent space values. 

        Notes
        -----
        Due to the implementation of the cond_resnet, a single input 
        (shape [input_dim]) can't be processed. The input is therefore tiled to 
        shape [2,1,input_dim].
        '''
        z, c = data
        device = 'cuda' if torch.cuda.is_available() and self.cuda else 'cpu'
        self.cond_net = self.cond_net.to(device)
        self.cinn = self.cinn.to(device)
        z, c = z.to(device), c.to(device)

        # Set the running mean of BatchNorm layers to eval mode
        self.cond_net.eval()
        self.cinn.eval()
        if z.dim() > 1:
            is_batch = True
        else:
            is_batch = False
        with torch.no_grad():
            c = c.unsqueeze(dim=1) if is_batch else tile(c.unsqueeze(dim=0).unsqueeze(dim=0),0,2)
            c = self.cond_net(c)
            c = [c.squeeze() for i in range(self.num_of_blocks)]
            # pass z to INN and get transformed variable x_hat and log Jacobian determinant
            z = z if is_batch else tile(z.unsqueeze(dim=0),0,2)
            x_hat, log_jac_det = self.cinn(z, c=c, rev=True)
        return (x_hat, log_jac_det) if is_batch else (x_hat[0], log_jac_det[0])

    def save(self, fname, save_optim=True):
        data_dict = {'inn': self.cinn.state_dict(),
                    'cond_net': self.cond_net.state_dict()}
        if save_optim:
            data_dict['optim'] = self._optimizer.state_dict()
        torch.save(data_dict, fname)
            
    def load(self, fname):
        data_dict = torch.load(fname)

        self.cinn.load_state_dict(data_dict['inn'])
        self.cond_net.load_state_dict(data_dict['cond_net'])

        try:
            self._optimizer.load_state_dict(data_dict['optim'])
        except KeyError:
            warnings.warn('No optimizer state saved in network data')


    
if __name__ == '__main__':
    print('run a small test on a cinn with random data to test the functionality')
    oscinn = OsCinn1D(9, 100, 8, cuda=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    c = c2 = torch.randn(50, 1, 100).to(device)
    x = torch.randn(50, 9).to(device)

    c = oscinn.cond_net(c)
    print('condition shape: ', c.shape)
    c = [c.squeeze() for i in range(8)]
    z, jac_z = oscinn.cinn(x, c=c, rev=True)
    print('latent space shape: ', z.shape)

    # using eval_... methods, additional dimensions are added automatically

    print('\nTest eval_forward...')
    z, jac = oscinn.eval_forward([x,c2[:,0]])
    print('latent space shape: ', z.shape)

    z_1D, jac_1D = oscinn.eval_forward([x[0], c2[0,0]])
    print('latent 1d space shape: ', z_1D.shape)

    print('\nTest eval_inverse...')
    x, jac = oscinn.eval_inverse([z,c2[:,0]])
    print('x_hat shape: ', x.shape)

    x_1D, jac_1D = oscinn.eval_inverse([z[0], c2[0,0]])
    print('x_hat 1d shape: ', x_1D.shape)

    print('\nTest training...')
    dataset = torch.utils.data.TensorDataset(x, c2[:,0])
    dataloader = torch.utils.data.DataLoader(dataset,
                                           batch_size=5,
                                           shuffle=True, 
                                           drop_last=False)
    oscinn.optimizer = torch.optim.Adam
    oscinn.optimizer_kwargs = {'lr':0.001}
    print(oscinn.optimizer)
    oscinn.train(dataloader, 2)

    print('oscinn finished successfully ')