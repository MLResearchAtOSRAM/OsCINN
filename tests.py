import torch
from OScINN import OScINN1D


def test():
    print('run a small test on a cinn with random data to test the functionality')
    oscinn = OScINN1D(9, 100, 8, cuda=False)
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