from cv2 import reduce
import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F

class CEM():
    def __init__(self, N=20, K=10) -> None:
        init_mu = 0.0
        init_sigma = 2
        self.n_sample = N
        self.n_elite = K

        ### Initialise Mu and Sigma
        # we change 1,1 to 1,6
        self.mu = init_mu * torch.ones((1,6), requires_grad=True).cuda()
        self.sigma = init_sigma * torch.ones((1,6), requires_grad=True).cuda()
        self.dist = Normal(self.mu, self.sigma)
        self.pooling = torch.nn.AvgPool2d(kernel_size=1, stride=1)
        
    def forward(self, vel, Lsx, Lsy, horizon, f12):
        vel = self.dist.rsample((self.n_sample,)).cuda()
        vel = vel.view(self.n_sample, 1, 1, 6)
        if horizon < 0 :
            flag = 0
        else :
            flag = 1
        
        vels = vel * horizon
        # if flag == 1:
        #     vels = vel * horizon
        # else :
        #     return self.mu

        
        # vels[:,:,:,:] = 0
        vels[:,:,:,0] = 0
        Lsx = Lsx.view(1, f12.shape[2], f12.shape[3], 6)
        Lsy = Lsy.view(1, f12.shape[2], f12.shape[3], 6)
        
        f_hat = torch.cat((torch.sum(Lsx[:,:,:,1:3]*vels[:,:,:,1:3],-1).unsqueeze(-1) , \
                torch.sum(Lsy[:,:,:,1:3]*vels[:,:,:,1:3],-1).unsqueeze(-1)),-1)
        
        
        f_hat = self.pooling(f_hat.permute(0, 3, 1, 2))
        norm_fhat = torch.norm(f_hat, dim=1).view(self.n_sample, 1, 384, 512)
        # f_hat[:, 1, :, :] = f_hat[:,1, :, :]
        # f_hat = f_hat/(norm_fhat + 0.01)
        
        loss_fn = torch.nn.MSELoss(size_average=False, reduce=False)
        # loss_fn = torch.nn.L1Loss(size_average=False, reduce=False)
        loss = loss_fn(f_hat, f12)
        loss = torch.sum(loss.reshape(self.n_sample, -1), dim =1)
        sorted, indices = torch.sort(loss)
        # print('loss  : ', loss[indices[0]])
        self.mu = torch.mean(vel[indices[:self.n_elite]], dim=0)
        self.var = torch.std(vel[indices[:self.n_elite]], dim=0)**2
        # print("Loss : ", loss)
        self.dist = Normal(self.mu, self.var**0.5)
        
        return vel[indices[0]], f_hat