import warnings
import numpy as np
from dcem_model import Model
from calculate_flow import FlowNet2Utils
import torch
import os
import cv2
import flowiz as fz
from cem import CEM


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
np.random.seed(0)
warnings.filterwarnings("ignore")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)


class Rtvs:
    '''
    Code for RTVS: Real-Time Visual Servoing, IROS 2021
    '''
    def __init__(self,
                 ct=1,
                 horizon=5,
                 LR=0.001,
                 iterations=5,
                 ):
        '''
        img_goal: RGB array for final pose
        ct = image downsampling parameter (high ct => faster but less accurate)
        LR = learning rate of NN
        iterations = iterations to train NN (high value => slower but more accurate)
        horizon = MPC horizon
        '''
        # self.img_goal = img_goal
        self.horizon = horizon
        self.iterations = iterations
        self.ct = ct
        self.flow_utils = FlowNet2Utils()
        self.vs_lstm = Model().to(device="cuda:0")
        self.optimiser = torch.optim.Adam(self.vs_lstm.parameters(),
                                          lr=LR, betas=(0.93, 0.999))
        self.loss_fn = torch.nn.MSELoss(size_average=False)
        # self.loss_fn = torch.nn.L1Loss(size_average=False)
        self.cem = CEM()



    def get_vel(self, img_src, pre_img_src, f12, img_name, step, mask, depth =None):
        '''
            img_src = current RGB camera image
            prev_img_src = previous RGB camera image 
                        (to be used for depth estimation using flowdepth)
        '''
        # self.flow_utils = FlowNet2Utils()
        # img_goal = self.img_goal
        # flow_utils = self.flow_utils
        # vs_lstm = self.vs_lstm
        # loss_fn = self.loss_fn
        # optimiser = self.optimiser
        ct = self.ct
        # if step %10 ==0:
        #     del self.vs_lstm
        #     self.vs_lstm = Model().to(device="cuda:0")

        # photo_error_val = mse_(img_src, img_goal)
        # if photo_error_val < 6000 and photo_error_val > 3600:
        #     self.horizon = 10*(photo_error_val/6000)
        # elif photo_error_val < 3000:
        #     self.horizon = 6

        # f12 = flow_utils.flow_calculate(img_src, img_goal)[::ct, ::ct]
        
        # f12 = f12
        # flow_depth_proxy = self.flow_utils.flow_calculate(
            # img_src, pre_img_src).astype('float64')

        # flow_depth_proxy_2 = np.abs(flow_depth_proxy[:, :, 1])
        # flow_depth = np.linalg.norm(flow_depth_proxy[::ct, ::ct], axis=2) 
        
        # flow_depth = flow_depth.astype('float64')
        if depth is None:
            fd = 0.6*(1/(1+np.exp(-1/flow_depth)))
        else :
            fd = depth
        
        Cy, Cx = fd.shape[1]/2, fd.shape[0]/2

        # fd[:, :] =1
        vel, Lsx, Lsy = get_interaction_data_1(
            fd, ct, Cy, Cx)
        # print("Mask Shape : ", mask.shape)
        # print("Lsx shape : ", Lsx.shape)
        # print("Lsy shape : ", Lsy.shape)
        # print("mask shape : ", (mask==1).shape)
        Lsx = Lsx * mask
        Lsy = Lsy * mask
        # Lsx = Lsx[:, :, :3]
        # Lsy = Lsy[:, :, :3]

        # fd = (1/(1+np.exp(-1/flow_depth_proxy_2)) - 0.5 )*2
        # cv2.imwrite("/home/rrc/vs_airsim/vs/RTVS/save_depth/" +  img_name + "/" + "_.%05d.png" % (step), fd*255)
        # cv2.imwrite("/home/rrc/vs_airsim/vs/RTVS/save_depth_mask/" +  img_name + "/" + "_.%05d.png" % (step), (fd>0.1)*255)

        Lsx = torch.tensor(Lsx, dtype=torch.float32).to(device="cuda:0")
        Lsy = torch.tensor(Lsy, dtype=torch.float32).to(device="cuda:0")
        f12 = torch.tensor(f12, dtype=torch.float32).to(device="cuda:0")
        f12 = self.vs_lstm.pooling(f12.permute(2, 0, 1).unsqueeze(dim=0))

        for itr in range(self.iterations):
            self.vs_lstm.v_interm = []
            self.vs_lstm.f_interm = []
            self.vs_lstm.mean_interm = []

            self.vs_lstm.zero_grad()
            f_hat = self.vs_lstm.forward(vel, Lsx, Lsy, self.horizon, f12)
            norm_fhat = torch.norm(f_hat, dim=1).view(16, 1, 384, 512)
            f_hat = f_hat/(norm_fhat + 0.01)
            loss = self.loss_fn(f_hat, f12)
            print("MSE:", str(np.sqrt(loss.item())))
            loss.backward(retain_graph=True)
            # del loss
            self.optimiser.step()
        
        print('fhat shape : ', f_hat.shape)
        print('max fhat : ', torch.max(f_hat))
        print('min fhat : ', torch.min(f_hat))
        f_hat = f_hat.detach().cpu()[0].permute(1, 2, 0).reshape(384, 512, 2).numpy()
        f_hat[:, :, 0] = 0
        flow = fz.convert_from_flow(f_hat)
        cv2.imwrite('/ssd_scratch/shankara/IROS22_Collision_VS/RTVS/Main_Pipeline/save_rgb/flow_'+str(step)+'.png', flow)

        # del flow_depth_proxy
        #Do not accumulate flow and velocity at train time
        self.vs_lstm.v_interm = []
        self.vs_lstm.f_interm = []
        self.vs_lstm.mean_interm = []

        f_hat = self.vs_lstm.forward(vel, Lsx, Lsy, -self.horizon,
                                f12.to(torch.device('cuda:0')))
        vel = self.vs_lstm.v_interm[0]
        del f12
        del Lsx, Lsy
        
        
        
        return vel


def get_interaction_data(d1, ct, Cy, Cx):
    ky = Cy
    kx = Cx
    xyz = np.zeros([d1.shape[0], d1.shape[1], 3])
    Lsx = np.zeros([d1.shape[0], d1.shape[1], 6])
    Lsy = np.zeros([d1.shape[0], d1.shape[1], 6])

    med = np.median(d1)
    xyz = np.fromfunction(lambda i, j, k: 0.5*(k-1)*(k-2)*(ct*j-float(Cx))/float(kx) - k*(k-2)*(ct*i-float(Cy))/float(ky) + 0.5*k*(
        k-1)*((d1[i.astype(int), j.astype(int)] == 0)*med + d1[i.astype(int), j.astype(int)]), (d1.shape[0], d1.shape[1], 3), dtype=float)

    Lsx = np.fromfunction(lambda i, j, k: (k == 0).astype(int) * -1/xyz[i.astype(int), j.astype(int), 2] + (k == 2).astype(int) * xyz[i.astype(int), j.astype(int), 0]/xyz[i.astype(int), j.astype(int), 2] + (k == 3).astype(int) * xyz[i.astype(
        int), j.astype(int), 0]*xyz[i.astype(int), j.astype(int), 1] + (k == 4).astype(int)*(-(1+xyz[i.astype(int), j.astype(int), 0]**2)) + (k == 5).astype(int)*xyz[i.astype(int), j.astype(int), 1], (d1.shape[0], d1.shape[1], 6), dtype=float)

    Lsy = np.fromfunction(lambda i, j, k: (k == 1).astype(int) * -1/xyz[i.astype(int), j.astype(int), 2] + (k == 2).astype(int) * xyz[i.astype(int), j.astype(int), 1]/xyz[i.astype(int), j.astype(int), 2] + (k == 3).astype(int) * (1+xyz[i.astype(
        int), j.astype(int), 1]**2) + (k == 4).astype(int)*-xyz[i.astype(int), j.astype(int), 0]*xyz[i.astype(int), j.astype(int), 1] + (k == 5).astype(int) * -xyz[i.astype(int), j.astype(int), 0], (d1.shape[0], d1.shape[1], 6), dtype=float)

    return None, Lsx, Lsy



def get_interaction_data_1(d1, ct, Cy, Cx):
    ky = Cy
    kx = Cx
    xyz = np.zeros([d1.shape[0], d1.shape[1], 3])
    Lsx = np.zeros([d1.shape[0], d1.shape[1], 6])
    Lsy = np.zeros([d1.shape[0], d1.shape[1], 6])
    Zi = 1/(d1*100+1)
    Lsx = np.fromfunction(lambda i, j, k: (k == 0).astype(int)*-1*Zi[i.astype(int), j.astype(int)] 
                          + (k == 2).astype(int) *((i-Cx)/(Cx)) * Zi[i.astype(int), j.astype(int)] 
                          + (k == 3).astype(int) *((j-Cy)/Cy) * ((i-Cx)/Cx)
                          + (k == 4).astype(int) *-1*(1+((i-Cx)/Cx)**2)
                          + (k == 5).astype(int) * (j-Cy)/Cy
                          , (d1.shape[0], d1.shape[1], 6), dtype=float)
    
    Lsy = np.fromfunction(lambda i, j, k: (k == 1).astype(int)*-1*Zi[i.astype(int), j.astype(int)] 
                          + (k == 2).astype(int) *((j-Cy)/(Cy) )*Zi[i.astype(int), j.astype(int)]
                          + (k == 3).astype(int) *((1+(i-Cx)/Cx)**2)
                          + (k == 4).astype(int) *((1+(j-Cy)/Cy)**2)
                          + (k == 5).astype(int) * (i-Cx)/Cx
                          , (d1.shape[0], d1.shape[1], 6), dtype=float)
    
    
    return None, Lsx, Lsy
