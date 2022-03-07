import warnings
import numpy as np
from dcem_model import Model
from calculate_flow import FlowNet2Utils
import torch
import os
# from utils.photo_error import mse_
import cv2


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
                 horizon=10,
                 LR=0.005,
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



    def get_vel(self, img_src, pre_img_src, f12, img_name, step, mask):
        '''
            img_src = current RGB camera image
            prev_img_src = previous RGB camera image 
                        (to be used for depth estimation using flowdepth)
        '''
       
        ct = self.ct
       
        flow_depth_proxy = self.flow_utils.flow_calculate(
            img_src, pre_img_src).astype('float64')
        # flow_depth_proxy = f12
        print(flow_depth_proxy.shape)
        left_flow = flow_depth_proxy[: , : 256 , :]
        right_flow =  flow_depth_proxy[: , 256 :  , :]
        up_flow = flow_depth_proxy[:192,  :  , :]
        down_flow = flow_depth_proxy[192:,  :  , :]
        
        
        left_norm = np.sum(np.linalg.norm(left_flow, axis = 2, ord=1))
        right_norm = np.sum(np.linalg.norm(right_flow, axis = 2, ord=1))
        
        up_norm = np.sum(np.linalg.norm(up_flow, axis = 1))
        down_norm = np.sum(np.linalg.norm(down_flow, axis = 1))
        
        
        # print("left norm shape", left_norm.shape)
        # print("left flow shape ", left_flow.shape)
        
        dFlow = (left_norm - right_norm)/(left_norm + right_norm)
        print("DFLOW = ", dFlow)
        
        print("(Left, Right)", left_norm, right_norm)
        k = 5.
        yaw = dFlow * k
        print("yaw", yaw)
        vel = [0.0,0,0.75,0]
        
        
        
        
        
                   
        if up_norm > down_norm:
            vel[0] = 1
        else:
            vel[0] = -1
            
        vel[3] = yaw

        
        
        
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




