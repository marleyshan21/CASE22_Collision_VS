import sys
sys.path.append("..")
from calculate_flow import FlowNet2Utils
import numpy as np
from PIL import Image
import math
sign = lambda x: math.copysign(1, x)


def outward_flow():
    const1 = 1/2.0
    const2 = const1
    # xyz = np.fromfunction(lambda i, j, k : (k==0)*const1* (i - 192) + (k==1)*const1* (j - 256), (384, 512, 2), dtype=float)

    # xyz = np.fromfunction(lambda i, j, k : (k==0)*const1* (i - 192) / ((( (i-192)**2)+(j-256)**2)**0.5 +0.01) + (k==1)*const1* (j - 256)/ ((( (i-192)**2)+(j-256)**2)**0.5 +0.01  ), (384, 512, 2), dtype=float)
    xyz  = np.fromfunction(lambda i, j, k : (k==0)*(i<192)*(j<256)*const1* (-i) 
                          + (k==0)*(i>192)*(j<256)*const1* (384-i)
                          + (k==0)*(i<192)*(j>256)*const1* (-i) 
                          + (k==0)*(i>192)*(j>256)*const1* (384 - i)
                          + (k==1)*(i<192)*(j<256)*const2* (-j)
                          + (k==1)*(i<192)*(j>256)*const2* (512-j)
                          + (k==1)*(i>192)*(j>256)*const2* (512-j)
                          + (k==1)*(i>192)*(j<256)*const2* (-j),
                          (384, 512, 2)
                          , dtype=float)
    # xyz = (xyz + xyz_1)
    # # normalise from -1 to 1 by dividing by 192 and 256
    return xyz

class Dfvs:
    '''
        Contains code for "DFVS: Deep Flow Guided Scene Agnostic Image Based Visual Servoing"
        ICRA 2020
    '''
    TRUEDEPTH = 0
    FLOWDEPTH = 1

    def __init__(self, LM_LAMBDA=0.08, LM_MU=0.03, v_max_abs = 1):
        '''
            des_img_path:   path to destination pose image
            LM_LAMBDA:      scales the velocity
            LM_MU:          LM method parameter
            v_max_abs:      maximum value of any element velocity vector
                            (set inifinity for unbounded velocity)
        '''
        self.v_max_abs = v_max_abs
        # self.des_img = np.array(Image.open(des_img_path).convert("RGB"))
        self.LM_LAMBDA = LM_LAMBDA
        self.LM_MU = LM_MU
        self.flow_utils = FlowNet2Utils()
        self.set_interaction_utils()

    @property
    def img_shape(self, img_src):
        return img_src.shape[:2]

    def get_next_velocity(self, cur_img, prev_img=None, depth=None, mask = None):
        '''
            all parameters should be numpy arrays
            cur_img: current RGB camera image
            prev_img: previous RGB camera image (to be used for depth estimation using flowdepth)
            depth: depth sensor readings (prev_img is not used if depth is available)
        '''
        assert not(prev_img is None and depth is None)

        # flow_error = get_flow(self.des_img, cur_img)
        # flow_error = flow_error.transpose(1, 0, 2).flatten()

        if depth is not None:
            L = self.get_interaction_mat(depth, Dfvs.TRUEDEPTH, mask=mask)
        else:
            flow_inv_depth = self.flow_utils.flow_calculate(cur_img, prev_img).astype('float64')
            flow_inv_depth = np.linalg.norm(flow_inv_depth, axis=2)
            L = self.get_interaction_mat(flow_inv_depth, Dfvs.FLOWDEPTH, mask)
            print("L shape : ", L.shape)

        flow_error = outward_flow()*mask
        flow_error = flow_error.transpose(1, 0, 2).flatten()
        H = L.T @ L
        vel = -self.LM_LAMBDA * np.linalg.pinv(
            H + self.LM_MU*(H.diagonal())) @ L.T @ flow_error

        vel[vel >1] = 1
        vel[vel<-1] = -1
        # bounding velocity to given range
        # max_v_i = np.abs(vel).max()
        # if max_v_i > self.v_max_abs:
        #     vel = vel / max_v_i

        return vel

    def set_interaction_utils(self):
        row_cnt, col_cnt = 384, 512
        # v~y , u~x
        px = col_cnt // 2
        u_0 = col_cnt // 2
        py = row_cnt // 2
        v_0 = row_cnt // 2
        u, v = np.indices((col_cnt, row_cnt))
        x = (u - u_0)/px
        y = (v - v_0)/py

        x = x.flatten()
        y = y.flatten()

        self.inter_utils = {
            "x": x,
            "y": y,
            "zero": np.zeros_like(x)
        }

    def get_interaction_mat(self, Z, mode, mask):
        # assert Z.shape == self.img_shape, (Z.shape, (512, 384))
        if mode == self.TRUEDEPTH:
            Zi = 10/(Z + 1)
        elif mode == self.FLOWDEPTH:
            Zi = Z + 1
        Zi = Zi.flatten()
        mask1 = mask.flatten()

        x = self.inter_utils["x"]
        y = self.inter_utils["y"]
        zero = self.inter_utils["zero"]

        L = np.array([
            [-Zi*mask1, zero*mask1, x*Zi*mask1, x*y*mask1, -(1 + x**2)*mask1, y*mask1],
            [zero*mask1, -Zi*mask1, y*Zi*mask1, (1 + y**2)*mask1, -x*y*mask1, -x*mask1],
        ])
        mask = mask.flatten().reshape(1, 1, 384*512)
        # L = L*mask
        print("Mask shape : ", mask.shape)
        print("L shape : ", L.shape)

        assert L.shape == (2, 6, x.shape[0])
        L = L.transpose(2, 0, 1).reshape(-1, 6)

        return L
