import warnings
from utils.photo_error import mse_
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from PIL import Image
import airsim
from os.path import join as pathjoin
from calculate_flow import FlowNet2Utils
from rtvs_balance import Rtvs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
np.random.seed(0)
import cv2
warnings.filterwarnings("ignore")
from BASNet.basnet_infer import Basnet
import torch
        

def outward_flow():
    const = 1
    xyz = np.fromfunction(lambda i, j, k : (k==0)*const* (i - 192) + (k==1)*const* (j - 256), (384, 512, 2), dtype=float)
    # normalise from -1 to 1 by dividing by 192 and 256
    return xyz

def save_mask(mask, step, rgb_image, img_name):
    
    folder = "/ssd_scratch/shankara/vs_airsim/vs/RTVS/save_mask"
    folder1 = "/ssd_scratch/shankara/vs_airsim/vs/RTVS/save_rgb"
    if not os.path.exists(folder+'/' + img_name):
        os.makedirs(folder+'/' + img_name)
    if not os.path.exists(folder1+'/' + img_name):
        os.makedirs(folder1+'/' + img_name)
      
    # keep_mask = mask < 0.3
    keep_mask = mask/255.
    # print(keep_mask)
    cv2.imwrite("/ssd_scratch/shankara/vs_airsim/vs/RTVS/save_mask/" +  img_name + "/" + "_.%05d.png" % (step), keep_mask * 255)
    cv2.imwrite("/ssd_scratch/shankara/vs_airsim/vs/RTVS/save_rgb/" +  img_name + "/" + "_.%05d.png" % (step), rgb_image)
    # cv2.imwrite("/ssd_scratch/shankara/vs_airsim/vs/RTVS/save_mask" + "/goal.png", img_name)
    # print(" saved mask")

    return keep_mask.reshape(384,512,1)


def main():
    global client, radial_flow, rtvs, target_image
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    
    client.armDisarm(True)
    client.takeoffAsync().join()
    
    target_image = "building1"
    
    

# # buildin 1
    # pos_x = 15
    # pos_y = 0
    # pos_z = -15
    # vel = 3
    # angle_z = 270
    
    # client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), int(vel), 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join() #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
    # client.hoverAsync().join()
    # client.rotateToYawAsync(int(angle_z),5,1).join()

    # client.setAngleLevelControllerGains(angle_level_gains=airsim.AngleLevelControllerGains(roll_gains = airsim.PIDGains(1.5, 0, 0),
    #                    pitch_gains = airsim.PIDGains(1.5, 0, 0),
    #                    yaw_gains = airsim.PIDGains(1.5, 0, 0)))
   
      
# building 3
    pos_x = 5
    pos_y = -30
    pos_z = -15
    vel = 3
    angle_z = 180

    client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), int(vel), 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join() #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
    client.hoverAsync().join()
    client.rotateToYawAsync(int(angle_z),5,1).join()
    
############

    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
    # reshape array to 4 channel image array H X W X 4
    img_src = img1d.reshape(response.height, response.width, 3)
    
    pre_img_src = img_src
        
    
    rtvs = Rtvs()
    
       
    radial_flow = outward_flow()
       
    plan(pre_img_src)
    
def is_obstacle_there(mask):
    print(mask.sum())
    return mask.sum() > 60000 #80000
    # return mask[128:384, 96:288].sum() > 10000
    
def quaternion_to_euler_angle_vectorized1(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2

    t2 = np.where(t2<-1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z 
    
def plan(pre_img_src):
    global client, radial_flow, rtvs, target_image
    model = Basnet(weights_path='/ssd_scratch/shankara/vs_airsim/vs/RTVS/BASNet/saved_models/basnet_r18/299basnet_bsi_resnet18.pth')
    model.load_model()
    goal =  [-50, -30, -15, 1] #[20, -60, -15, 1]  #    #[-50, -35, -15, 1]# #[20, -75, -15, 0.5]
    pos_x = goal[0]
    pos_y = goal[1]
    pos_z = goal[2]
    vel_goal = goal[3]
   
    final_position = np.array([pos_x, pos_y, pos_z])
   
    step = 1
    count = 0
    old_obthere = False
    
    
    
    while True:
        
       
        
        
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        
        # reshape array to 4 channel image array H X W X 4
        img_src = img1d.reshape(response.height, response.width, 3)
        
        
        mask = model.infer(img_src, show=True)
        mask_orig = mask
        print(mask.shape)
        
        # gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  
        grayImage = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        print("shape", grayImage.shape)  
        
        gray_mask = grayImage
            
               
        mask = save_mask(gray_mask, step, img_src, target_image )

        ct=1
        f12 = (radial_flow*mask)[::ct, ::ct]
        vel = rtvs.get_vel(img_src, pre_img_src, f12, target_image, step, mask)
        
        
        real_orient = client.simGetGroundTruthKinematics().orientation
        
        # quat = np.array((real_orient.w_val, real_orient.x_val, real_orient.y_val,real_orient.z_val))
        euler = quaternion_to_euler_angle_vectorized1(real_orient.w_val, real_orient.x_val, real_orient.y_val,real_orient.z_val)
        
        # if euler[2] > 180:
        euler_z = euler[2] 
        theta = np.deg2rad(euler_z)
        print("theta", theta)
        theta = theta + (np.pi)
        c, s = np.cos(theta), np.sin(theta)
        
        # R_y = np.array([[c], [s]])
        # vel = np.array([[vel[0], vel[1]] ])
        
        vel_x = vel[0]*s
        vel_y = vel[0]*c
        
        
        vel_x = -vel[0] * s
        vel_y = vel[0]*c
        
        
        # R_y = np.array([[c, 0, s], [0, 1, 0],  [-s, 1, c]])
        # c, s = np.cos(theta), np.sin(theta)
        # R_z = np.array([[c, -s, 0], [s, c, 0],  [0, 0, 1]])
        # for vel in [vel[0], vel[1]]:
                         
        
        #     vel = np.array([[vel], [0], [0]])
        # # vel_mat = np.array((vel[0]), (vel[1]))
        # vel_rotated = np.dot(R_y, vel)
        
        #     theta = theta - (np.pi/2)
            
        #     vel_rotated_z = np.dot(R_z, vel_rotated)
        #     vel_.append(vel_rotated_z)
        #     print(vel_)
            
        # vel= [vel_rotated[1], vel_rotated[0]]
        
        # print(vel)
        # for  i in range(len(vel)):
        #     if vel[i] < 0:
        #         vel[i] = 0
        # print(vel)
        # print("algo time: ", time.time() - stime)
        # vel = []
        # vel = vel/np.linalg.norm(vel_, axis = 1)

        
        # print(vel)
        dt = 0.1
        threshold = 1
        # print("Velocity = ", vel)

        obthere = is_obstacle_there(mask)
        obthere1 = obthere
        if obthere == True and old_obthere == False and count > threshold:
            count = 0
        if count < threshold :
            obthere1 = True
            
        old_obthere = obthere
        
        # obthere = False
        
        real_pos = client.simGetGroundTruthKinematics().position #x_val
        real_position = np.array([real_pos.x_val, real_pos.y_val, real_pos.z_val])
        
        distance = np.linalg.norm(real_position - final_position)
        print("distance", distance)
        # print(vel_rotated)
        print("using vel along x as ", vel_x , vel_y)
        if obthere:
            print("obstacle!!!!!!!")
            count +=1
            # vel[1].item()
            client.moveByVelocityAsync(vel[0], -vel[2], vel[1] , 5)
            # client.moveByVelocityAsync( vel_x,vel_y, 0 , 5)
            # client.rotateByYawRateAsync(-vel[4].item(), 10)vel[1].item()
        
        # elif distance < 15:
        #     client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), vel_goal, 3e+38 , drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)) #airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(False, 0)) #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
            
     
        
        else :
           
                       
            # vel1 = np.array([pos_x - x, pos_y - y, pos_z - z])
            # vel1 = vel1/np.linalg.norm(vel1)
            # vel_obs = np.array([vel[0].item(), -vel[2].item(), vel[1].item()] )
            # vel_ovr = 0.5*vel1 + 0.5*vel_obs
            # print(vel_ovr)
            client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), vel_goal, 3e+38 ,  drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)) #yaw_mode=airsim.YawMode(True,0.1) ) #airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(False, 0)) #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
            # client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), vel_goal, 3e+38 , drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(True,0.1) ) #airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(False, 0)) #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
           
            # client.moveByVelocityAsync(vel_ovr[0], vel_ovr[1], 0 , 5)
                
        pre_img_src = img_src # previous time step image
        
        step = step + 1
    
if __name__ == '__main__':
    main()
