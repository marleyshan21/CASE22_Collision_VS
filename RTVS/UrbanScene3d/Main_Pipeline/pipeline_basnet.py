import sys
sys.path.append("../..")


import warnings
import numpy as np
import time
import os
import sys
from PIL import Image
import airsim
from os.path import join as pathjoin
from calculate_flow import FlowNet2Utils
from rtvs import Rtvs
from dcem_model import Model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
np.random.seed(0)
import cv2
warnings.filterwarnings("ignore")
from BASNet.basnet_infer import Basnet
import torch
        

def outward_flow():
    const = 1
    xyz = np.fromfunction(lambda i, j, k : (k==0)*const* (i - 192) + (k==1)*const* (j - 256), (384, 512, 2), dtype=float)
    # xyz_1 = np.fromfunction(lambda i, j, k : (k==0)*(i<192)*(j<256)*const* (-i) 
    #                     + (k==0)*(i>192)*(j<256)*const* (384-i)
    #                     + (k==0)*(i<192)*(j>256)*const* (-i) 
    #                     + (k==0)*(i>192)*(j>256)*const* (384 - i)
    #                     + (k==1)*(i<192)*(j<256)*const* (-j)
    #                     + (k==1)*(i<192)*(j>256)*const* (512-j)
    #                     + (k==1)*(i>192)*(j>256)*const* (-j)
    #                     + (k==1)*(i>192)*(j<256)*const* (512-j),
    #                     (384, 512, 2)
    #                     , dtype=float)
    # xyz = (xyz + xyz_1)/2.0
    # normalise from -1 to 1 by dividing by 192 and 256
    return xyz

def save_mask(mask, step, rgb_image, img_name):
    
    folder = "/scratch/shankara/IROS22_Collision_VS/RTVS/UrbanScene3d/Main_Pipeline/save_mask"
    folder1 = "/scratch/shankara/IROS22_Collision_VS/RTVS/UrbanScene3d/Main_Pipeline/save_rgb"
    if not os.path.exists(folder+'/' + img_name):
        os.makedirs(folder+'/' + img_name)
    if not os.path.exists(folder1+'/' + img_name):
        os.makedirs(folder1+'/' + img_name)
      
    
    keep_mask = mask/255.
    
    
    cv2.imwrite("/scratch/shankara/IROS22_Collision_VS/RTVS/UrbanScene3d/Main_Pipeline/save_mask/" +  img_name + "/" + "_.%05d.png" % (step), keep_mask * 255)
    cv2.imwrite("/scratch/shankara/IROS22_Collision_VS/RTVS/UrbanScene3d/Main_Pipeline/save_rgb/" +  img_name + "/" + "_.%05d.png" % (step), rgb_image)
    

    return keep_mask.reshape(384,512,1)


def main():
    
    global client, radial_flow, rtvs, env, z, hard
    
    env = sys.argv[1] 
    z = sys.argv[2]
    hard = sys.argv[3]
    
    client = airsim.MultirotorClient()
    # client = airsim.MultirotorClient(ip = "10.2.36.227")
    client.confirmConnection()
    client.enableApiControl(True)
    
    client.armDisarm(True)
    client.takeoffAsync().join()
    
     
    if env == "square":
        pos_x = -60
        pos_y = 30
        pos_z = 40
        vel = 10
        angle_z = 270
        
        client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), int(vel), 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join() #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
        client.hoverAsync().join()
        client.rotateToYawAsync(int(angle_z),5,1).join()
        
    if env == "shen":
        pos_x = 15
        pos_y = 20
        pos_z = -10
        vel = 10
        angle_z = 45

        client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), int(vel), 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join() #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
        client.hoverAsync().join()
        client.rotateToYawAsync(int(angle_z),5,1).join()
        
        
    
    if env == "sciart":

        pos_x = 10
        pos_y = -10
        pos_z = 0
        vel = 10
        angle_z = 135

        client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), int(vel), 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join() #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
        client.hoverAsync().join()
        client.rotateToYawAsync(int(angle_z),5,1).join()
    
    if env == "ying":
        pos_x = -5
        pos_y = 20
        pos_z = -10
        vel = 10
        angle_z = 90

        client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), int(vel), 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join() #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
        client.hoverAsync().join()
        client.rotateToYawAsync(int(angle_z),5,1).join()
        
    
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
    img_src = img1d.reshape(response.height, response.width, 3)
    img_src = cv2.resize(img_src, (512,384))
    pre_img_src = img_src
        
    rtvs = Rtvs()
    
       
    radial_flow = outward_flow()
       
    plan(pre_img_src)
    
def is_obstacle_there(mask):
    print(mask.sum())
    return mask.sum() > 70000
    
def plan(pre_img_src):
    global client, radial_flow, rtvs, env, z, hard
    # model = Basnet(weights_path='/scratch/shankara/IROS22_Collision_VS/RTVS/BASNet_retrained/saved_models/basnet_r18/299basnet_bsi_resnet18.pth')
    model = Basnet(weights_path='/scratch/shankara/Basenet_retrained/saved_models/basnet_r18/249basnet_bsi_resnet18.pth')
    model.load_model()
    
    if env == "square":
        if hard == "True":
            goal = [-100, -125, 40, 1] 
        else:
            #easy
            goal = [-135, -65, 40, 1]
    elif env == "shen":
        if hard == "True":
            goal = [130, 150, -10, 1] 
        else:
            #easy
            goal = [50, 120, 0, 1]
    
    elif env == "sciart":
        if hard == "True":
            goal = [-70, 90, 0, 1] 
        else:
            #easy
            goal = [-70, 30, 0, 1]
        
    elif env == "ying":
        if hard == "True":
            goal = [0, 110, -20, 1] 
        else:
            #easy
            goal = [20, 70, -10, 1]
    
       
    pos_x = goal[0]
    pos_y = goal[1]
    pos_z = goal[2]
    vel_goal = goal[3]
   
    final_position = np.array([pos_x, pos_y, pos_z])
   
    step = 1
     
        
    while True:
        
       
        
        
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        
        # reshape array to 4 channel image array H X W X 4
        img_src = img1d.reshape(response.height, response.width, 3)
        img_src = cv2.resize(img_src, (512,384))
        # img_src = img_src.reshape(384, , 3)
        
        # print(response.height, response.width)
        mask = model.infer(img_src, show=True)
                
        grayImage = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
     
        mask = save_mask(grayImage, step, img_src, env )

        ct=1
        f12 = (radial_flow*mask)[::ct, ::ct]
        vel = rtvs.get_vel(img_src, pre_img_src, f12, env, step, mask)
              
       
        # print("Velocity = ", vel)
        if step % 10 == 0:
            rtvs.vs_lstm = Model().to(device="cuda:0")

        obthere = is_obstacle_there(mask)
              
        real_pos = client.simGetGroundTruthKinematics().position #x_val
        real_position = np.array([real_pos.x_val, real_pos.y_val, real_pos.z_val])
        
        distance = np.linalg.norm(real_position - final_position)
        print("distance", distance)
        print(vel)
        if obthere:
            print("obstacle!!!!!!!")
           
            if z == '1':
                client.moveByVelocityAsync(vel[0].item(), -vel[2].item(), vel[1].item() , 5)
                
            elif z == '0':
                client.moveByVelocityAsync(vel[0].item(), -vel[2].item(), vel[1].item() , 5)
                # client.moveByVelocityBodyFrameAsync(0, 1, 0 , 5, drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, 0))
                # client.moveByVelocityBodyFrameAsync(-vel[2].item(), vel[0].item(), 0 , 5, drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, 0))
         
              
        else :
           
            client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), vel_goal, 3e+38 ,  drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)) #yaw_mode=airsim.YawMode(True,0.1) ) #airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(False, 0)) #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
                            
        pre_img_src = img_src # previous time step image
        
        step = step + 1
    
if __name__ == '__main__':
    main()
