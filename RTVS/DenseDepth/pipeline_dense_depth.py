import sys
sys.path.append("..")

import warnings
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from PIL import Image
import airsim
from os.path import join as pathjoin
from calculate_flow import FlowNet2Utils
from rtvs import Rtvs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
np.random.seed(0)
import cv2
warnings.filterwarnings("ignore")
import torch
from depth_net.load import DepthNorm,scale_up, load_depth_net_model       

def outward_flow():
    const = 1
    xyz = np.fromfunction(lambda i, j, k : (k==0)*const* (i - 192) + (k==1)*const* (j - 256), (384, 512, 2), dtype=float)
    # normalise from -1 to 1 by dividing by 192 and 256
    return xyz

def save_mask(depth_img, step, rgb_image, img_name):
        
    folder = "/ssd_scratch/shankara/VS_obs/RTVS/DenseDepth/save_mask"
    folder1 = "/ssd_scratch/shankara/VS_obs/RTVS/DenseDepth/save_rgb"
    folder2 = "/ssd_scratch/shankara/VS_obs/RTVS/DenseDepth/save_depth"
    
    
    if not os.path.exists(folder+'/' + img_name):
        os.makedirs(folder+'/' + img_name)
    if not os.path.exists(folder1+'/' + img_name):
        os.makedirs(folder1+'/' + img_name)
    if not os.path.exists(folder2+'/' + img_name):
        os.makedirs(folder2+'/' + img_name)
    
    
    print(depth_img) 
    
    depth_img1 = depth_img
    print("max" , np.max(depth_img))
    keep_mask = depth_img1/100 < 0.2
    
    cv2.imwrite("/ssd_scratch/shankara/VS_obs/RTVS/DenseDepth/save_depth/" +  img_name + "/" + "_.%05d.png" % (step), depth_img )
   
    cv2.imwrite("/ssd_scratch/shankara/VS_obs/RTVS/DenseDepth/save_mask/" +  img_name + "/" + "_.%05d.png" % (step), keep_mask * 255)
    cv2.imwrite("/ssd_scratch/shankara/VS_obs/RTVS/DenseDepth/save_rgb/" +  img_name + "/" + "_.%05d.png" % (step), rgb_image)
    
    
    return keep_mask.reshape(384,512,1)


def main():
    
    global client, radial_flow, rtvs, target, z
    
    target = sys.argv[1]
    z = sys.argv[2]
    
    
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    
    client.armDisarm(True)
    client.takeoffAsync().join()
    
     
    if target == "building1":
        pos_x = 15
        pos_y = 0
        pos_z = -15
        vel = 3
        angle_z = 270
        
        client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), int(vel), 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join() #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
        client.hoverAsync().join()
        client.rotateToYawAsync(int(angle_z),5,1).join()

    if target == "building2":
        pos_x = 0
        pos_y = -75
        pos_z = -15
        vel = 10
        angle_z = 270

        client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), int(vel), 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join() #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
        client.hoverAsync().join()
        client.rotateToYawAsync(int(angle_z),5,1).join()
        
        pos_x = 20
        pos_y = -75
        pos_z = -15
        vel = 3
        angle_z = 270

        client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), int(vel), 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join() #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
        client.hoverAsync().join()
        client.rotateToYawAsync(int(angle_z),5,1).join()
    
    if target == "building3":

        pos_x = 5
        pos_y = -30
        pos_z = -15
        vel = 3
        angle_z = 180

        client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), int(vel), 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join() #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
        client.hoverAsync().join()
        client.rotateToYawAsync(int(angle_z),5,1).join()
    
    if target == "building4":
        pos_x = 0
        pos_y = 10
        pos_z = -15
        vel = 5
        angle_z = 180

        client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), int(vel), 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join() #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
        client.hoverAsync().join()
        client.rotateToYawAsync(int(angle_z),5,1).join()
        
        pos_x = -35
        pos_y = 15
        pos_z = -15
        vel = 10
        angle_z = 180

        client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), int(vel), 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join() #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
        client.hoverAsync().join()
        client.rotateToYawAsync(int(angle_z),5,1).join()


    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
    img_src = img1d.reshape(response.height, response.width, 3)
    pre_img_src = img_src
        
    
    rtvs = Rtvs()
    
       
    radial_flow = outward_flow()
       
    plan(pre_img_src)
    
def is_obstacle_there(mask):
    print(mask.sum())
    return mask.sum() > 55000
    
    
def plan(pre_img_src):
    global client, radial_flow, rtvs, target, z
    
    input_image_tensor, pred_depth_tensor, sess_depth_net = load_depth_net_model()
    if target == "building1":
        goal = [20, -60, -15, 1] # [-50, -30, -15, 1]  #[-50, -35, -15, 1]# #[20, -75, -15, 0.5]
    elif target == "building2":
        goal = [23, -140, -15, 1]
    
    elif target == "building3":
        goal = [-50, -30, -15, 1]
        
    elif target == "building4":
        goal = [-120, 30, -15, 1]
    
       
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
        img_src1 = img_src/255.
        inp_source = np.expand_dims(np.clip(cv2.resize(img_src1, (640, 480), interpolation = cv2.INTER_AREA), 0, 1), axis = 0)

        predicted_depth = sess_depth_net.run(pred_depth_tensor, feed_dict = {input_image_tensor : inp_source})
        pred_depth = np.clip(DepthNorm(predicted_depth, maxDepth=1000), 10, 1000) / 1000
        pred_depth = scale_up(2, pred_depth[:,:,:,0]) * 100.0
        depth_img = np.zeros((480,640,1))
        depth_img[:, :, 0] = pred_depth[0, :, :]

        depth_img = cv2.resize(depth_img, (512, 384), interpolation = cv2.INTER_AREA)
        depth_from_network = depth_img #process it however you process true_depth
        # response, = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False),])
        # depth_img_in_meters = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
        # depth_img_in_meters = depth_img_in_meters.reshape(response.height, response.width, 1)
        # depth_8bit_lerped = np.interp(depth_img_in_meters, (0, 100), (0, 1))
        
        # # depth_8bit_lerped = depth_8bit_lerped.astype('uint8')
        # depth_img = cv2.resize(depth_8bit_lerped, (512, 384))

        
    
        mask = save_mask(depth_from_network, step, img_src, target )

        ct=1
        f12 = (radial_flow*mask)[::ct, ::ct]   
                
        vel = rtvs.get_vel(img_src, pre_img_src, f12, target, step, mask)

        ct=1
        f12 = (radial_flow*mask)[::ct, ::ct]
        vel = rtvs.get_vel(img_src, pre_img_src, f12, target, step, mask)
              
       
        # print("Velocity = ", vel)

        obthere = is_obstacle_there(mask)
              
        real_pos = client.simGetGroundTruthKinematics().position #x_val
        real_position = np.array([real_pos.x_val, real_pos.y_val, real_pos.z_val])
        
        distance = np.linalg.norm(real_position - final_position)
        print("distance", distance)
        
        if obthere:
            print("obstacle!!!!!!!")
           
            if z == '1':
                client.moveByVelocityAsync(vel[0].item(), -vel[2].item(), vel[1].item() , 5)
                
            elif z == '0':
                client.moveByVelocityAsync(vel[0].item(), -vel[2].item(), 0 , 5)
         
        
        
        
        else :
           
            client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), vel_goal, 3e+38 ,  drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)) #yaw_mode=airsim.YawMode(True,0.1) ) #airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(False, 0)) #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
                            
        pre_img_src = img_src # previous time step image
        
        step = step + 1
    
if __name__ == '__main__':
    main()
