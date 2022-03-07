import sys
sys.path.append("../..")

import json
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
from dcem_model import Model
# from rtvs import Rtvs
from rtvs_balance import Rtvs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
np.random.seed(0)
import cv2
warnings.filterwarnings("ignore")
from BASNet.basnet_infer import Basnet
import torch
import math, time
# from dfvs import Dfvs
sign = lambda x: math.copysign(1, x)

import flowiz as fz

def outward_flow():
    const = 1/100.
    const1 = 1.
    # xyz = np.fromfunction(lambda i, j, k : (k==0)*const* (i - 192) + (k==1)*const* (j - 256), (384, 512, 2), dtype=float)
    # xyz = np.fromfunction(lambda i, j, k :     (k==0)*const1* (i - 192) / ((( (i-192)**2)+(j-256)**2)**0.5 +0.01) + (k==1)*const1* (j - 256)/ ((( (i-192)**2)+(j-256)**2)**0.5 +0.01  ), (384, 512, 2), dtype=float)
    xyz = np.fromfunction(lambda i, j, k : (k==0)*const1* (i - 192) / ((( (i-192)**2)+((j-256))**2)**0.5 +0.01) + 10*(k==1)*const1* ((j - 256))/ ((( (i-192)**2)+((j-256))**2)**0.5 +0.01  ), (384, 512, 2), dtype=float)
    
    
    return xyz

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

def save_mask(mask, step, rgb_image, img_name):
    
    folder = "./logs/" +img_name + "/save_mask"
    folder1 = "./logs/" +img_name + "/save_rgb"
    if not os.path.exists(folder+'/' + img_name):
        os.makedirs(folder+'/' + img_name)
    if not os.path.exists(folder1+'/' + img_name):
        os.makedirs(folder1+'/' + img_name)
      
    
    keep_mask = mask/255.
    
    cv2.imwrite(folder + "/" + "_.%05d.png" % (step), keep_mask * 255)
    cv2.imwrite(folder1 +  "/" + "_.%05d.png" % (step), rgb_image)
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
    client.simSetTraceLine([1, 0, 0, 1],   0.01)
    
    client.armDisarm(True)
    client.takeoffAsync().join()
    # client.moveToPositionAsync(int(0), int(130), int(-100), int(10), 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join() #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
   
    #client.moveToPositionAsync(int(0), int(120), int(-50), int(10), 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join() #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
    # client.hoverAsync().join()
    # client.rotateToYawAsync(int(90),5,1).join()
     
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

        # client.moveToPositionAsync(int(20), int(70), int(-10), int(1), 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join() #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()


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
    target = env + '_' + hard
       
    try:
    #     print("Recording Start")
    #     client.startRecording()
        client.simSetTraceLine([0, 1, 0, 1],   150)
        plan(pre_img_src)   



    # except KeyboardInterrupt:
    #     pass
    finally:
        try:
            client.simSetTraceLine([1, 0, 0, 0.01],   0.01)
        except:
            pass
        length = 0
        for i in range(len(poses) -1):
            length += np.linalg.norm(poses[i]- poses[i+1])
        data = [target, np.min(min_depths), length]
        with open("./logs/data.txt", 'a') as f:
            print("data = ", data)
            f.writelines([json.dumps(data)])
    #     print("Recording end")
    #     client.stopRecording()
    #     print("Recording end")
    #     exit(0)
    
def is_obstacle_there(mask):
    print(mask.sum())
    return mask.sum() > 75000
    


   
def plan(pre_img_src):
   
    global client, radial_flow, rtvs, env, z, hard, min_depths, poses
    poses = []
    min_depths = []
    target = env + '_' + hard
    folder = "./logs/" + target 
    
    if not os.path.exists(folder+'/' + target):
        os.makedirs(folder+'/' + target)
    
    f =  open(f'./logs/{target}/trajectory_balancing.txt', 'a')
    f.truncate(0)
    model = Basnet(weights_path='/scratch/shankara/Basenet_retrained/saved_models/basnet_r18/249basnet_bsi_resnet18.pth')
    model.load_model()
  
    
    if env == "square":
        if hard == "True":
            goal = [-100, -125, 40, 1] 
            count_max = 4
        else:
            #easy
            goal = [-135, -65, 40, 1]
            count_max = 4
    elif env == "shen":
        if hard == "True":
            goal = [130, 150, -10, 1] 
            count_max = 7
        else:
            #easy
            goal = [50, 120, 0, 1]
            count_max = 3
    elif env == "sciart":
        if hard == "True":
            goal = [-70, 90, -20, 1] 
            count_max = 1
        else:
            #easy
            goal = [-70, 30, -20, 1] 
            count_max = 4
    elif env == "ying":
        if hard == "True":
            goal = [0, 110, -20, 1] 
            count_max = 3
        else:
            #easy
            goal = [20, 70, -10, 1]
            count_max = 4
    
       
    pos_x = goal[0]
    pos_y = goal[1]
    pos_z = goal[2]
    vel_goal = goal[3]
   
    final_position = np.array([pos_x, pos_y, pos_z])
   
    step = 1
    count = 0
    while True:
        
        
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        
        # reshape array to 4 channel image array H X W X 4
        img_src = img1d.reshape(response.height, response.width, 3)
        

        
        mask = model.infer(img_src, show=False)
                
        grayImage = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
     
        mask = save_mask(grayImage, step, img_src, target )
        
        response, = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False),])
        depth_img_in_meters = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
        depth_img_in_meters = depth_img_in_meters.reshape(response.height, response.width, 1)
        depth_8bit_lerped = np.interp(depth_img_in_meters, (0, 100), (0, 1))
        
        # depth_8bit_lerped = depth_8bit_lerped.astype('uint8')
        ## COPY
        depth_img = cv2.resize(depth_8bit_lerped, (512, 384))
        min_depths.append(np.min(depth_img))
        ##




        ct=1
        f12 = (radial_flow*mask)[::ct, ::ct]
        vel = rtvs.get_vel(img_src, pre_img_src, f12, env, step, mask)
        # f12_copy = f12
        # print("flow shape", f12_copy.shape)
        # f12_copy[:, :, 0] = 0
        # flow = fz.convert_from_flow(f12_copy)
        # cv2.imwrite('/ssd_scratch/shankara/IROS22_Collision_VS/RTVS/Main_Pipeline/save_rgb/flow_r_'+str(step)+'.png', flow)
        
        folder = f"./logs/{target}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(f'./logs/{target}/min_depths.json', 'w') as json_file:
            json.dump(min_depths, json_file)

        obthere = is_obstacle_there(mask)
             
              
        real_pos = client.simGetGroundTruthKinematics().position #x_val
        real_position = np.array([real_pos.x_val, real_pos.y_val, real_pos.z_val])
        
        distance = np.linalg.norm(real_position - final_position)
        print("distance ", distance)
        if distance < 2:
            break
        rtvs.vs_lstm = Model().to(device="cuda:0")
        t1 = time.time()
        
        if count < count_max:
            obthere = True
        if count == count_max:
            count = 0



        print(count, "Count!")
        if obthere and (distance > 10):
            count = count + 1
            print("OBSTACLE!!!!!!!")
            temp_var = True
            if z == '1':
                # client.moveByVelocityBodyFrameAsync(vel[0], -vel[2], vel[1] , 5)
                client.moveByVelocityBodyFrameAsync(vel[2], vel[1], vel[0] , 5, drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(True, 0))
                
            elif z == '0':
               
                # print("VEL HERE ", vel)
                real_orient = client.simGetGroundTruthKinematics().orientation
                orient_world = quaternion_to_euler_angle_vectorized1(real_orient.w_val, real_orient.x_val,real_orient.y_val,real_orient.z_val)
                # print("orient world, yaw",  orient_world[2], vel[3])
                
                # if vel[3] > 60:
                #     vel[3] = 60
                # elif vel[3] < -60:
                #     vel[3] = -60
                
                yaw = orient_world[2] +  vel[3]
                print("YAW", yaw)
                print("QORLD + VEL[3] ",  orient_world[2] ,  vel[3])
                
                if yaw > 0 and yaw  < 360:
                    yaw1 = yaw
                elif yaw > 360:
                    yaw1=  yaw - 360
                else:
                    yaw1 =  yaw + 360
                print("Current, Rotated", orient_world[2], yaw1)
            
                yaw1 = np.random.randint(200,270)
                print("YAAAAAAAAA", yaw1)
                # client.moveByRollPitchYawZAsync(0, 0, 235,  real_pos.z_val, 5).join()
                client.moveByVelocityBodyFrameAsync(vel[2], vel[1], 0 , 3, drivetrain =   airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, vel[3])) #.join()
                

                # client.moveByVelocityBodyFrameAsync(-vel[2], vel[0], 0 , 5,drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, 0))

        else :
            real_pos = client.simGetGroundTruthKinematics().position #x_val
            client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), vel_goal, 3e+38 ,  drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)) #yaw_mode=airsim.YawMode(True,0.1) ) #airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(False, 0)) #, 3e+38, drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0)).join()
            
            real_orient = client.simGetGroundTruthKinematics().orientation
            orient_world = quaternion_to_euler_angle_vectorized1(real_orient.w_val, real_orient.x_val,real_orient.y_val,real_orient.z_val)
            print("ORIENTATION", orient_world[2])
              
        pre_img_src = img_src# previous time step image
        
        real_pos = client.simGetGroundTruthKinematics().position #x_val
        real_position = np.array([real_pos.x_val, real_pos.y_val, real_pos.z_val])
        poses.append(real_position)
        
        folder = f"./logs/{target}"
                
        # with open(f'./logs/{target}/trajectory.json', 'a') as json_file:
            # json.dump(real_position, json_file)
        np.savetxt(f, real_position,  newline=" ")
        f.write("\n")

        step = step + 1
    
if __name__ == '__main__':
    main()
