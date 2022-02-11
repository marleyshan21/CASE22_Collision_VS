import airsim

import numpy as np
import os
import cv2

# connect to the AirSim simulator
client = airsim.MultirotorClient(ip = "10.2.36.227")
# client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)



airsim.wait_key('Press any key to takeoff')
print("Taking off...")
client.armDisarm(True)
client.takeoffAsync().join()
real_pos = client.simGetGroundTruthKinematics().position
print(real_pos)

i = 10

while True:

    real_pos = client.simGetGroundTruthKinematics().position
    print(real_pos)

    pos_x = input("x: ")
    pos_y = input("y: ")
    pos_z = input("z: ")
    vel = input("vel: ")
    # angle_z = input("yaw in deg: ")

    
    # client.rotateByYawRateAsync(90, 1).join()
    client.moveToPositionAsync(int(pos_x), int(pos_y), int(pos_z), int(vel)).join()
    client.hoverAsync().join()
    # client.rotateToYawAsync(int(angle_z),5,1).join()

    rot = input('yaw: ')
    client.rotateToYawAsync(int(rot),5,1).join()



    
    img = input('Press y to click image')
    if(img == 'y'):
        # responses = client.simGetImages([airsim.ImageRequest("high_res", airsim.ImageType.Scene, False, False)])
        responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis),
                airsim.ImageRequest("0", airsim.ImageType.Scene)])  
        print('Retrieved images: %d' % len(responses))
        i=i+1
        tmp_dir = os.path.join("/home/rrc/vs_airsim/vs/dfvs", "images_airsim")
        print ("Saving images to %s" % tmp_dir)
        try:
            os.makedirs(tmp_dir)
        except OSError:
            if not os.path.isdir(tmp_dir):
                raise

        
        for idx, response in enumerate(responses):

            filename = os.path.join(tmp_dir, str(i) + "_" + str(idx))

            if response.pixels_as_float:
                # print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
                airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
                # print("this")
            elif response.compress: #png format
                # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
            else: #uncompressed array
                # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                print(response.height, response.width)
                img_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)  # reshape array to 4 channel image array H X W X 3
                cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png # write to png
                
            # time.sleep(2) 
    
    q = input("Press q to exit ")
    if(q == 'q'):
        break
    

client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)