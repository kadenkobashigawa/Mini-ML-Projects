import os
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from torchvision import transforms
from skimage import color



class GCoder:


    def split_gradients(self, image, number_of_tiers = 7, contrast = 2.5, show_graph = False):

        '''take in a PIL image or JPG path, convert the image into a high 
        contrast achromatic image, then split tones into layers...'''
        
        #if image is a path, open image as a PIL image...
        if type(image) == str:
            image = Image.open(os.path.join(image))
        
        #expand the image to fit the arm coordinate system...
        self.cartesian_size = 500
        expand = transforms.Resize((self.cartesian_size, self.cartesian_size))
        image = expand(image)

        #image prep for high contrast achromatic realism...
        image = transforms.functional.adjust_contrast(image, contrast_factor = contrast)
        image = color.rgb2gray(np.array(image))

        #normalize the achromatic pixel values to a 0 - 255 scale...
        gradients = (image - image.min()) / (image.max() - image.min()) * 255

        #find how many intensities to include in a tier...
        self.number_of_tiers = number_of_tiers
        tier_width = 255 // number_of_tiers

        #create a stack of layers to organize pixels by tier...
        layers = [np.ones(image.shape[:2], dtype = np.uint8) * number_of_tiers for i in range(number_of_tiers + 1)]
        tiers_visual = np.zeros(image.shape[:2], dtype = np.uint8)

        #assign tiers to layers using pixel location...
        for i, pixel_value in enumerate(gradients.flatten()):
            tier = pixel_value // tier_width
            layers[int(tier)].flat[i] = tier
            tiers_visual.flat[i] = tier

        #graph the visual if wanted...
        if show_graph:
            plt.imshow(tiers_visual, cmap = 'gray')
            plt.title('Tiers Visualization')
            plt.colorbar(label = 'Tier Gradient')
            plt.show()

        #return a list of gradient values organized by brightness and location: [layers, height, width]...
        return layers
    
    
    def map_layers(self, layers, folder_path = '', density = 50, decay_factor = 0.0025, 
                   offset_min = 15, offset_max = 50, definition = 50):
        
        '''convert array - like image into g-code path...'''

        #create a list to hold sketch plan and gcode parameters...
        all_coordinates = []
        self.num_scribbles = []
        self.offset = []

        #define paths for every layer of the drawing...
        for intensity, layer in enumerate(layers[:-1]):

            #get height and width dimensions of the image...
            height, width = layer.shape

            #equations to determine offset and number of scribbles...
            num_scribbles = round(density * decay_factor ** (intensity / (len(layers) - 1)))
            offset = offset_min if intensity < (len(layers) / 5) else offset_max
            self.num_scribbles.append(num_scribbles)
            self.offset.append(offset)
            
            #for every column and every row, create randomized scribbles with densities predetermined by the layer...
            for y in range(0, height, self.cartesian_size // definition): 
                for x in range(0, width, self.cartesian_size // definition):

                    #plot scribbles...
                    if layer[y, x] != len(layers) - 1:
                        for _ in range(num_scribbles):
                            offset_x = random.randint(-offset, offset)
                            offset_y = random.randint(-offset, offset)
                            all_coordinates.append(((x + offset_x) - width - 200, height - (y + offset_y) + 100))

            #save plotted coordinates as jpgs if path is given...
            if folder_path:
                plt.figure(figsize = (5, 5))
                x_vals = [coor[0] for coor in all_coordinates]
                y_vals = [coor[1] for coor in all_coordinates]
                plt.plot(x_vals, y_vals, color = 'k', linewidth = 0.075)
                plt.savefig(f"{folder_path}/layer_{intensity + 1}.jpg", format = "jpg", dpi = 300, bbox_inches = 'tight')
                print(f'Layer {intensity + 1} | Scribble Density: {self.num_scribbles[intensity]}  Scribble Offset: {self.offset[intensity]}')

        #return sketch plan from separated gradients...
        print(f'\nSketches saved in "{folder_path}."')
        return all_coordinates


    def find_arm_angles(self, points, R_length = 117.5, H_length = 144.0):

        '''convert cartesian g-code path into a sequence of shoulder and elbow angles...'''
        
        #split h and k values for calculation...
        h, k = zip(*points)
        h = np.array(h)
        k = np.array(k)

        #define virtual radius and humerus length from actual lengths in mm...
        RtoH_ratio = R_length / H_length 
        H = 2 * self.cartesian_size / (1 + RtoH_ratio)
        R = self.cartesian_size * 2 - H

        #calculate angle between humerus and x axis...
        D = (h**2 + k**2) ** 0.5
        theta_s_rad = (np.arctan2(k, h) - np.arccos((D**2 + H**2 - R**2) / (2 * D * H)))
        theta_s = np.round(theta_s_rad * 180 / np.pi, 0)

        #get x and y components for H and R...
        Hy = H * np.sin(theta_s_rad)
        Hx = H * np.cos(theta_s_rad)
        Ry = k - Hy
        Rx = h - Hx

        #use humerus and radius vectors to get the elbow angle...
        theta_e_rad = np.arccos((Hx * Rx + Hy * Ry) / (H * R))
        theta_e = np.round(theta_e_rad * 180 / np.pi, 0)
        
        #return full gcode... 
        return zip(theta_s, theta_e)
    

    def tag(self):

        '''code to add to the end of the gcode file...'''
        
        #this way gcode can just be run as a python file...
        tag = """\



import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)
GPIO.setup(12, GPIO.OUT)
servo_s = GPIO.PWM(11, 50)
servo_e = GPIO.PWM(12, 50)
servo_s.start(0)
servo_e.start(0)
print('Beret: Setup Complete!')
time.sleep(1)

for coordinate in gcode:
    print(f's:{coordinate[0]} | e:{coordinate[1]}')
    servo_s.ChangeDutyCycle(coordinate[0] / 18 + 2)
    servo_e.ChangeDutyCycle(coordinate[1] / 18 + 2)
    time.sleep(0.025)

servo_s.ChangeDutyCycle(90 / 18 +2)
servo_e.ChangeDutyCycle(90 / 18 +2)
time.sleep(0.5)
servo_s.ChangeDutyCycle(0)
servo_e.ChangeDutyCycle(0)
servo_s.stop()
servo_e.stop()
GPIO.cleanup()
print('Beret: Drawing Complete!')"""



        #copy and paste...
        return tag