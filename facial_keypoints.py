# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:15:03 2019

@author: Angelo Antonio Manzatto

The databased used for this project was provided by #Udacity# provided on the following link: 
https://github.com/udacity/P1_Facial_Keypoints

The base model used for this project was based on the following paper: 
https://arxiv.org/pdf/1710.00977.pdf

The two test videos for this project were use from the dataset stored  site here where I will put a reference for their work:

[1] M. Kim, S. Kumar, V. Pavlovic, and H. Rowley, “Face Tracking and Recognition with Visual Constraints in Real-World Videos,” 
in IEEE Conf. Computer Vision and Pattern Recognition, Anchorage, AK, 2008. 

web page: http://seqamlab.com/youtube-celebrities-face-tracking-and-recognition-dataset/

Project Objectives

Facial keypoints detections is an important problem in computer vision field used on the following tasks:
* Facial tracking
* Facial pose recognition 
* Facial filters
* Emotion recognition

"""

##################################################################################
# Libraries
##################################################################################  

import os
import copy
import csv
import random
import numpy as np

import matplotlib.pyplot as plt

import cv2

from keras.models import Sequential

from keras.layers import Dropout, Dense, Flatten, Reshape, Lambda
from keras.layers import Conv2D,  MaxPooling2D 

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint, CSVLogger

from moviepy.editor import VideoFileClip

############################################################################################
# Files and folders and default parameters
############################################################################################

data_folder = "data"
training_folder = os.path.join(data_folder,"training" )
test_folder = os.path.join(data_folder,"test" )

training_keypoints_file = os.path.join(data_folder,'training_frames_keypoints.csv')
test_keypoints_file = os.path.join(data_folder,'test_frames_keypoints.csv')

############################################################################################
# Load data
############################################################################################
def load_data(keypoints_file, image_folder):

    data = []
    with open(keypoints_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            # Skip header
            if line_count > 0:
                
                # Get image name
                image_path = os.path.join(image_folder,row[0])
                
                # Get a list of x,y coordinates
                key_pts = row[1:]
                
                # Calculane total number of coordinates. This is supposed to not vary for each sample
                n_points = int(len(key_pts) / 2)
                
                # Convert list of coordinates into a matrix of size (n_coords, 2)
                key_pts = np.asarray(key_pts).reshape((n_points,2)).astype('float')
                
                data.append((image_path, key_pts))
                
            line_count +=1
            
    return data

# Load and split data into train / test / validation sets 
train_test_data = load_data(training_keypoints_file,training_folder)
valid_data = load_data(test_keypoints_file,test_folder)
train_data, test_data = train_test_split(train_test_data, test_size=0.20, random_state=42)

print('Size of train data: {0}'.format(len(train_data)))
print('Size of test data: {0}'.format(len(test_data)))
print('Size of valid data: {0}'.format(len(valid_data)))

############################################################################################
# Show data
############################################################################################

# Display a few of the images from the dataset
n_samples = 5
    
for i in range(n_samples):
    
    # define the size of images
    fig = plt.figure(figsize=(18,12))
    
    # randomly select a sample
    idx = np.random.randint(0, len(train_data))
    sample = train_data[idx]
    
    ax = plt.subplot(1, n_samples, i + 1)
    ax.set_title('Sample #{}'.format(i))
                 
    image_path, key_pts = sample
    
    image = plt.imread(image_path)
    
    plt.imshow(image)
    
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')
    
############################################################################################
# Data Augmentation 
############################################################################################
    
#############################
# Resize Image
#############################
class Resize(object):
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, key_pts = None):
        
        # Get image shape
        h, w = image.shape[:2]
        
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
            
        new_h, new_w = int(new_h), int(new_w)

        image = cv2.resize(image, (new_w, new_h))

        if key_pts is not None:
            
            new_key_pts = copy.deepcopy(key_pts)

            for k_p in new_key_pts:
                k_p[0] = k_p[0] * new_w / w
                k_p[1] = k_p[1] * new_h / h
                
            return image, new_key_pts
        
        return image, key_pts
    
#############################
# Translate Image
#############################    
class RandomTranslation(object):

    def __init__(self, ratio = 0.4, background_color = (0,0,0,255) , prob=0.5):
        
        self.background_color  = background_color
        self.ratio = ratio
        self.prob  = prob
        
    def __call__(self, image, key_pts):
                
        if random.uniform(0, 1) <= self.prob:
            
            img_h, img_w, img_c = image.shape
            
            x = int(np.random.uniform(-self.ratio,self.ratio) * img_w)
            y = int(np.random.uniform(-self.ratio,self.ratio) * img_h)

            M = np.float32([[1, 0, x],
                            [0, 1, y]])
                
            image_translated = cv2.warpAffine(image,M,(img_w,img_h), borderValue=self.background_color)
            
            new_key_pts = np.zeros_like(key_pts)
                
            new_key_pts[:,0] = key_pts[:,0] + x
            new_key_pts[:,1] = key_pts[:,1] + y

            return image_translated.astype("uint8") , new_key_pts
           
        return image, key_pts
    
#############################
# Scale Image
#############################  
class RandomScale(object):

    def __init__(self, lower = 0.4,upper = 1.4, background_color = (0,0,0,255) , prob=0.5):
        
        self.background_color = background_color
        self.lower = lower
        self.upper = upper
        self.prob  = prob  
        
    def __call__(self, image, key_pts):
                
        if random.uniform(0, 1) <= self.prob:
            
            img_h, img_w, img_c = image.shape
            
             # Create canvas with random ration between lower and upper
            ratio = random.uniform(self.lower,self.upper)
            
            scale_x = ratio
            scale_y = ratio
            
            # Scale the image
            scaled_image = cv2.resize(image.astype('float32'),(0,0),fx=scale_x,fy=scale_y)
            
            top = 0
            left = 0
            
            if ratio < 1:
                    
                background = np.zeros((img_h, img_w, img_c), dtype = np.uint8)
                
                background[:,:,:] = self.background_color 
            
                y_lim = int(min(scale_x,1)*img_h)
                x_lim = int(min(scale_y,1)*img_w)
                
                top  = (img_h - y_lim) // 2
                left = (img_w - x_lim) // 2

                background[top:y_lim+top,left:x_lim+left,:] = scaled_image[:y_lim,:x_lim,:]
                
                scaled_image = background
                    
            else:
                
                top  = (scaled_image.shape[0] -  img_h) // 2
                left = (scaled_image.shape[1] -  img_w) // 2
                
                scaled_image = scaled_image[top:img_h+top,left:img_w+left,:]
                
            # Correct key pts coordinates
            new_key_pts = np.zeros_like(key_pts)
            
            if ratio > 1:
                new_key_pts[:,0] = key_pts[:,0] * scale_x - left
                new_key_pts[:,1] = key_pts[:,1] * scale_y - top
            else:
                new_key_pts[:,0] = key_pts[:,0] * scale_x + left
                new_key_pts[:,1] = key_pts[:,1] * scale_y + top 
                
            return scaled_image.astype("uint8"), new_key_pts

        return image, key_pts
    
#############################
# Flip image
#############################    
class RandomFlip(object):
 
    def __init__(self, prob=0.5):

        self.prob = prob
        
    def __call__(self, image, key_pts= None):
        
        if random.uniform(0, 1) <= self.prob:
        
            # Get image shape
            h, w = image.shape[:2]
            
            # Flip image
            image = image[:, ::-1]

            if key_pts is not None:
                
                new_key_pts = copy.deepcopy(key_pts)
                
                new_key_pts[:,0] = w - key_pts[:,0]
                
                # We have to do a mapping between each point of left face to the right face
                # Brute force
                new_key_pts_flipped = np.zeros_like(new_key_pts)
                
                new_key_pts_flipped[0] = new_key_pts[16]
                new_key_pts_flipped[1] = new_key_pts[15]
                new_key_pts_flipped[2] = new_key_pts[14]
                new_key_pts_flipped[3] = new_key_pts[13]
                new_key_pts_flipped[4] = new_key_pts[12]
                new_key_pts_flipped[5] = new_key_pts[11]
                new_key_pts_flipped[6] = new_key_pts[10]
                new_key_pts_flipped[7] = new_key_pts[9]
                new_key_pts_flipped[8] = new_key_pts[8]
                new_key_pts_flipped[9] = new_key_pts[7]
                new_key_pts_flipped[10] = new_key_pts[6]
                new_key_pts_flipped[11] = new_key_pts[5]
                new_key_pts_flipped[12] = new_key_pts[4]
                new_key_pts_flipped[13] = new_key_pts[3]
                new_key_pts_flipped[14] = new_key_pts[2]
                new_key_pts_flipped[15] = new_key_pts[1]
                new_key_pts_flipped[16] = new_key_pts[0]
                new_key_pts_flipped[17] = new_key_pts[26]
                new_key_pts_flipped[18] = new_key_pts[25]
                new_key_pts_flipped[19] = new_key_pts[24]
                new_key_pts_flipped[20] = new_key_pts[23]
                new_key_pts_flipped[21] = new_key_pts[22]
                new_key_pts_flipped[22] = new_key_pts[21]
                new_key_pts_flipped[23] = new_key_pts[20]
                new_key_pts_flipped[24] = new_key_pts[19]
                new_key_pts_flipped[25] = new_key_pts[18]
                new_key_pts_flipped[26] = new_key_pts[17]
                new_key_pts_flipped[27] = new_key_pts[27]
                new_key_pts_flipped[28] = new_key_pts[28]
                new_key_pts_flipped[29] = new_key_pts[29]
                new_key_pts_flipped[30] = new_key_pts[30]
                new_key_pts_flipped[31] = new_key_pts[35]
                new_key_pts_flipped[32] = new_key_pts[34]
                new_key_pts_flipped[33] = new_key_pts[33]
                new_key_pts_flipped[34] = new_key_pts[32]
                new_key_pts_flipped[35] = new_key_pts[31]
                new_key_pts_flipped[36] = new_key_pts[45]
                new_key_pts_flipped[37] = new_key_pts[44]
                new_key_pts_flipped[38] = new_key_pts[43]
                new_key_pts_flipped[39] = new_key_pts[42]
                new_key_pts_flipped[40] = new_key_pts[47]
                new_key_pts_flipped[41] = new_key_pts[46]
                new_key_pts_flipped[42] = new_key_pts[39]
                new_key_pts_flipped[43] = new_key_pts[38]
                new_key_pts_flipped[44] = new_key_pts[37]
                new_key_pts_flipped[45] = new_key_pts[36]
                new_key_pts_flipped[46] = new_key_pts[41]
                new_key_pts_flipped[47] = new_key_pts[40]
                new_key_pts_flipped[48] = new_key_pts[54]
                new_key_pts_flipped[49] = new_key_pts[53]
                new_key_pts_flipped[50] = new_key_pts[52]
                new_key_pts_flipped[51] = new_key_pts[51]
                new_key_pts_flipped[52] = new_key_pts[50]
                new_key_pts_flipped[53] = new_key_pts[49]
                new_key_pts_flipped[54] = new_key_pts[48]
                new_key_pts_flipped[55] = new_key_pts[59]
                new_key_pts_flipped[56] = new_key_pts[58]
                new_key_pts_flipped[57] = new_key_pts[57]
                new_key_pts_flipped[58] = new_key_pts[56]
                new_key_pts_flipped[59] = new_key_pts[55]
                new_key_pts_flipped[60] = new_key_pts[64]
                new_key_pts_flipped[61] = new_key_pts[63]
                new_key_pts_flipped[62] = new_key_pts[62]
                new_key_pts_flipped[63] = new_key_pts[61]
                new_key_pts_flipped[64] = new_key_pts[60]
                new_key_pts_flipped[65] = new_key_pts[67]
                new_key_pts_flipped[66] = new_key_pts[66]
                new_key_pts_flipped[67] = new_key_pts[65]
 
                return image, new_key_pts_flipped
            
        return image, key_pts  
    
#############################
# Blur image
#############################    
class RandomBlur(object):
    def __init__(self, lower = 1, upper = 5, prob=0.5):

        self.lower = lower
        self.upper = upper
        self.prob = prob
        
    def __call__(self, image, key_pts = None):
        
        if random.uniform(0, 1) <= self.prob:
            
            amount = random.randint(self.lower, self.upper)
            
            if(amount % 2 == 0):
                amount += 1 if random.random() < 0.5 else -1
           
            blur_image = cv2.GaussianBlur(image,(amount,amount),cv2.BORDER_DEFAULT)
            
            return blur_image, key_pts
        
        return image, key_pts
    
#############################
# Convert image to gray
#############################    
class ToGray(object):

    def __call__(self, image, key_pts = None):
        
        #Convert image to gray scale
        gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        
        gray = np.expand_dims(gray, axis=-1)
        
        return gray, key_pts
    
############################################################################################
# Test Data Augmentation 
############################################################################################
def plot_transformation(transformation, n_samples = 3):

    for i in range(n_samples):
    
        # define the size of images
        f, (ax1, ax2) = plt.subplots(1, 2)
        f.set_figwidth(14)

        # randomly select a sample
        idx = np.random.randint(0, len(train_data))
        image_path, key_pts = train_data[idx]

        image = plt.imread(image_path)

        new_image, new_key_pts = transformation(image, key_pts)
        
        if(image.shape[-1] == 1):
            ax1.imshow(np.squeeze(image), cmap = 'gray')
        else:
            ax1.imshow(image)
        ax1.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')
        ax1.set_title('Original')

        if(new_image.shape[-1] == 1):
            ax2.imshow(np.squeeze(new_image), cmap = 'gray')
        else:
            ax2.imshow(new_image)
        ax2.scatter(new_key_pts[:, 0], new_key_pts[:, 1], s=20, marker='.', c='m')
        ax2.set_title(type(transformation).__name__)

        plt.show()
        
##########################
# Resize Test
##########################
resize = Resize((96, 96))
plot_transformation(resize)

##########################
# Translation Test
##########################
random_translation = RandomTranslation(ratio=0.2,background_color=(104, 117, 123, 255),prob=1.0)
plot_transformation(random_translation)

##########################
# Scale Test
##########################
random_scale = RandomScale(lower=0.6,upper=1.4,background_color=(104, 117, 123,255),prob=1.0)
plot_transformation(random_scale)

##########################
# Flip Test
##########################
random_flip = RandomFlip(prob=0.5)
plot_transformation(random_flip)

##########################
# Blur Test
##########################
random_blur = RandomBlur(prob=1.0)
plot_transformation(random_blur)

##########################
# Gray Test
##########################
to_gray = ToGray()
plot_transformation(to_gray)

############################################################################################
# Key Point operations
############################################################################################
def normalize_key_pts(key_pts, img_size):
    
    w, h = img_size
    
    half_w = w / 2
    half_h = h / 2
    
    new_key_pts = copy.deepcopy(key_pts)
    
    new_key_pts[:,0] = (new_key_pts[:,0] - half_w) / half_w
    new_key_pts[:,1] = (new_key_pts[:,1] - half_h) / half_h
    
    return new_key_pts
    
def denormalize_key_pts(key_pts, img_size):
    
    w, h = img_size
    
    half_w = w / 2
    half_h = h / 2
    
    new_key_pts = copy.deepcopy(key_pts)
    
    new_key_pts[:,0] = (new_key_pts[:,0] * half_w) + half_w
    new_key_pts[:,1] = (new_key_pts[:,1] * half_h) + half_h
    
    return new_key_pts

############################################################################################
# Create Dataset
############################################################################################
    
# Create X, y tuple from image_path, key_pts tuple
def createXy(data, transformations = None):
    
    image_path, key_pts = data
    
    image = plt.imread(image_path)
    
    # Apply transformations for the tuple (image, labels, boxes)
    if transformations:
        for t in transformations:
            image, key_pts = t(image,key_pts)
    
    key_pts_normalized = normalize_key_pts(key_pts, image.shape[:2])
            
    return image, key_pts_normalized

# Generator for using with model
def generator(data, transformations = None, batch_size = 4, shuffle_data= True):
    
    n_samples = len(data)
    
    # Loop forever for the generator
    while 1:
        
        if shuffle_data:
            data = shuffle(data)
        
        for offset in range(0, n_samples, batch_size): 
            
            batch_samples = data[offset:offset + batch_size]
            
            X = []
            y = []
            
            for sample_data in batch_samples:
                
                image, target = createXy(sample_data, transformations)

                X.append(image)
                y.append(target)
                
            X = np.asarray(X).astype('float32')
            y = np.asarray(y).astype('float32')
            
            yield (shuffle(X, y))
            
############################################################################################
# Model
############################################################################################

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 , input_shape=(96,96,1)))

################################
# Block 1
################################
model.add(Conv2D(32, kernel_size=(4,4), activation='elu', kernel_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

################################
# Block 2
################################
model.add(Conv2D(64, kernel_size=(3,3), activation='elu', kernel_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

################################
# Block 3
################################
model.add(Conv2D(128, kernel_size=(2,2), activation='elu', kernel_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

################################
# Block 4
################################
model.add(Conv2D(256, kernel_size=(1,1), activation='elu', kernel_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(1000, activation='elu', kernel_initializer='glorot_uniform'))
model.add(Dropout(0.5))

model.add(Dense(1000, activation='elu', kernel_initializer='glorot_uniform'))
model.add(Dropout(0.6))

model.add(Dense(136, kernel_initializer='glorot_uniform'))

model.add(Reshape((68,2)))

model.compile(optimizer='adam',loss='mse')

model.summary()

############################################################################################
# Training pipeline
############################################################################################

# Data augmentation 
train_transformations = [
        RandomBlur(prob=0.5),
        RandomTranslation(ratio=0.2,background_color=(104, 117, 123, 255),prob=0.5),
        RandomScale(lower=0.8,upper=1.2,background_color=(104, 117, 123,255),prob=0.5),
        RandomFlip(prob=0.5),
        Resize((96,96)),
        ToGray()
        ]

test_transformations = [
        RandomBlur(prob=0.5),
        RandomTranslation(ratio=0.2,background_color=(104, 117, 123, 255),prob=0.5),
        RandomScale(lower=0.6,upper=1.4,background_color=(104, 117, 123,255),prob=0.5),
        RandomFlip(prob=0.5),
        Resize((96,96)),
        ToGray()
        ]

valid_transformations = [
        Resize((96,96)),
        ToGray()
        ]

# Hyperparameters
epochs = 500
batch_size = 64
learning_rate = 0.001
weight_decay = 5e-4
momentum = .9

train_generator = generator(train_data, train_transformations, batch_size)
test_generator = generator(test_data, test_transformations, batch_size)

# callbacks
model_path = 'saved_models'

# File were the best model will be saved during checkpoint     
model_file = os.path.join(model_path,'facial_keypoints-{val_loss:.4f}.h5')

# Check point for saving the best model
check_pointer = ModelCheckpoint(model_file, monitor='val_loss', mode='min',verbose=1, save_best_only=True)

# Logger to store loss on a csv file
csv_logger = CSVLogger(filename='facial_keypoints.csv',separator=',', append=True)

history = model.fit_generator(train_generator,steps_per_epoch=int(len(train_data) / batch_size),
                              validation_data=test_generator,validation_steps=int(len(test_data) / batch_size),
                              epochs=epochs, verbose=1, callbacks=[check_pointer,csv_logger],workers=1)

############################################################################################
# Predict
############################################################################################

# If we want to test on a pre trained model use the following line
model.load_weights(os.path.join(model_path,'facial_keypoints-0.0082.h5'), by_name=False)

n_samples = 5

for i in range(n_samples):
    
    # define the size of images
    fig = plt.figure(figsize=(18,12))
    
    # randomly select a sample
    idx = np.random.randint(0, len(valid_data))
    sample = valid_data[idx]
    
    ax = plt.subplot(1, n_samples, i + 1)
    ax.set_title('Prediction #{}'.format(i))
                 
    image_path, key_pts = sample
    
    image = plt.imread(image_path)
    
    h, w = image.shape[:2]

    image_copy = np.copy(image)
    key_pts_copy = np.copy(key_pts)

    for t in valid_transformations:
        image_copy, key_pts_copy = t(image_copy, key_pts_copy)

    key_pts_pred = model.predict(image_copy[np.newaxis,...]) 
    
    key_pts_pred = denormalize_key_pts(np.squeeze(key_pts_pred),image_copy.shape[:2])

    plt.imshow(np.squeeze(image_copy), cmap = 'gray')
    plt.scatter(key_pts_copy[:, 0], key_pts_copy[:, 1], s=20, marker='.', c='m')
    plt.scatter(key_pts_pred[:, 0], key_pts_pred[:, 1], s=20, marker='x', c='r')
    
############################################################################################
# Video Testing
############################################################################################    

video_folder = 'videos'

# Class for detecting key points on a video
class KeyPtsDetector():
    
    def __init__(self,  xy_origin = (0,0), crop_size=(96,96), patch_size = (96,96)):
    
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.xy_origin = xy_origin
       
    def process_frame(self,image):
    
        x, y = self.xy_origin
        
        # Crop video from starting point
        cropped_image = image[y:y+self.crop_size[1] ,x:x+self.crop_size[0]]
        
        # Resize to fit the model input
        scaled_image = cv2.resize(cropped_image,self.patch_size)
        
        #Convert image to gray scale
        gray = cv2.cvtColor(scaled_image,cv2.COLOR_RGB2GRAY)
        
        gray = np.expand_dims(gray, axis=-1)
            
        # Predict value using model
        key_pts = model.predict(gray[np.newaxis,...])
        
        # Denormilize from [-1,1] to the normal size again
        key_pts = denormalize_key_pts(np.squeeze(key_pts),scaled_image.shape[:2])
        
        # Draw Key Points
        for key_pt in key_pts:
            
            cv2.circle(scaled_image,(key_pt[0],key_pt[1]),1,(0,255,0),1)
         
        return scaled_image
    

video_path_1 = os.path.join(video_folder,'0553_01_006_donald_trump.avi')

video = VideoFileClip(video_path_1)
video.reader.close()

kpt_detector = KeyPtsDetector()
kpt_detector.xy_origin = (40,0)
kpt_detector.crop_size = (200,200)
kpt_detector.patch_size = (96,96)

output_clip = video.fl_image(kpt_detector.process_frame) #NOTE: this function expects color images!!

# Save processed video clip
output_clip.write_videofile(os.path.join(video_folder,'video_test_1.mp4'),audio=False)

##########################

video_path_2 = os.path.join(video_folder,'0492_03_009_bill_gates.avi')

video = VideoFileClip(video_path_2)
video.reader.close()

kpt_detector = KeyPtsDetector()
kpt_detector.xy_origin = (120,0)
kpt_detector.crop_size = (200,200)
kpt_detector.patch_size = (96,96)

output_clip = video.fl_image(kpt_detector.process_frame) #NOTE: this function expects color images!!

# Save processed video clip
output_clip.write_videofile(os.path.join(video_folder,'video_test_2.mp4'),audio=False)
