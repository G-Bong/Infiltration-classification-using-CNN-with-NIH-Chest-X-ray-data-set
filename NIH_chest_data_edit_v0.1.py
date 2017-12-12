import tensorflow as tf
import numpy as np
import random
import cv2
import os
import glob
import pandas as pd
import matplotlib.pylab as plt


#../NIH_Chest_Xray_CNN
Data_PATH = os.path.abspath(os.path.join('..', 'NIH_Chest_Xray_CNN', 'NIH_Chest_X-ray_Dataset'))
print(Data_PATH)
# ../NIH_Chest_Xray_CNN/sample/images/
SOURCE_IMAGES = os.path.join(Data_PATH, "sample", "images")
print(SOURCE_IMAGES)

# ../NIH_Chest_Xray_CNN/sample/images/*.png
images = glob.glob(os.path.join(SOURCE_IMAGES, '*.png'))
# print(type(images))


# Load labels
labels = pd.read_csv('../NIH_Chest_Xray_CNN/NIH_Chest_X-ray_Dataset/sample/sample_labels.csv')
# print(labels)
# print(type(labels), '\n')

#Check the first five images paths
# print(images[0:5], '\n')

#Show three random images
r = random.sample(images, 3)
# print(r, '\n')

# plt.figure(figsize=(16, 16))
# for i in range(3):
#     plt.subplot(1, 3, i+1)
#     plt.imshow(cv2.imread(r[i]))


# Example of bad x-ray and good reason to use data augmentation
# e = cv2.imread(os.path.join(SOURCE_IMAGES, '00030209_008.png'))
# plt.imshow(e)
#print(labels[labels["Image Index"] == '00030209_008.png'])

# Turn images into arrays and make a list of classes
def image_pre_processing():
    """
        Returns two arrays:
            x is an array of resized images
            y is an array of labels
    """
    disease = "Infiltration"

    x = [] # images as arrays
    y = [] # labels Infiltration or Not_infiltration
    WIDTH = 128
    HEIGHT = 128

    for img in images:
        base = os.path.basename(img)
        finding = labels["Finding Labels"][labels["Image Index"] == base].values[0]
        # print(finding)

        # Read and resize image
        full_size_image = cv2.imread(img)
        x.append(cv2.resize(full_size_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC))
        # print(x)
        # print(np.shape(x))


        #Labels
        if disease in finding:
            #finding = str(disease)
            finding = 1
            y.append(finding)
        else:
            #finding = "Not_" + str(disease)
            finding = 0
            y.append(finding)

    return x,y


x, y = image_pre_processing()

# Set it up as a dataframe if you like
df = pd.DataFrame()
df["labels"]=y
df["images"]=x

np.savez("x_images_arrays", x)
np.savez("y_infiltration_labels", y)