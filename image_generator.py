from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image_utils
import matplotlib.pyplot as plt
import numpy as np
import settings
from tqdm import tqdm
import cv2
import os

stage1_images_directory = './images/stage_1/'
stage2_images_directory = './images/stage_2/'

stage1_raw_image_directory = './raw_data/stage_1/'
stage2_raw_image_directory = './raw_data/stage_2/'

datagen = ImageDataGenerator(
    rotation_range=40, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True, 
    fill_mode='nearest'
)
stages = [settings.STAGE1_CATEGORIES, settings.STAGE2_CATEGORIES]
image_directories = [stage1_images_directory, stage2_images_directory]
raw_image_directories = [stage1_raw_image_directory, stage2_raw_image_directory]

for i in range(len(stages)):
    for category in stages[i]:
        path = os.path.join(image_directories[i], category)
        path_out = os.path.join(raw_image_directories[i], category)
        for j, file in tqdm(enumerate(os.listdir(path))):
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (settings.picture_width, settings.picture_height))
            cv2.imwrite(os.path.join(path_out, f'{category}-{str(j)}.jpeg'), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_array = image_utils.img_to_array(img).reshape((1,) + img.shape)
            z = 0
            for batch in datagen.flow(img_array, save_prefix='text', save_format='jpeg'):
                new_image = cv2.cvtColor(batch[0], cv2.COLOR_BGR2RGB).astype(int)
                cv2.imwrite(os.path.join(path_out, f'{category}-{str(j)}-{str(z)}.jpeg'), new_image)
                z += 1
                if z > 4:
                    break