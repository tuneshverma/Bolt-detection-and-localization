import cv2
import numpy as np
import imutils
from skimage.util import random_noise
from matplotlib import pyplot
import os
from tqdm import tqdm


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def augmentation(folder_of_images, storing_images):
    for image in tqdm(os.listdir(folder_of_images)):
        raw_image = cv2.imread(folder_of_images + '/' + image)

        # for angle in (np.arange(0, 360, 180)):
        #     rotated = imutils.rotate_bound(raw_image, angle)
            # cv2.imshow("Rotated (Correct)", rotated)
            # cv2.waitKey(0)
        for kernal in range(1, 13, 4):
            blur = cv2.GaussianBlur(raw_image, (kernal, kernal), 0)

            for value in range(10, 90, 60):
                frame = increase_brightness(blur, value=value)

                list_for_amount = [0.0050, 0.0750, 0.1550]

                for amount in list_for_amount:
                    noise_img = random_noise(frame, mode='s&p', amount=amount)
                    pyplot.imsave(storing_images + '/' + image + str(kernal) + '_' + str(value) + '_' +
                                  str(amount).split('.')[1] + '.png', noise_img)
                # cv2.waitKey(0)

            # cv2.imshow('new_image' + str(kernal), blur)
            # cv2.waitKey(0)

# augmentation('without_bolt', 'augmented_without_bolt')


def sliding_window(image, saving_folder):

    image = cv2.imread(image)
    print(image.shape)
    resize_image = cv2.resize(image, (640, 640))
    # print(resize_image.shape)

    filter = 150
    s = 100
    result_image_num = int((resize_image.shape[0] - filter) / s + 1)

    for i in range(result_image_num):
        for j in range(result_image_num):
            cropped_image = resize_image[(s * j): (filter + (s * j)), (s * i): (filter + (s * i))]
            cv2.imwrite(saving_folder + str(i) + '_' + str(j) + '_' + str(s) + str(filter) + '.png', cropped_image)