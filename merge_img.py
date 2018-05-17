# TODO: из кучи маленьких картинок, надо сделать одну большую.
# TODO: сначала надо нарезать ОДНУ картинку на РОВНЫЕ части и пронумеровать их по порядку
# TODO: верхний левый угол -- это №_1, правый нижний -- это №_n
# TODO: затем надо запустить предикт по всем этим картинкам, получить результат

import os
import cv2
import numpy as np
from keras.preprocessing.image import array_to_img

h = 416
w = 608
res_im = 0


def crop(path, path_save):

    files = os.listdir(path)
    for file in files:

        print(file)
        img = cv2.imread(path + file)
        print(path + file)

        h_im, w_im, channels = img.shape

        w_list = []
        h_list = []

        for i in np.arange(0, w_im, w):
            w_list.append(i)

        for j in np.arange(0, h_im, h):
            h_list.append(j)

        print(h_list, w_list)
        print(path)

        for i in range(len(w_list)):
            for j in range(len(h_list)):
                try:
                    crop_img = img[h_list[i]: h_list[i+1], w_list[j]: w_list[j+1]]
                    cv2.imwrite(path_save + str(i) + '_' + str(j) + '.png', crop_img)
                except:
                    IndexError


def concat(path, path_save):
    files = os.listdir(path)
    for i in range(len(files)):
        try:
            index = files[i][:1]
            index_next = files[i+1][:1]

            if index == index_next:
                    im1 = cv2.imread(path + files[i])
                    im2 = cv2.imread(path + files[i + 1])
                    res_im = np.concatenate((im1, im2), axis=1)
                    os.remove(path + files[i])
                    cv2.imwrite(path + files[i], res_im)
                    os.remove(path + files[i+1])
        except:
            IndexError
    # delete_img(path)
    files = os.listdir(path)
    files_new = os.listdir(path_save)
    Flag = True
    print(files)

    for i in range(len(files)):
        try:
            if Flag:
                    im1 = cv2.imread(path + files[i])
                    im2 = cv2.imread(path + files[i + 1])
                    res_im = np.concatenate((im1, im2), axis=0)
                    cv2.imwrite(path_save + files[i], res_im)
                    cv2.imwrite(path_save + str(i) + 'result.png', res_im)
                    Flag = False
            else:
                    im1 = cv2.imread(path_save + files_new[i])
                    im2 = cv2.imread(path + files[i + 1])
                    res_im = np.concatenate((im1, im2), axis=0)

                    cv2.imwrite(path_save + files_new[i], res_im)
                    cv2.imwrite(path_save + str(i) + str(i) + 'two_result.png', res_im)
        except:
                IndexError


def delete_img(path):
    files = os.listdir(path)
    for file in files:
        img = cv2.imread(path + file)
        if np.shape(img) == (416, 608, 3):
            os.remove(path + file)
    return 0


def big_image_from_small_image(path_crop, path_save):
    crop_arr = []

    files = os.listdir(path_crop)
    files.sort()
    print(files)

    for file in files:
        crop_arr.append(cv2.imread(path_crop + file))

    arr = np.zeros((3, 4, 416, 608, 3))
    k = 0
    for i in range(3):
        for j in range(4):
            arr[i][j] = crop_arr[k]
            k += 1

    crop_arr = np.reshape(crop_arr, (608*3, 416*4, 3))
    arr = np.reshape(arr, (416*3, 608*4, 3))
    rgb = array_to_img(arr)
    rgb.save(path_save + 'res.png')
    # cv2.imwrite(path_save + 'res.png', crop_arr)


