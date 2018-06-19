# TODO: из кучи маленьких картинок, надо сделать одну большую.
# TODO: сначала надо нарезать ОДНУ картинку на РОВНЫЕ части и пронумеровать их по порядку
# TODO: верхний левый угол -- это №_1, правый нижний -- это №_n
# TODO: затем надо запустить предикт по всем этим картинкам, получить результат

import os
import cv2
import numpy as np

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
        os.remove(path + file)
    return 0


def delete_res_img(path):
    k = number_of_splices(path) - 1
    files = os.listdir(path)
    for file in files:
        if file[:1] != str(k):
            print(file[:1], str(k))
            os.remove(path + file)


def number_of_splices(path_crop):
    crop_arr = []
    files = os.listdir(path_crop)
    n = len(files)
    print(n, files[n-1])
    k = int(files[n-1][:1]) + 1
    print(k)
    files.sort()
    print(files)
    return k


def rename_final_image(path):
    files = os.listdir(path)
    for filename in files:
        print('new name: ', filename[-10:])
        os.rename(os.getcwd() + '/data/img/' + filename, os.getcwd() + '/data/img/' + filename[-10:])
