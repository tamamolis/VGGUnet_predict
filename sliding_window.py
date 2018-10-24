import cv2
import os
import numpy as np
from  predict import create_predict


h = 416
w = 608

DataPath = 'data/'

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


def rename_final_image(path):
    files = os.listdir(path)
    for filename in files:
        print('new name: ', filename[-10:])
        os.rename(os.getcwd() + '/data/img/' + filename, os.getcwd() + '/data/img/' + filename[-10:])


def ceil(a, b):
    if (b == 0):
        raise Exception("Division By Zero Error!!")  # throw an division by zero error
    if int(a / b) != a / b:
        return int(a / b) + 1
    return int(a / b)


def step_size(img):
    h_orig, w_orig, channels = img.shape
    x = ceil(h_orig, h)
    y = ceil(w_orig, w)

    step_x = ceil((h_orig - h * x), x)
    print(x, (h_orig - h * x) / x)
    step_y = ceil((w_orig - w * y), y)
    print(y, (w_orig - w * y) / y)

    print(w_orig, w * y, w)

    return abs(step_x), abs(step_y)


def sliding_window(path, path_save):
    files = os.listdir(path)
    step_x, step_y = 0, 0
    for file in files:
        print(file)
        img = cv2.imread(path + file)
        print(path + file)

        h_orig, w_orig, channels = img.shape

        print('ширина и высота: ', w_orig, h_orig)
        print(path)

        windowSize = [h, w]
        step_x, step_y = step_size(img)
        print(step_x, step_y)

        i = 0
        j = 0

        for y in range(0, h_orig, h + step_y):
            for x in range(0, w_orig, w + step_x):

                if (y + windowSize[0] <= h_orig):
                    if (x + windowSize[1] <= w_orig):
                        crop_img = img[y: y + windowSize[0], x: x + windowSize[1]]
                        l = len(str(file))
                        name = file[:l - 4]
                        cv2.imwrite(path_save + str(i) + '_' + str(j) + '.png', crop_img)
                        j += 1
            i += 1
            j = 0
    return step_x, step_y


def delete_img(path):
    files = os.listdir(path)
    for file in files:
        img = cv2.imread(path + file)
        if np.shape(img) != (h, w, 3):
            os.remove(path + file)
    return 0


def crop_small_image(img, path_save, step_x, step_y):
    windowSize = [h, w]
    crop_img = img[0: windowSize[0] - step_y, 0: windowSize[1] - step_x]
    cv2.imwrite(path_save, crop_img)
    return 0


def merge_im(path, path_save, step_x, step_y, axis, flag):
    files = os.listdir(path)
    if axis == 1:
        for i in range(len(files)):
            try:
                index = files[i][:1]
                index_next = files[i + 1][:1]

                if index == index_next:
                    # print(index, index_next)
                    im1 = cv2.imread(path + files[i])
                    im2 = cv2.imread(path + files[i + 1])

                    if flag:
                        crop_small_image(im1, path + files[i], step_x, step_y)
                        crop_small_image(im2, path + files[i + 1], step_x, step_y)

                    # print(files[i], files[i+1])
                    res_im = np.concatenate((im1, im2), axis=axis)
                    os.remove(path + files[i])
                    cv2.imwrite(path + files[i], res_im)
                    cv2.imwrite(path_save + files[i], res_im)
                    os.remove(path + files[i + 1])
            except:
                IndexError

    if axis == 0:
        for i in range(len(files)):
            # print(files[i])
            try:

                im1 = cv2.imread(path + files[i])
                im2 = cv2.imread(path + files[i + 1])
                res_im = np.concatenate((im1, im2), axis=axis)
                os.remove(path + files[i])
                cv2.imwrite(path + files[i], res_im)
                cv2.imwrite(path_save + files[i], res_im)
                os.remove(path + files[i + 1])
            except:
                IndexError


def number_of_splices(path_crop):
    files = os.listdir(path_crop)
    print(files)
    n = len(files)
    print(n, files[n - 1])
    k = int(files[n - 1][:1]) + 1
    print(k)
    files.sort()
    print(files)
    return k


if __name__ == '__main__':

    os.system("find /Users/kate/PycharmProjects/seasonReport -name '.DS_Store' -delete")
    path = DataPath + 'orig/'
    path_save = DataPath + 'crop/'

    step_x, step_y = sliding_window(path, path_save)

    weights_path = os.getcwd() + '/weights/VGGUnet.weights.best.hdf5'
    create_predict(path_save, path_save, w, h, weights_path)

    path = DataPath + 'crop/'
    path_save = DataPath + 'res/'

    merge_im(path, path_save, step_x, step_y, 1, True)
    k = number_of_splices(path_save)
    for i in range(k):
        merge_im(path, path_save, step_x, step_y, 1, False)

    print("теперь по вертикали!\n")

    path = DataPath + 'crop/'
    path_save = DataPath + 'img/'

    # merge_im(path, path_save, step_x, step_y, 0, False)
    files = os.listdir(path)

    while len(files) > 1:
        merge_im(path, path, step_x, step_y, 0, False)
        files = os.listdir(path)

    # delete_img(path_save)
