import os
from sliding_window import sliding_window, merge_im, number_of_splices
from predict import create_predict
from CRF import crf, crop_orig, stretch
from skimage.io import imread


DataPath = os.getcwd() + '/data/'
n_classes = 5
w = 416
h = 608
# weights_path = os.getcwd() + '/weights/VGGUnet.7.weights.best.hdf5'
weights_path = os.getcwd() + '/weights/VGGUnet.5.weights.best.hdf5'


if __name__ == '__main__':

    os.system("find /Users/kate/PycharmProjects/VGGUnet-predict -name '.DS_Store' -delete")
    print(DataPath)
    if not os.path.exists(DataPath):
        os.makedirs(DataPath)

    os.system("find /Users/kate/PycharmProjects/VGGUnet-predict -name '.DS_Store' -delete")
    path_orig = DataPath + 'orig/'
    if not os.path.exists(path_orig):
        os.makedirs(path_orig)
    path_crop = DataPath + 'crop/'
    if not os.path.exists(path_crop):
        os.makedirs(path_crop)

    os.system("find /Users/kate/PycharmProjects/VGGUnet-predict -name '.DS_Store' -delete")

    path_orig = DataPath + 'orig/'
    path_crop = DataPath + 'crop/'

    step_x, step_y = sliding_window(path_orig, path_crop)

    print("step_x, step_y: ", step_x, step_y)
    create_predict(path_crop, path_crop, h, w, weights_path, n_classes)
    os.system("find /Users/kate/PycharmProjects/VGGUnet-predict -name '.DS_Store' -delete")

    merge_im(path_crop, path_crop, step_x, step_y, 1, True)

    k = number_of_splices(path_crop)
    for i in range(k):
        merge_im(path_crop, path_crop, step_x, step_y, 1, False)

    print("теперь по вертикали!\n")
    files = os.listdir(path_crop)

    while len(files) > 1:
        merge_im(path_crop, path_crop, step_x, step_y, 0, False)
        files = os.listdir(path_crop)
        os.system("find /Users/kate/PycharmProjects/VGGUnet-predict -name '.DS_Store' -delete")

    os.system("find /Users/kate/PycharmProjects/VGGUnet-predict -name '.DS_Store' -delete")
    path = 'data/orig/'
    path_seg = 'data/crop/'
    stretch(path_seg, path_seg)
    crop_orig(path, path_seg)

    os.system("find /Users/kate/PycharmProjects/VGGUnet-predict -name '.DS_Store' -delete")

    orig = 'data/crop/crop.png'
    seg = 'data/crop/resize.png'

    image = imread(orig)
    seg_image = imread(seg)

    crf(image, seg_image, "data/crop/result_crf.jpg")