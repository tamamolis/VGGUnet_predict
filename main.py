import os
from merge_img import crop, concat, number_of_splices, delete_img, delete_res_img
from predict import create_predict


DataPath = os.getcwd() + '/data/'
n_classes = 7
input_width = 416
input_height = 608
weights_path = os.getcwd() + "/weights/weights.best.hdf5"

if __name__ == '__main__':
    print(DataPath)
    if not os.path.exists(DataPath):
        os.makedirs(DataPath)

    os.system("find /Users/kate/PycharmProjects/seasonReport -name '.DS_Store' -delete")
    path_orig = DataPath + 'orig/'
    if not os.path.exists(path_orig):
        os.makedirs(path_orig)
    path_crop = DataPath + 'crop/'
    if not os.path.exists(path_crop):
        os.makedirs(path_crop)
    path_save = DataPath + 'res/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    path_img = DataPath + 'img/'
    if not os.path.exists(path_save):
        os.makedirs(path_img)

    crop(path_orig, path_crop)
    create_predict(path_crop, path_save, input_height, input_width, n_classes, weights_path)
    k = number_of_splices(path_save)
    for i in range(k):
        concat(path_save, path_img)

    delete_img(path_crop)
    delete_img(path_save)
    delete_res_img(path_img)