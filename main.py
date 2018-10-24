import os
from sliding_window import sliding_window, merge_im, number_of_splices, rename_final_image
from predict import create_predict

DataPath = os.getcwd() + '/data/'
n_classes = 7
w = 416
h = 608
weights_path = os.getcwd() + '/weights/weights.best.hdf5'
# weights_path = os.getcwd() + '/weights/VGGUnet.weights.best.hdf5'


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

    os.system("find /Users/kate/PycharmProjects/seasonReport -name '.DS_Store' -delete")

    path_orig = DataPath + 'orig/'
    path_crop = DataPath + 'crop/'

    step_x, step_y = sliding_window(path_orig, path_crop)
    create_predict(path_crop, path_crop, h, w, weights_path)
    os.system("find /Users/kate/PycharmProjects/seasonReport -name '.DS_Store' -delete")

    merge_im(path_crop, path_crop, step_x, step_y, 1, True)

    k = number_of_splices(path_crop)
    for i in range(k):
        merge_im(path_crop, path_crop, step_x, step_y, 1, False)

    print("теперь по вертикали!\n")
    files = os.listdir(path_crop)

    while len(files) > 1:
        merge_im(path_crop, path_crop, step_x, step_y, 0, False)
        files = os.listdir(path_crop)
        os.system("find /Users/kate/PycharmProjects/seasonReport -name '.DS_Store' -delete")

    # delete_img(path_crop)
    # rename_final_image(path_crop)