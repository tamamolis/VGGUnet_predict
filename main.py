import os
from merge_img import crop, concat


DataPath = os.getcwd() + '/data/'


if __name__ == '__main__':
    print(DataPath)
    if not os.path.exists(DataPath):
        os.makedirs(DataPath)

    os.system("find /Users/kate/PycharmProjects/make_data -name '.DS_Store' -delete")
    path_orig = DataPath + 'orig/'
    if not os.path.exists(path_orig):
        os.makedirs(path_orig)
    path_crop = DataPath + 'res/'
    if not os.path.exists(path_crop):
        os.makedirs(path_crop)
    path_save = DataPath + 'img/'
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    Flag = 0

    if Flag:
        crop(path_orig, path_crop)
    else:
        for i in range(4):
            concat(path_crop, path_save)