from multiprocessing import Pool
from constants_proj.AI_proj_params import PreprocParams
import matplotlib.pyplot as plt
from config.MainConfig import get_preproc_config
from inout_common.io_common import create_folder
from os.path import join
import skimage.io as io
from io_project.read_utils import get_all_files
from labelme_ozutils import printLabelMeSummary, getLabelMeImgData, shape_to_maskLabelMe
import json
import numpy as np

import os

NUM_PROC = 1

def main():
    # ----------- Parallel -------
    p = Pool(NUM_PROC)
    # p.map(img_generation_all, range(NUM_PROC))
    # p.map(gen_imgs_fromjson, range(NUM_PROC))
    gen_imgs_fromjson()
    # x = np.zeros((10,10,3))
    # x[:,:,0] = 255
    # y = np.ones((10,10,3)) * 255
    # y[2:4, 2:4, :] = 0
    # alphas = np.ones((10,10)) * .5
    # plt.imshow(x, vmin=0, vmax=255)
    # plt.imshow(y, alpha=alphas, vmin=0, vmax=255)
    # plt.show()

def gen_imgs_fromjson(projc_id=1):
    config = get_preproc_config()
    input_folder = config[PreprocParams.input_folder_raw]
    output_folder = config[PreprocParams.imgs_output_folder]

    class_folders = os.listdir(input_folder)
    for c_class in class_folders:
        print(F"------------------- Class {c_class} ----------------")
        c_class_folder = join(input_folder, c_class)
        c_class_files = [join(c_class_folder, x) for x in os.listdir(c_class_folder)]
        output_class_folder = join(output_folder, c_class)
        create_folder(output_class_folder)
        for c_file in c_class_files:
            print(F"File {c_file}")
            with open(c_file) as f:
                imgObj = json.load(f)

            imgData = np.array(getLabelMeImgData(imgObj))
            printLabelMeSummary(imgObj)
            shapes = imgObj["shapes"]
            all_masks = np.empty(imgData.shape)
            for i, c_shape in enumerate(shapes):
                c_mask = shape_to_maskLabelMe(imgData.shape, c_shape['points'], c_shape['shape_type'])
                all_masks[c_mask] = [0, 255,0]

            c_path = join(output_class_folder, os.path.basename(c_file).replace("json", "png"))
            fig, ax = plt.subplots()
            ax.imshow(imgData)
            ax.imshow(all_masks, alpha=.3)
            plt.title(F"Shape {i} with label {c_shape['label']} and type: {c_shape['shape_type']}")
            plt.savefig(c_path)
            plt.show()

def img_generation_all(proc_id):
    """
    Makes images of the available data
    :param proc_id:
    :return:
    """
    config = get_preproc_config()
    # input_folder = config[PreprocParams.input_folder_raw]
    input_folder = "/data/BetyFishClassification/PREPROC"
    output_folder = config[PreprocParams.imgs_output_folder]

    all_files, all_paths, Ys = get_all_files(input_folder)
    ids_0 = all_paths[Ys == 0]
    ids_1 = all_paths[Ys == 1]
    ids_2 = all_paths[Ys == 2]
    ids_3 = all_paths[Ys == 3]
    plot_n = 80  # How many images to plot
    for i in range(plot_n):
        if i % NUM_PROC == proc_id:
            im_0 = io.imread(ids_0[i])
            im_1 = io.imread(ids_1[i])
            im_2 = io.imread(ids_2[i])
            im_3 = io.imread(ids_3[i])
            fig, axs = plt.subplots(2,2, figsize=(14,10))
            axs[0,0].imshow(im_0)
            axs[0,0].set_title("0")
            axs[0,1].imshow(im_1)
            axs[0,1].set_title("1")
            axs[1,0].imshow(im_2)
            axs[1,0].set_title("2")
            axs[1,1].imshow(im_3)
            axs[1,1].set_title("3")
            # plt.show()
            plt.savefig(join(output_folder, F"{i}.png"))
        plt.close()
    # all_classes = os.listdir(input_folder)
    # for c_class in all_classes:
    #     class_output_folder = join(output_folder, c_class)
    #     create_folder(class_output_folder)
    #     all_files_per_class = os.listdir(join(input_folder, c_class))
    #     for c_file_name in all_files_per_class:
    #         c_path = join(input_folder, c_class, c_file_name)
    #         im = io.imread(c_path)
    #         plt.imshow(im)
    #         plt.title(c_class)
    #         # plt.savefig(join(class_output_folder, c_file_name))
    #         plt.show()
    #         break

if __name__ == '__main__':
    main()

##

