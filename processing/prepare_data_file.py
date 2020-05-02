# main imports
import numpy as np
import pandas as pd
import sys, os, argparse

# image processing
from PIL import Image
from ipfml import utils
from ipfml.processing import transform, segmentation, compression

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt

from features_extractions import extract_data

dataset_folder = cfg.dataset_path
zones_indices  = cfg.zones_indices
block_size     = (200, 200)


'''
Display progress information as progress bar
'''
def write_progress(progress):
    barWidth = 180

    output_str = "["
    pos = barWidth * progress
    for i in range(barWidth):
        if i < pos:
           output_str = output_str + "="
        elif i == pos:
           output_str = output_str + ">"
        else:
            output_str = output_str + " "

    output_str = output_str + "] " + str(int(progress * 100.0)) + " %\r"
    print(output_str)
    sys.stdout.write("\033[F")


def main():

    parser = argparse.ArgumentParser(description="Extract data from image dataset")

    parser.add_argument('--thresholds', type=str, help='file which cantains all thresholds')
    parser.add_argument('--output', type=str, help='save computed data for each zone of each scene into file')
    parser.add_argument('--method', type=str, help='method name to used', choices=cfg.features_choices_labels, default=cfg.features_choices_labels[0])
    parser.add_argument('--params', type=str, help='param of the method used', default="")
    parser.add_argument('--imnorm', type=int, help="specify if image is normalized before computing something", default=0, choices=[0, 1])

    args = parser.parse_args()

    p_thresholds = args.thresholds
    p_output     = args.output
    p_method     = args.method
    p_params     = args.params
    p_imnorm     = args.imnorm

    p_output_path = os.path.join(cfg.output_data_generated, p_output)

    # create output path if not exists
    if not os.path.exists(cfg.output_data_generated):
        os.makedirs(os.path.join(cfg.output_data_generated))

    # extract all thresholds from threshold file
    thresholds = {}
    scenes_list = []
    zones_list = np.arange(16)

    with open(p_thresholds) as f:
        thresholds_line = f.readlines()

        for line in thresholds_line:
            data = line.split(';')
            del data[-1] # remove unused last element `\n`
            scene = data[0]
            thresholds_scene = data[1:]

            scenes_list.append(scene)
            thresholds[scene] = thresholds_scene

    images_path = {}
    number_of_images = 0

    # create dictionnary of threshold and get all images path
    for scene in scenes_list:

        scene_path = os.path.join(dataset_folder, scene)

        images_path[scene] = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])
        number_of_images = number_of_images + len(images_path[scene])


    with open(p_output_path, 'w') as f:
        print("Erase", p_output_path, "previous file if exists")

    image_counter = 0
    # compute entropy for each zones of each scene images
    for scene in scenes_list:

        image_indices = [ dt.get_scene_image_quality(img_path) for img_path in images_path[scene] ]

        blocks_entropy = []

        # append empty list
        for zone in zones_list:
            blocks_entropy.append([])

        for img_path in images_path[scene]:

            blocks = segmentation.divide_in_blocks(Image.open(img_path), block_size)

            for index, block in enumerate(blocks):
                
                # normalize if necessary
                if p_imnorm:
                    block = np.array(block) / 255.

                blocks_entropy[index].append(extract_data(block, p_method, p_params))

            # write progress bar
            write_progress((image_counter + 1) / number_of_images)
            
            image_counter = image_counter + 1
        
        # write data into files
        with open(p_output_path, 'a') as f:
            for index, zone in enumerate(zones_list):
                f.write(scene + ';')
                f.write(str(index) + ';')
                f.write(zone + ';')

                f.write(str(thresholds[scene][index]) + ';')

                for index_img, img_quality in enumerate(image_indices):
                    f.write(str(img_quality))

                    if index_img + 1 < len(image_indices):
                        f.write(',')

                f.write(';')

                for index_b, values in enumerate(blocks_entropy[index]):
                    
                    # check if single values or multiple
                    if type(values) is list or (np.ndarray and not np.float64):
                        
                        for index_v, v in enumerate(values):
                            f.write(str(v))

                            if index_v + 1 < len(values):
                                f.write(' ')
                    else:
                        f.write(str(values))


                    if index_b + 1 < len(blocks_entropy[index]):
                        f.write(',')
                
                f.write(';\n')

    f.close()

if __name__== "__main__":
    main()
