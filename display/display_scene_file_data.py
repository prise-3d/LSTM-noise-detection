# main imports
import numpy as np
import pandas as pd
import sys, os, argparse

# image processing
from PIL import Image
from ipfml import utils
from ipfml.processing import transform, segmentation

import matplotlib.pyplot as plt

# model imports
import joblib

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt

from processing.features_extractions import extract_data

dataset_folder = cfg.dataset_path
scenes_list    = cfg.scenes_names
zones_indices  = cfg.zones_indices

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

def display_simulation_thresholds(scene, zones_data, humans, image_indices, zones_learned=None):
    
    # get reference image
    #images = sorted([ img for img in os.listdir(os.path.join(dataset_folder, scene)) if cfg.scene_image_extension in img ])

    #reference_img_path = os.path.join(dataset_folder, scene, images[-1])
    #blocks = segmentation.divide_in_blocks(Image.open(reference_img_path), (200, 200), pil=False)

    fig=plt.figure(figsize=(35, 22))
    fig.suptitle("Displayed computed data for " + scene + " scene", fontsize=20)

    label_freq = 10

    # dataset information
    start_index = int(image_indices[1]) - int(image_indices[0])
    step_value = int(image_indices[1]) - int(image_indices[0])

    y_min_lim = sys.float_info.max
    y_max_lim = 0
    
    # get min max from data
    for data in zones_data:

        data = utils.normalize_arr_with_range(data)

        if min(data) < y_min_lim:
            y_min_lim = min(data)

        if max(data) > y_max_lim:
            y_max_lim = max(data)

    # display data
    for index, data in enumerate(zones_data):

        data = utils.normalize_arr_with_range(data)

        # get index of current value
        counter_index = 0
        current_value = start_index

        while(current_value < humans[index]):
            counter_index += 1
            current_value += step_value

        fig.add_subplot(4, 4, (index + 1))
        plt.plot(data)
        #plt.imshow(blocks[index], extent=[0, len(data), y_min_lim, y_max_lim])

        # draw vertical line from (70,100) to (70, 250)
        plt.plot([counter_index, counter_index], [y_min_lim, y_max_lim], 'k-', lw=2, color='red')

        if index % 4 == 0:
            plt.ylabel('Data values', fontsize=24)

        if index >= 12:
            plt.xlabel('Samples per pixel', fontsize=24)

        x_labels = [id * step_value + start_index for id, val in enumerate(data) if id % label_freq == 0]

        x = [v for v in np.arange(0, len(data)) if v % label_freq == 0]

        plt.xticks(x, x_labels, rotation=45)
        plt.ylim(y_min_lim, y_max_lim)

    plt.savefig(scene + '.png')
    #plt.show()

def main():

    parser = argparse.ArgumentParser(description="Read and process data file and display data for scene by zones")

    parser.add_argument('--file', type=str, help='data file path')
    parser.add_argument('--scene', type=str, help='Scene index to use', choices=cfg.scenes_indices)

    args = parser.parse_args()

    p_file     = args.file
    p_scene    = args.scene

    # 1. get scene name
    scenes_list = cfg.scenes_names
    scenes_indices = cfg.scenes_indices

    scene_index = scenes_indices.index(p_scene.strip())
    scene = scenes_list[scene_index]

    scene_path = os.path.join(cfg.dataset_path, scene)

    # 2. retrieve human_thresholds
    # construct zones folder
    human_thresholds = []
    zones_list = []

    for index in zones_indices:

        index_str = str(index)

        while len(index_str) < 2:
            index_str = "0" + index_str
        
        zones_list.append(cfg.zone_folder + index_str)

    for zone in zones_list:
            zone_path = os.path.join(scene_path, zone)

            with open(os.path.join(zone_path, cfg.seuil_expe_filename), 'r') as f:
                human_thresholds.append(int(f.readline()))


    # 3. load data by zone to display
    zone_data = []
    image_indices = None

    with open(p_file, 'r') as f:

        for line in f.readlines():

            data = line.split(';')

            scene_name = data[0]

            if scene == scene_name:
                scene_data = [ int(v) for v in data[5].split(',') ]

                zone_data.append(scene_data)

                if image_indices is None:
                    image_indices = data[4].split(',')

    # X. display results
    display_simulation_thresholds(scene, zone_data, human_thresholds, image_indices)

if __name__== "__main__":
    main()
