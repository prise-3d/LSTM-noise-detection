# main imports
import numpy as np
import pandas as pd
import sys, os, argparse
import random
import math

# image processing
from PIL import Image
from ipfml import utils
from ipfml.processing import transform, segmentation

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg

dataset_folder = cfg.dataset_path
scenes_list    = cfg.scenes_names
zones_indices  = cfg.zones_indices


def save_learned_zones(output_name, scene, zones):

    with open(output_name, 'a') as f:
        f.write(scene + ';')

        for zone in zones:
            f.write(str(zone) + ';')

        f.write('\n')


def get_random_zones(scene, zones, percent):

    random.shuffle(zones)

    n_zones = int(math.ceil(len(zones) * percent))
    
    return zones[0:n_zones]

def main():

    parser = argparse.ArgumentParser(description="Read and compute entropy data file for each cluster data file")

    parser.add_argument('--folder', type=str, help='folder with cluster file')
    parser.add_argument('--output', type=str, help='output folder data')
    parser.add_argument('--sequence', type=int, help='sequence length expected')
    parser.add_argument('--percent', type=float, help='percent of zones to select per scene for each cluster data file', default=0.75)

    args = parser.parse_args()

    p_folder       = args.folder
    p_output       = args.output
    p_sequence     = args.sequence
    p_percent      = args.percent

    # take care of well sorted folders (cluster identifier depends on it)
    cluster_data = sorted(os.listdir(p_folder))
    print(cluster_data)

    learned_zones_output = os.path.join(cfg.output_zones_learned, p_output)

    if not os.path.exists(learned_zones_output):
        os.makedirs(learned_zones_output)

    for index_file, c_data in enumerate(cluster_data):

        data_file = os.path.join(p_folder, c_data)

        # create output path if not exists
        p_output_path = os.path.join(cfg.output_datasets, p_output)
        p_output_path_train = os.path.join(p_output_path, 'cluster_data_{}.train'.format(index_file))
        p_output_path_test = os.path.join(p_output_path, 'cluster_data_{}.test'.format(index_file))

        if not os.path.exists(p_output_path):
            os.makedirs(p_output_path)

        # read line by line file to estimate threshold entropy stopping criteria
        f_train = open(p_output_path_train, 'w')
        f_test = open(p_output_path_test, 'w')

        available_zones = {}

        with open(data_file, 'r') as f:

            lines = f.readlines()

            for line in lines:

                data = line.split(';')

                # only if scene is used for training part
                scene_name = data[0]
                zones_index = int(data[1])

                if scene_name not in available_zones:
                    available_zones[scene_name] = []
                if zones_index not in available_zones[scene_name]:
                    available_zones[scene_name].append(zones_index)

        # depending of available zones, define the zones to take in training set based on percent
        with open(data_file, 'r') as f:

            lines = f.readlines()

            new_scene = False
            current_scene = None
            selected_zones = None

            for line in lines:

                data = line.split(';')

                # only if scene is used for training part
                scene_name = data[0]
                zones_index = int(data[1])
                threshold = int(data[3])
                image_indices = data[4].split(',')
                values_list = data[5].split(',')

                sequence_data = []
                # one element is removed using this function (first element of list for computing first difference)
                # TODO : remove previous and add new

                if current_scene == None:
                    new_scene == True
                    current_scene = scene_name

                    zones = available_zones[scene_name]
                    selected_zones = get_random_zones(scene_name, zones, p_percent)
                    #print(selected_zones, 'are from', zones)

                    save_learned_zones(os.path.join(learned_zones_output, 'cluster_data_{}.csv'.format(index_file)), scene_name, selected_zones)

                if scene_name != current_scene:
                    new_scene = True
                    
                    zones = available_zones[scene_name]

                    # check if use of selected zones
                    selected_zones = get_random_zones(scene_name, zones, p_percent)
                    #print(selected_zones, 'are from', zones)

                    save_learned_zones(os.path.join(learned_zones_output, 'cluster_data_{}.csv'.format(index_file)), scene_name, selected_zones)
                else:
                    new_scene = False

                current_scene = scene_name

                for i, index in enumerate(image_indices):
                    
                    values = values_list[i].split(' ')

                    # append new sequence
                    sequence_data.append(values)

                    if i + 1 >= p_sequence:

                        label = int(threshold > int(index))

                        line = str(label) + ';'

                        for index_v, values in enumerate(sequence_data):

                            for index_x, x in enumerate(values):
                                line += str(x)

                                if index_x + 1 < len(values):
                                    line += ' '

                            if index_v + 1 < len(sequence_data):
                                line += ';'

                        line += '\n'

                        if zones_index in selected_zones:
                            f_train.write(line)
                        else:
                            f_test.write(line)

                        # del previous element
                        del sequence_data[0]

        f_test.close()
        f_train.close()    

if __name__== "__main__":
    main()
