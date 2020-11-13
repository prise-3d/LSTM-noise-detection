# main imports
import numpy as np
import pandas as pd
import sys, os, argparse
import random

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

    if not os.path.exists(cfg.output_zones_learned):
        os.makedirs(cfg.output_zones_learned)

    with open(os.path.join(cfg.output_zones_learned, output_name), 'a') as f:
        f.write(scene + ';')

        for zone in zones:
            f.write(str(zone) + ';')

        f.write('\n')


def get_random_zones(scene, zones, n_zones):

    random.shuffle(zones)

    # specific case for 'Cuisine01' (zone 12 is also noisy even in reference image)
    if scene == 'Cuisine01':
        while 12 in zones[0:n_zones]:
            random.shuffle(zones)
    
    return zones[0:n_zones]

def main():

    parser = argparse.ArgumentParser(description="Read and compute entropy data file (using diff)")

    parser.add_argument('--data', type=str, help='file data to read and compute')
    parser.add_argument('--output', type=str, help='output dataset prefix file used (saved into .train and .test extension)')
    parser.add_argument('--sequence', type=int, help='sequence length expected')
    parser.add_argument('--n_zones', type=int, help='number of zones used in train', default=0)
    parser.add_argument('--selected_zones', type=str, help='file with specific training zone')
    parser.add_argument('--zones_val', type=int, help='number of zones for separated validation file', default=3)
    parser.add_argument('--seq_norm', type=int, help='normalization sequence by features', choices=[0, 1])

    args = parser.parse_args()

    p_data         = args.data
    p_output       = args.output
    p_sequence     = args.sequence
    p_n_zones      = args.n_zones
    p_selected_zones = args.selected_zones
    p_zones_val    = args.zones_val
    p_seq_norm     = bool(args.seq_norm)

    learned_zones = None

    if p_selected_zones is not None:
        print('Use of specific learned zones')

        with open(p_selected_zones, 'r') as f:
            lines = f.readlines()

            learned_zones = {}

            for line in lines:
                data = line.split(';')
                del data[-1]
                print(data)
                learned_zones[data[0]] = [ int(d) for d in data[1:] ]
    else:
        print("Number of zones used in train:", p_n_zones)

    # create output path if not exists
    p_output_path = os.path.join(cfg.output_datasets, p_output)
    p_output_path_train = os.path.join(p_output_path, p_output + '.train')
    p_output_path_val = os.path.join(p_output_path, p_output + '.val')
    p_output_path_test = os.path.join(p_output_path, p_output + '.test')

    if not os.path.exists(p_output_path):
        os.makedirs(p_output_path)

    # read line by line file to estimate threshold entropy stopping criteria
    f_train = open(p_output_path_train, 'w')
    f_val = open(p_output_path_val, 'w')
    f_test = open(p_output_path_test, 'w')

    zones = np.arange(16)

    with open(p_data, 'r') as f:
        lines = f.readlines()

        new_scene = False
        current_scene = None
        selected_zones = None
        val_zones = None

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

                # check if use of selected zones
                if learned_zones:
                    selected_zones = learned_zones[scene_name]
                else:
                    selected_zones = get_random_zones(scene_name, zones, p_n_zones)
                
                val_zones = random.sample(list(selected_zones), k=p_zones_val)
                save_learned_zones(p_output, scene_name, selected_zones)

            if scene_name != current_scene:
                new_scene = True
                random.shuffle(zones)
                
                # check if use of selected zones
                if learned_zones:
                    selected_zones = learned_zones[scene_name]
                else:
                    selected_zones = get_random_zones(scene_name, zones, p_n_zones)

                val_zones = random.sample(list(selected_zones), k=p_zones_val)
                save_learned_zones(p_output, scene_name, selected_zones)
            else:
                new_scene = False

            current_scene = scene_name

            for i, index in enumerate(image_indices):
                
                values = values_list[i].split(' ')

                # append new sequence
                sequence_data.append(values)

                if i + 1 >= p_sequence:

                    label = int(threshold > int(index))

                    line = scene_name + ';'
                    line += str(zones_index) + ';'
                    line += str(index) + ';'
                    line += str(label) + ';'

                    current_sequence_data = sequence_data.copy()

                    # do seq normalisation here if necessary
                    if p_seq_norm:
                        
                        data = np.array(current_sequence_data, 'float32')
                        _, f = data.shape
                        for i in range(f):
                            #final_arr[index][]
                            data[:, i] = utils.normalize_arr_with_range(data[:, i])

                        current_sequence_data = data.tolist()
                    
                    for index_v, values in enumerate(current_sequence_data):

                        for index_x, x in enumerate(values):
                            line += str(x)

                            if index_x + 1 < len(values):
                                line += ' '

                        if index_v + 1 < len(current_sequence_data):
                            line += ';'

                    line += '\n'

                    if zones_index in val_zones:
                        f_val.write(line)
                    elif zones_index in selected_zones:
                        f_train.write(line)
                    else:
                        f_test.write(line)

                    # del previous element
                    del sequence_data[0]

    f_test.close()
    f_val.close()
    f_train.close()    
    


if __name__== "__main__":
    main()
