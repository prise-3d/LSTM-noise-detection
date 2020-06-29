# main imports
import numpy as np
import pandas as pd
import sys, os, argparse
import random


# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg


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
    # if scene == 'Cuisine01':
    #     while 12 in zones[0:n_zones]:
    #         random.shuffle(zones)
    
    return zones[0:n_zones]

def main():

    parser = argparse.ArgumentParser(description="Read and compute entropy data file (using diff)")

    parser.add_argument('--dataset', type=str, help='dataset scene folder', required=True)
    parser.add_argument('--n_zones', type=int, help='number of zones used in train', default=10)
    parser.add_argument('--output', type=str, help='file with specific training zone', required=True)
    parser.add_argument('--thresholds', type=str, help='file with specific thresholds (using only scene from this file', default='')

    args = parser.parse_args()

    p_folder       = args.dataset
    p_n_zones      = args.n_zones
    p_output       = args.output
    p_thresholds   = args.thresholds

    # extract scenes to use if specified
    available_scenes = None

    if len(p_thresholds) > 0:
        
        available_scenes = []

        with open(p_thresholds) as f:
            thresholds_line = f.readlines()

            for line in thresholds_line:
                data = line.split(';')
                del data[-1] # remove unused last element `\n`
                current_scene = data[0]

                # need to rename `current_name` because we only used part6
                # scene_split = current_scene.split('_')
                # del scene_split[-1]
                # scene_name = '_'.join(scene_split)

                available_scenes.append(current_scene)


    # specific number of zones (zones indices)
    zones = np.arange(16)

    # get all scene names
    scenes = os.listdir(p_folder)

    # create output thresholds directory if necessary
    folder, _ = os.path.split(p_output)

    if len(folder) > 0:
        os.makedirs(folder)

    # for each scene we generate random zones choice
    for folder_scene in scenes:

        if available_scenes is not None:

            if folder_scene in available_scenes:
                selected_zones = get_random_zones(folder_scene, zones, p_n_zones)
                save_learned_zones(p_output, folder_scene, selected_zones)
        else:
            selected_zones = get_random_zones(folder_scene, zones, p_n_zones)
            save_learned_zones(p_output, folder_scene, selected_zones)
            

if __name__== "__main__":
    main()
