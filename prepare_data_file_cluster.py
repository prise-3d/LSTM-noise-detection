# main imports
import numpy as np
import pandas as pd
import sys, os, argparse
import joblib

# image processing
from PIL import Image
from ipfml import utils
from ipfml.processing import transform, segmentation, compression

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt

from processing.features_extractions import extract_data
from complexity.run.estimators import estimate, estimators_list

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

    parser.add_argument('--dataset', type=str, help='folder dataset with all scenes', required=True)
    parser.add_argument('--cluster', type=str, help='clustering model to use', required=True)
    parser.add_argument('--nclusters', type=int, help='number of clusters', required=True)
    parser.add_argument('--estimators', type=str, help='list of estimators', default='l_mean,l_variance')
    parser.add_argument('--thresholds', type=str, help='file which contains all thresholds', required=True)
    parser.add_argument('--method', type=str, help='method name to used', choices=cfg.features_choices_labels, default=cfg.features_choices_labels[0])
    parser.add_argument('--params', type=str, help='param of the method used', default="", required=True)
    parser.add_argument('--imnorm', type=int, help="specify if image is normalized before computing something", default=0, choices=[0, 1])
    parser.add_argument('--output', type=str, help='output folder name with all clusters files', required=True)

    args = parser.parse_args()

    p_folder     = args.dataset
    p_thresholds = args.thresholds
    p_cluster    = args.cluster
    p_nclusters  = args.nclusters
    p_estimators = [ i.strip() for i in args.estimators.split(',') ]
    p_output     = args.output
    p_method     = args.method
    p_params     = args.params
    p_imnorm     = args.imnorm

    # load cluster model
    cluster_model = joblib.load(p_cluster)

    # prepare output_file path
    p_output_path = os.path.join(cfg.output_data_generated, p_output)

    # create output path if not exists
    if not os.path.exists(p_output_path):
        os.makedirs(os.path.join(p_output_path))

    output_files_list = []
    for i in range(p_nclusters):
        outfile = os.path.join(p_output_path, 'cluster_data_{}.csv'.format(i))
        output_files_list.append(outfile)

        with open(outfile, 'w') as f:
            print('Creation of empty {0} data file'.format(outfile))

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

    # get all images path
    for scene in scenes_list:

        scene_path = os.path.join(p_folder, scene)

        images_path[scene] = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])
        number_of_images = number_of_images + len(images_path[scene])

    # construct here dictionnary of associated cluster for each block
    clusters_block = {}

    for scene in scenes_list:
        first_image = images_path[scene][0]

        blocks = segmentation.divide_in_blocks(Image.open(first_image), block_size)

        clusters_block[scene] = {}

        for id_b, block in enumerate(blocks):

            # extract data and write into file
            x = []
            
            for estimator in p_estimators:
                estimated = estimate(estimator, block)
                
                if not isinstance(estimated, np.float64):
                    for v in estimated:
                        x.append(v)
                else:
                    x.append(estimated)

            # call cluster model
            predicted_label = cluster_model.predict([x])[0]

            # add label for this specific zone
            clusters_block[scene][id_b] = predicted_label


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
        for index, zone in enumerate(zones_list):

            # get associated cluster for this zone
            cluster_label = clusters_block[scene][index]

            with open(output_files_list[cluster_label], 'a') as f:

                zone_str = "zone" + str(zone)

                if len(zone_str) < 2:
                    zone_str = '0' + zone_str

                f.write(scene + ';')
                f.write(str(index) + ';')
                f.write(zone_str + ';')

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

if __name__== "__main__":
    main()
