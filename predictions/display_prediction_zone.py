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


def specific_display_label(label, chunk_size=3):
    label = label[::-1] # reverse label
    labels = [ label[i:i+chunk_size] for i in range(0, len(label), chunk_size) ]
    return ' '.join(labels)[::-1]

def display_simulation_thresholds(scene, zones_predictions, humans, image_indices, output, n_predictions, nsamples, p_zone, every=None, zones_learned=None):
    
    # get reference image
    fig=plt.figure(figsize=(35, 22))
    # fig.suptitle("Detection simulation for " + scene + " scene", fontsize=20)

    # dataset information
    start_index = int(image_indices[1]) - int(image_indices[0])
    step_value = int(image_indices[1]) - int(image_indices[0])

    label_freq = (nsamples / step_value / 100 * 3) - 1

    if every is not None:
        step_value = every * step_value
    
    if every == 1:
        label_freq = 2 * label_freq

    y_min_lim, y_max_lim = (-0.2, 1.2)

    prediction_found = []

    predictions_data = zones_predictions[p_zone]

    predictions = []
    predictions_label = []

    threshold_model = None
    predictions_counter = 0

    for index_v, v in enumerate(predictions_data):
        v = float(v)
        predictions.append(v)
        predictions_label.append([0 if v < 0.5 else 1])

        if threshold_model is None:
            if v < 0.5:
                predictions_counter += 1

                if predictions_counter >= n_predictions:
                    threshold_model = index_v
                    prediction_found.append(index_v * step_value + start_index)
            else:
                predictions_counter = 0

    
    # get index of current value
    counter_index = 0
    current_value = start_index

    while(current_value < humans[p_zone]):
        counter_index += 1
        current_value += step_value

    # fig.add_subplot(4, 4, (index + 1))
    plt.plot(predictions, lw=6)
    plt.plot(predictions_label, linestyle='--', color='slategray', lw=4)
    #plt.imshow(blocks[index], extent=[0, len(predictions), y_min_lim, y_max_lim])

    # if zones_learned is not None:
    #     if p_zone in zones_learned:
    #         ax = plt.gca()
            # ax.set_facecolor((0.9, 0.95, 0.95))

    if threshold_model is None:
        threshold_model = len(predictions_data) - 1
        prediction_found.append(len(predictions_data) * step_value)

    # draw vertical line from (70,100) to (70, 250)
    plt.plot([counter_index, counter_index], [-2, 2], 'k-', lw=10, color='red')
    plt.plot([threshold_model, threshold_model], [-2, 2], 'k-', lw=8, color='blue')

#        if index % 4 == 0:
    plt.ylabel('Bruité / Non bruité', fontsize=70)

#        if index >= 12:
    plt.xlabel('Échantillons par pixel', fontsize=70)

    x_labels = [id * step_value + start_index for id, val in enumerate(predictions) if id % label_freq == 0]  + [nsamples]
    #x_labels = [id * step_value + start_index for id, val in enumerate(predictions) if id % label_freq == 0]

    x = [v for v in np.arange(0, len(predictions)) if v % label_freq == 0] + [int(nsamples / (20 * every))]
    y = np.arange(-1, 2, 10)

    plt.xticks(x, [ specific_display_label(str(x)) for x in x_labels ], rotation=45, fontsize=56)
    
    ax = plt.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(56)

    ax.tick_params(width=5, length=15, size=15)
    # ax[1].tick_params(width=3, length=6)

    plt.ylim(y_min_lim, y_max_lim)

    fig.tight_layout()
    plt.savefig(output + '.pdf', dpi=100)
    #plt.show()()


def main():

    parser = argparse.ArgumentParser(description="Read and process prediction file")

    parser.add_argument('--predictions', type=str, help='prediction file of scene')
    parser.add_argument('--npredictions', type=int, help='expected number of predictions before gettint threshold')
    parser.add_argument('--sequence', type=int, help='sequence length expected')
    parser.add_argument('--thresholds', type=str, help='file which cantains all thresholds')
    parser.add_argument('--learned_zones', type=str, help="Filename which specifies if zones are learned or not and which zones", default="")
    parser.add_argument('--every', type=int, help="every images only", default=1)
    parser.add_argument('--scene', type=str, help='Scene folder to use')
    parser.add_argument('--nsamples', type=int, help='expected max number of samples')
    parser.add_argument('--zone', type=int, help='expected zone number')
    parser.add_argument('--output', type=str, help='output image file name')

    args = parser.parse_args()

    p_prediction = args.predictions
    p_n_prediction = args.npredictions
    p_sequence = args.sequence
    p_thresholds = args.thresholds
    p_zones    = args.learned_zones
    p_every    = args.every
    p_scene    = args.scene
    p_output   = args.output
    p_zone     = args.zone
    p_nsamples = args.nsamples

    # 1. get scene name
    scene_path = p_scene

    if scene_path[-1] == '/':
        scene_path = scene_path[:-1]

    _, scene_name = os.path.split(scene_path)


    human_thresholds = []

    # 3. retrieve human_thresholds
    # construct zones folder
    with open(p_thresholds) as f:
        thresholds_line = f.readlines()

        for line in thresholds_line:
            data = line.split(';')
            del data[-1] # remove unused last element `\n`
            current_scene = data[0]
            thresholds_scene = data[1:]

            if scene_name == current_scene:
                human_thresholds = [ int(threshold) for threshold in  thresholds_scene ]

    if len(human_thresholds) == 0:
        print('Cannot manage this scene, no thresholds available')
        return
    
    print(human_thresholds)
    
    images_path = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])
    number_of_images = len(images_path)
    image_indices = [ dt.get_scene_image_quality(img_path) for img_path in images_path ]

    zones_predictions = []
    # Get predictions from model (let by default to 1 the first sequence values)
    with open(p_prediction, 'r') as f:
        lines = f.readlines()

        for line in lines:
            data = line.split(';')

            if data[0] == scene_name:

                predictions = []
                
                for _ in range(p_sequence - 1):
                    predictions.append(1)

                for v in data[2:-1]:
                    predictions.append(v)

                zones_predictions.append(predictions)

    # 5. check if learned zones
    zones_learned = None

    if len(p_zones) > 0:
        with open(p_zones, 'r') as f:
            lines = f.readlines()

            for line in lines:
                data = line.split(';')

                if data[0] == scene_name:
                    zones_selected = data[1:]
                    del zones_selected[-1]
                    zones_learned = [ int(zone) for zone in zones_selected ]

    # 6. display results
    display_simulation_thresholds(scene_name, zones_predictions, human_thresholds, image_indices, p_output, p_n_prediction, p_nsamples, p_zone, p_every, zones_learned)

if __name__== "__main__":
    main()
