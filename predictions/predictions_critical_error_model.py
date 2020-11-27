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

def display_simulation_thresholds(scene, zones_predictions, humans, image_indices, output, critical_error, every, zones_learned, margin, thresholds_indices):
    
    # get reference image
    fig=plt.figure(figsize=(35, 22))

    label_freq = 10

    # dataset information
    start_index = int(image_indices[1]) - int(image_indices[0])
    step_value = int(image_indices[1]) - int(image_indices[0])
    basic_step_value = int(image_indices[1]) - int(image_indices[0])

    if every is not None:
        step_value = every * step_value
    
    if every == 1:
        label_freq = 2 * label_freq

    arr = np.array(critical_error).flatten()
    percent = np.count_nonzero(arr == 1, axis=0) / len(arr)
    # fig.suptitle("Detection simulation for {0} scene (margin: {1}%, error: {2:.3f}%)".format(scene, margin*100, percent*100), fontsize=26)
            
    y_min_lim, y_max_lim = (-0.2, 1.2)

    for index, predictions_data in enumerate(zones_predictions):

        predictions = []
        predictions_label = []

        for v in predictions_data:
            v = float(v)
            predictions.append(v)
            predictions_label.append([0 if v < 0.5 else 1])

        # get index of current value
        counter_index = 0
        current_value = start_index

        while(current_value < humans[index]):
            counter_index += 1
            current_value += step_value

        min_counter_index = int((humans[index] - int(len(image_indices) * margin) * basic_step_value) / step_value)
        max_counter_index = int((humans[index] + int(len(image_indices) * margin) * basic_step_value) / step_value)
        # print(min_counter_index, max_counter_index)

        # if human thresholds is 10000, escape as all images is noisy
        if humans[index] == 10000:
            min_counter_index = counter_index
            max_counter_index = counter_index

        fig.add_subplot(4, 4, (index + 1))
        plt.plot(predictions, lw=2)
        plt.plot(predictions_label, linestyle='--', color='darkslategrey', lw=2, alpha=0.5)
        #plt.imshow(blocks[index], extent=[0, len(predictions), y_min_lim, y_max_lim])

        if zones_learned is not None:
            if index in zones_learned:
                ax = plt.gca()
                # ax.set_facecolor((0.9, 0.95, 0.95))
                ax.spines['left'].set_linewidth(2)
                ax.spines['top'].set_linewidth(2)
                ax.spines['right'].set_linewidth(2)
                ax.spines['bottom'].set_linewidth(2)


        # draw vertical line from (70,100) to (70, 250)
        plt.plot([counter_index, counter_index], [-2, 2], 'k-', lw=2, color='red')

        current_thresholds_index = int(thresholds_indices[index] / every)
        
        plt.plot([current_thresholds_index, current_thresholds_index], [-2, 2], 'k-', lw=4, color='dimgray')

        #plt.axhspan(i, i+.2, facecolor='0.2', alpha=0.5)
        # plt.axvspan(min_counter_index, max_counter_index, facecolor='r', alpha=0.4)

        #zone_percent = np.count_nonzero(np.array(critical_error[index]) == 1, axis=0) / len(critical_error[index])

        for i in range(len(critical_error[index])):
            if critical_error[index][i] == 1:
                # pass
                plt.axvspan(i, i + 1, facecolor='orange', alpha=0.2)
            else:
                plt.axvspan(i, i + 1, facecolor='g', alpha=0.2)
                # pass

        for i in range(len(critical_error[index])):

            current_threshold_index = thresholds_indices[index]

            if current_threshold_index < i:
                plt.axvspan(i, i + 1, facecolor='orange', alpha=0.2)
            else:
                plt.axvspan(i, i + 1, facecolor='g', alpha=0.2)

        # if index % 4 == 0:
        plt.ylabel('Not noisy / Noisy', fontsize=20)

        # if index >= 12:
        plt.xlabel('Samples per pixel\n', fontsize=20)

        x_labels = [id * step_value + start_index for id, val in enumerate(predictions) if id % label_freq == 0]  + [10000]
        #x_labels = [id * step_value + start_index for id, val in enumerate(predictions) if id % label_freq == 0]

        x = [v for v in np.arange(0, len(predictions)) if v % label_freq == 0] + [int(10000 / (20 * every))]
        y = np.arange(-1, 2, 10)

        plt.xticks(x, x_labels, rotation=45)
        #plt.yticks(y, y)
        plt.ylim(y_min_lim, y_max_lim)

    fig.tight_layout(h_pad=2)

    plt.savefig(output + '.png')
    #plt.show()()

def main():

    parser = argparse.ArgumentParser(description="Read and process prediction file")

    parser.add_argument('--data', type=str, help='data predictions')
    parser.add_argument('--dataset', type=str, help='scenes dataset folder')
    parser.add_argument('--nstop', type=int, help='number of expected correct stopping criteria', required=True)
    parser.add_argument('--sequence', type=int, help='sequence length expected')
    parser.add_argument('--thresholds', type=str, help='file which cantains all thresholds')
    parser.add_argument('--learned_zones', type=str, help="Filename which specifies if zones are learned or not and which zones", default="")
    parser.add_argument('--every', type=int, help="every images only", default=1)
    parser.add_argument('--margin', type=float, help="\"accepted\" margin percent error", default=0.05)
    parser.add_argument('--output', type=str, help='output image file name')

    args = parser.parse_args()

    p_data     = args.data
    p_dataset  = args.dataset
    p_nstop    = args.nstop
    p_sequence = args.sequence
    p_thresholds = args.thresholds
    p_zones    = args.learned_zones
    p_every    = args.every
    p_margin   = args.margin
    p_output   = args.output

    if not os.path.exists(p_output):
        os.makedirs(p_output)

    # 1. get all thresholds
    human_thresholds = {}

    # 2. retrieve human_thresholds
    # construct zones folder
    with open(p_thresholds) as f:
        thresholds_line = f.readlines()

        for line in thresholds_line:
            data = line.split(';')
            del data[-1] # remove unused last element `\n`
            current_scene = data[0]
            thresholds_scene = data[1:]

            human_thresholds[current_scene] = [ int(threshold) for threshold in  thresholds_scene ]

    # 3. check if learned zones
    zones_learned = {}

    if len(p_zones) > 0:
        with open(p_zones, 'r') as f:
            lines = f.readlines()

            for line in lines:
                data = line.split(';')

                zones_selected = data[1:]
                del zones_selected[-1]
                zones_learned[data[0]] = [ int(zone) for zone in zones_selected ]

    # 4. get all predictions
    predictions_files = sorted(os.listdir(p_data))

    scene_critical_predictions = {}
    global_percent = []

    for predictions in predictions_files:

        prediction_filepath = os.path.join(p_data, predictions)
        zones_predictions = []

        # csv file need to have scene name as prefix
        scene_name = predictions.replace('.csv', '')

        if scene_name in human_thresholds:

            # Get predictions from model (let by default to 1 the first sequence values)
            with open(prediction_filepath, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    data = line.split(';')

                    predictions = []
                    
                    for _ in range(p_sequence - 1):
                        predictions.append(1)

                    for v in data[2:-1]:
                        predictions.append(v)

                    zones_predictions.append(predictions)

            scene_path = os.path.join(p_dataset, scene_name)

            images_path = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])
            number_of_images = len(images_path)
            image_indices = [ dt.get_scene_image_quality(img_path) for img_path in images_path ]
            step_index = image_indices[1] - image_indices[0]

            # print('Scene used is', scene_name)
            # print('nimages =>', number_of_images)
            # print('zones_predictions =>', len(zones_predictions[0]))

            scene_critical_predictions[scene_name] = []

            counter_zones_predictions = []
            zones_stopping_indices = []

            for i in range(len(zones_predictions)):

                counter_zones_predictions.append(0)
                zones_stopping_indices.append(None)

                critical_predictions = []

                min_margin_threshold = human_thresholds[scene_name][i] - int(len(image_indices) * p_margin) * step_index
                max_margin_threshold = human_thresholds[scene_name][i] + int(len(image_indices) * p_margin) * step_index

                # if human thresholds is 10000, escape as all images is noisy
                if human_thresholds[scene_name][i] == 10000:
                    min_margin_threshold = 10000
                    max_margin_threshold = 10000

                # print(i, '=>', min_margin_threshold, human_thresholds[scene_name][i], max_margin_threshold)

                pred_index = 0
                
                for j in range(len(image_indices)):

                    if j % p_every == 0:
                        pred_label = 0 if float(zones_predictions[i][pred_index]) < 0.5 else 1

                        if pred_label == 0:

                            counter_zones_predictions[i] += 1

                            # keeping in memory the current `j` which is the index used for detected image
                            if zones_stopping_indices[i] is None and counter_zones_predictions[i] >= p_nstop:
                                zones_stopping_indices[i] = j
                        else:
                            counter_zones_predictions[i] = 0

                        pred_index += 1

                if zones_stopping_indices[i] is None:
                    zones_stopping_indices[i] = int(len(image_indices) - 1)
                
                pred_index = 0

                for j in range(len(image_indices)):

                    if j % p_every == 0:
                    
                        pred_label = 0 if float(zones_predictions[i][pred_index]) < 0.5 else 1
                        
                        critical_error = None

                        # TODO : revoir les conditions à ce niveau (erreur après seuil à ne plus afficher car arrêt)

                        if image_indices[j] <= human_thresholds[scene_name][i] and image_indices[j] <= image_indices[zones_stopping_indices[i]]:
                            critical_error = 0 # no error

                        if image_indices[j] >= human_thresholds[scene_name][i] and image_indices[j] >= image_indices[zones_stopping_indices[i]]:
                            critical_error = 0

                        # if image_indices[j] > min_margin_threshold and image_indices[j] < max_margin_threshold:
                        #     critical_error = 0
                        
                        if critical_error is None:
                            critical_error = 1

                        critical_predictions.append(critical_error)

                        pred_index += 1

                scene_critical_predictions[scene_name].append(critical_predictions)

            arr = np.array(scene_critical_predictions[scene_name]).flatten()
            percent = np.count_nonzero(arr == 1, axis=0) / len(arr)
            global_percent.append(percent)
            print('[margin: {0}] Critical percent error on {1} is {2}%'.format(p_margin, scene_name, percent))


            # 6. display results
            display_simulation_thresholds(scene_name, zones_predictions, human_thresholds[scene_name], image_indices, os.path.join(p_output, scene_name), scene_critical_predictions[scene_name], p_every, zones_learned[scene_name], p_margin, zones_stopping_indices)
    
    print('Global percent is {0}%'.format(sum(global_percent) / len(global_percent)))

if __name__== "__main__":
    main()
