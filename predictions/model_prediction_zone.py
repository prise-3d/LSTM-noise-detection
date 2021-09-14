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
from keras.models import load_model

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt

from processing.features_extractions import extract_data

dataset_folder = cfg.dataset_path
scenes_list    = cfg.scenes_names
zones_indices  = cfg.zones_indices

output_figures = cfg.output_figures

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

def display_simulation_thresholds(predictions_data, human_threshold, image_indices, output, nsamples, every=None):

    # get reference image
    fig =plt.figure(figsize=(35, 22))
    # fig.suptitle("Detection simulation for " + scene + " scene", fontsize=20)

    # dataset information
    start_index = int(image_indices[1]) - int(image_indices[0])
    step_value = int(image_indices[1]) - int(image_indices[0])

    label_freq = nsamples / step_value / 100 * 2

    if every is not None:
        step_value = every * step_value
    
    # if every >= 1:
    label_freq = 2 * label_freq

    y_min_lim, y_max_lim = (-0.2, 1.2)


    predictions = []
    predictions_label = []

    threshold_model = None

    for index_v, v in enumerate(predictions_data):
        v = float(v)
        predictions.append(v)
        predictions_label.append([0 if v < 0.5 else 1])

        if threshold_model is None:
            if v < 0.5:
                threshold_model = index_v


    # get index of current value
    counter_index = 0
    current_value = start_index

    while(current_value < human_threshold):
        counter_index += 1
        current_value += step_value

    plt.plot(predictions, lw=4)
    plt.plot(predictions_label, linestyle='--', color='slategray', lw=4)
    #plt.imshow(blocks[index], extent=[0, len(predictions), y_min_lim, y_max_lim])

    # if zones_learned is not None:
    #     if index in zones_learned:
    #         ax = plt.gca()
            # ax.set_facecolor((0.9, 0.95, 0.95))

    # draw vertical line from (70,100) to (70, 250)
    # plt.plot([counter_index, counter_index], [-2, 2], 'k-', lw=6, color='red')
    plt.plot([threshold_model, threshold_model], [-2, 2], 'k-', lw=5, color='blue')

#        if index % 4 == 0:
    plt.ylabel('Not noisy / Noisy', fontsize=30)

#        if index >= 12:
    plt.xlabel('Samples per pixel', fontsize=30)

    x_labels = [specific_display_label(str(id * step_value + start_index)) for id, val in enumerate(predictions) if id % label_freq == 0]  + [specific_display_label(str(nsamples))]
    #x_labels = [id * step_value + start_index for id, val in enumerate(predictions) if id % label_freq == 0]

    x = [v for v in np.arange(0, len(predictions)) if v % label_freq == 0] + [int(nsamples / (20 * every))]
    y = np.arange(-1, 2, 10)

    plt.xticks(x, x_labels, rotation=45, fontsize=24)
    
    ax = plt.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(24)

    plt.ylim(y_min_lim, y_max_lim)

    fig.tight_layout()
    plt.savefig(output + '.pdf', dpi=100)

def main():

    parser = argparse.ArgumentParser(description="Read and compute entropy data file")

    parser.add_argument('--model', type=str, help='entropy file data with estimated threshold to read and compute')
    parser.add_argument('--method', type=str, help='method name to used', choices=cfg.features_choices_labels, default=cfg.features_choices_labels[0])
    parser.add_argument('--params', type=str, help='param of the method used', default="")
    parser.add_argument('--sequence', type=int, help='sequence length expected')
    parser.add_argument('--imnorm', type=int, help="specify if image is normalized before computing something", default=0, choices=[0, 1])
    parser.add_argument('--scene', type=str, help='Scene folder to use')
    parser.add_argument('--zone', type=int, help='zone index to use')
    parser.add_argument('--seq_norm', type=int, help='normalization sequence by features', choices=[0, 1])
    parser.add_argument('--every', type=int, help="every images only", default=1)
    parser.add_argument('--threshold', type=int, help="Expected thresholds for targeted zone", default=1000)
    parser.add_argument('--output', type=str, help="output prediction file")
    parser.add_argument('--nsamples', type=int, help="max number of samples")

    args = parser.parse_args()

    p_model      = args.model
    p_method     = args.method
    p_params     = args.params
    p_sequence   = args.sequence
    p_imnorm     = args.imnorm
    p_scene      = args.scene
    p_zone       = args.zone
    p_seq_norm   = bool(args.seq_norm)
    p_every      = args.every
    p_threshold  = args.threshold
    p_output     = args.output
    p_nsamples   = args.nsamples


    # scene path by default
    scene_path = p_scene

    if scene_path[-1] == '/':
        scene_path = scene_path[:-1]

    _, scene_name = os.path.split(p_scene)

    # 2. load model and compile it
    model = load_model(p_model)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # 4. get estimated thresholds using model and specific method
    images_path = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])
    images_path = sorted([ img for i, img in enumerate(images_path) if i % p_every == 0 ])
    
    number_of_images = len(images_path)
    image_indices = [ dt.get_scene_image_quality(img_path) for img_path in images_path ]

    blocks_sequence = []
    blocks_predictions = []
    image_counter = 0


    for img_i, img_path in enumerate(images_path):

        blocks = segmentation.divide_in_blocks(Image.open(img_path), (200, 200))

        block = blocks[p_zone]
        # for index, block in enumerate(blocks):
            
        # normalize if necessary
        if p_imnorm:
            block = np.array(block) / 255.

        blocks_sequence.append(np.array(extract_data(block, p_method, p_params)))

        # check if prediction is possible
        if len(blocks_sequence) >= p_sequence:
            data = np.array(blocks_sequence)
            
            if data.ndim == 1:
                data = data.reshape(len(blocks_sequence), 1)
            else:
                # check if sequence normalization is used
                if p_seq_norm:

                    # check snorm process
                    #for _, seq in enumerate(data):
                        
                    s, f = data.shape
                    for i in range(f):
                        #final_arr[index][]
                        data[:, i] = utils.normalize_arr_with_range(data[:, i])
                            
            data = np.expand_dims(data, axis=0)
            
            prob = model.predict(data, batch_size=1)[0][0]
            #print(index, ':', image_indices[img_i], '=>', prob)

            blocks_predictions.append(prob)

            #print('Block @', index, ':', len(blocks_sequence[index]))
            # delete first element (just like sliding window)
            del blocks_sequence[0]

        # write progress bar
        write_progress((image_counter + 1) / number_of_images)
        
        image_counter = image_counter + 1
    

    # 6. display results
    f = open(p_output, 'w')
    f.write(scene_name + ';')
    f.write(str(p_zone) + ';')
    for i, data in enumerate(blocks_predictions):
        f.write(str(data) + ';')    
    f.write('\n')
    
    # default set threshold
    display_simulation_thresholds(blocks_predictions, p_threshold, image_indices, p_output + '_figure', p_nsamples, p_every)

if __name__== "__main__":
    main()
