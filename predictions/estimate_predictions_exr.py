# main imports
import numpy as np
import pandas as pd
import sys, os, argparse

# image processing
from PIL import Image
from ipfml import utils
import Imath
import OpenEXR
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

def get_center_image(img, width=200, height=200):

    _, w, h = img.shape
    return img[:, int(w/2) - int(width/2):int(w/2) + int(width/2), int(h/2) - int(height/2):int(h/2) + int(height/2)]

def read_image(img_path):

    extension = img_path.split('.')[-1]

    if extension == 'exr':
        
        if not OpenEXR.isOpenExrFile(img_path):
            raise ValueError(f'Image {img_path} is not a valid OpenEXR file')

        src = OpenEXR.InputFile(img_path)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        dw = src.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        
        # Read into numpy array
        #img = np.zeros((3, size[1], size[0]))
        img_channels = []
        for i, c in enumerate('RGB'):
            rgb32f = np.fromstring(src.channel(c, pixel_type), dtype=np.float32)
            #img[i, :, :] = rgb32f.reshape(size[1], size[0])
            img_channels.append(rgb32f.reshape(size[1], size[0]))

        img_data = np.array(img_channels)
        img_data = np.array(img_data / np.max(img_data), 'float32')

        img_data = get_center_image(img_data)
        img_data = np.moveaxis(img_data, 0, -1)

        return img_data
            
    if extension == 'png':
        img_data = Image.open(img_path)
        img_data = np.moveaxis(img_data, -1, 0)
        return np.array(img_data / 255., 'float32')

    return None


def main():

    parser = argparse.ArgumentParser(description="Read and compute entropy data file")

    parser.add_argument('--model', type=str, help='entropy file data with estimated threshold to read and compute')
    parser.add_argument('--method', type=str, help='method name to used', choices=cfg.features_choices_labels, default=cfg.features_choices_labels[0])
    parser.add_argument('--params', type=str, help='param of the method used', default="")
    parser.add_argument('--sequence', type=int, help='sequence length expected')
    parser.add_argument('--imnorm', type=int, help="specify if image is normalized before computing something", default=0, choices=[0, 1])
    parser.add_argument('--scenes', type=str, help='Scenes names to use')
    parser.add_argument('--dataset', type=str, help='dataset folder to use')
    parser.add_argument('--seq_norm', type=int, help='normalization sequence by features', choices=[0, 1])
    parser.add_argument('--every', type=int, help="every images only", default=1)
    parser.add_argument('--output', type=str, help="output prediction file")

    args = parser.parse_args()

    p_model      = args.model
    p_method     = args.method
    p_params     = args.params
    p_sequence   = args.sequence
    p_imnorm     = args.imnorm
    p_dataset    = args.dataset
    p_scenes     = args.scenes
    p_seq_norm   = bool(args.seq_norm)
    p_every      = args.every
    p_output     = args.output


    # scene path by default
    scenes_list = []

    with open(p_scenes, 'r') as f:
        for l in f.readlines():
            scenes_list.append(l.split(';')[0])

    # 2. load model and compile it
    model = load_model(p_model)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # 4. get estimated thresholds using model and specific method
    for s_i, scene in enumerate(sorted(scenes_list)):
        
        scenes_images = []

        # store all expected prediction for all zones of the image
        model_predictions = []

        print('----')
        print(f'    -- Load data for scene {s_i + 1} of {len(scenes_list)}...')

        scene_path = os.path.join(p_dataset, scene)

        images_names = sorted(os.listdir(scene_path))

        # read all images of scene
        for img in images_names:
            img_path = os.path.join(scene_path, img)
            scenes_images.append(img_path)
   
        blocks_sequence = []
        # blocks_predictions = []
        image_counter = 0

        # append empty list

        for img_i, img_path in enumerate(scenes_images):

            block = read_image(img_path)
   
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

                model_predictions.append(prob)

                #print('Block @', index, ':', len(blocks_sequence[index]))
                # delete first element (just like sliding window)
                del blocks_sequence[0]

            # write progress bar
            write_progress((image_counter + 1) / len(images_names))
            
            image_counter = image_counter + 1
        

        # 6. display results
        if not os.path.exists(p_output):
            os.makedirs(p_output)

        # be default only one zone
        with open(os.path.join(p_output, 'predictions.csv'), 'a') as f:
            
            f.write(f'{scene};0')

            for v in model_predictions:
                f.write(f';{v}')
                    
            f.write('\n')


if __name__== "__main__":
    main()
