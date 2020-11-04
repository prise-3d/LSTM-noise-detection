import os, sys
import argparse

import numpy as np

from ipfml import utils

from keras.models import load_model
from sklearn.metrics import roc_auc_score, accuracy_score

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

def predict_ensemble(models, data, cluster_id, weight):
    """Predict label from list of models (return probability of label)

    Args:
        models ([Sequential]): models list
        data ([float]): input data for model
        cluster_id (int): cluster id of current zone (in order to target specific model)
        weight (int): weight of targeted cluster model
    """
    prob_sum = 0.
    clusters_weight = len(models) - 1 + weight

    for i in range(len(models)):
        
        prob = models[i].predict(data, batch_size=1)[0][0]

        # if cluster id model is targeted
        if i == cluster_id:
            prob = prob * weight

        #print('[clusterId: {0}] Model {1} predicts => {2}'.format(cluster_id, i, prob))

        prob_sum += prob
    
    return prob_sum / clusters_weight

def main():

    parser = argparse.ArgumentParser(description="Compute KPI from ensemble model")

    # 1. From generated cluster data:
    # Compute dataset and keep from file the cluster ID
    # Separate Train and Test dataset using learned zones csv file
    # - data folder
    # - nclusters
    # - human thresholds
    parser.add_argument('--data', type=str, help='folder path with all clusters data', required=True)
    parser.add_argument('--nclusters', type=int, help='number of clusters', required=True)
    parser.add_argument('--sequence', type=int, help='sequence length expected', required=True)
   
    # 2. Prepare input dataset
    # - selected_zones file
    # - sequence size
    # - seqnorm or not
    parser.add_argument('--selected_zones', type=str, help='file which contains selected zones indices for training part', required=True)
    parser.add_argument('--seq_norm', type=int, help='normalization sequence by features', choices=[0, 1])

    # 3. Load ensemble models
    # - models folder
    # - weight for identified cluster model
    parser.add_argument('--models', type=str, help='folder with all models', required=True)
    parser.add_argument('--weight', type=float, help='associated weight to cluster model of zone', default=2.5)

    # 4. Compute each KPI from labels
    # - output results file
    parser.add_argument('--output', type=str, help='output prediction file', required=True)

    args = parser.parse_args()

    # 1. Generate dataset input from precomputer data
    p_data        = args.data
    p_nclusters   = args.nclusters
    p_sequence    = args.sequence

    # 2. Dataset preparation
    p_selected_zones = args.selected_zones
    p_seq_norm       = bool(args.seq_norm)

    # 3. Models param
    p_models      = args.models
    p_weight      = args.weight

    # 4. Output KPI of models
    p_output      = args.output

    # PART 1. Load generated data
    print('# PART 1: load generated data')

    """
    cluster_data = [
        (cluster_id, scene_name, zone_id, label, data)
    ]
    """
    clusters_data = []
    for c_id in range(p_nclusters):

        cluster_filepath = os.path.join(p_data, 'cluster_data_{0}.csv'.format(c_id))

        with open(cluster_filepath, 'r') as f:

            lines = f.readlines()

            for line in lines:

                data = line.split(';')

                # only if scene is used for training part
                scene_name = data[0]
                zones_index = int(data[1])
                threshold = int(data[3])
                image_indices = data[4].split(',')
                values_list = data[5].split(',')

                # compute sequence data here
                sequence_data = []

                for i, index in enumerate(image_indices):
                    
                    values = values_list[i].split(' ')

                    # append new sequence
                    sequence_data.append(values)

                    if i + 1 >= p_sequence:

                        label = int(threshold > int(index))

                        if len(sequence_data) != p_sequence:
                            print("error found => ", len(sequence_data))

                        # Add new data line into cluster data
                        clusters_data.append((c_id, scene_name, zones_index, label, sequence_data.copy()))
        
                        del sequence_data[0]

    # PART 2. Prepare input data
    print('# PART 2: preparation of input data')

    # extract learned zones from file
    zones_learned = {}

    if len(p_selected_zones) > 0:
        with open(p_selected_zones, 'r') as f:
            lines = f.readlines()

            for line in lines:
                data = line.replace(';\n', '').split(';')

                scene_name = data[0]
                zones_learned[scene_name] = [ int(z) for z in data[1:] ]

    """
        XXXX_dataset = [
            (cluster_id, label, data)
        ]
    """
    train_dataset = []
    test_dataset = []

    n_samples = len(clusters_data)

    for index_line, line in enumerate(clusters_data):

        data = np.array(line[-1], 'float32')
                    
        if data.ndim == 1:
            data = data.reshape(len(data[-1]), 1)
        else:
            # check if sequence normalization is used
            if p_seq_norm:
                    
                s, f = data.shape
                for i in range(f):
                    #final_arr[index][]
                    data[:, i] = utils.normalize_arr_with_range(data[:, i])
                        
        data = np.expand_dims(data, axis=0)

        if p_seq_norm:
            data_line = (line[0], line[3], data)
        else:
            data_line = (line[0], line[3], data)

        # use of stored zone index of scene
        zone_id = line[2]
        scene_name = line[1]
        if zone_id in zones_learned[scene_name]:
            train_dataset.append(data_line)
        else:
            test_dataset.append(data_line)

        write_progress((index_line + 1) / n_samples)
    print()

    print('=> Training set is of {:2f} of total samples'.format(len(train_dataset) / n_samples))
    print('=> Testing set is of {:2f} of total samples'.format(len(test_dataset) / n_samples))

    # PART 3. Load ensemble models
    print('# PART 3: load ensemble models and do predictions')

    # Load each classification model
    models_path = sorted([ os.path.join(p_models, m) for m in os.listdir(p_models)])

    models_list = []

    for model_p in models_path:

        model = load_model(model_p)
        model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

        models_list.append(model)

    # do predictions
    counter_sample = 0
    y_train_predicted = []
    y_test_predicted = []

    for line in train_dataset:

        cluster_id = line[0]
        data = line[-1]
        prob = predict_ensemble(models_list, data, cluster_id, p_weight)
        #print(index, ':', image_indices[img_i], '=>', prob)

        if prob < 0.5:
            y_train_predicted.append(0)
        else:
            y_train_predicted.append(1)

        counter_sample += 1
        write_progress((counter_sample + 1) / n_samples)

    for line in test_dataset:

        cluster_id = line[0]
        data = line[-1]
        prob = predict_ensemble(models_list, data, cluster_id, p_weight)
        #print(index, ':', image_indices[img_i], '=>', prob)

        if prob < 0.5:
            y_test_predicted.append(0)
        else:
            y_test_predicted.append(1)

        counter_sample += 1
        write_progress((counter_sample + 1) / n_samples)

    # PART 4. Get KPI using labels comparisons
    y_train = [ line[1] for line in train_dataset ]
    y_test = [ line[1] for line in test_dataset ]

    auc_train = roc_auc_score(y_train, y_train_predicted)
    auc_test = roc_auc_score(y_test, y_test_predicted)

    acc_train = accuracy_score(y_train, y_train_predicted)
    acc_test = accuracy_score(y_test, y_test_predicted)
    
    print('Train ACC:', acc_train)
    print('Train AUC', auc_train)
    print('Test ACC:', acc_test)
    print('Test AUC:', auc_test)
    
    if not os.path.exists(p_output):
        with open(p_output, 'w') as f:
            f.write('name;acc_train;auc_train;acc_test;auc_test\n')
    
    with open(p_output, 'a') as f:
        f.write('{0};{1};{2};{3};{4}\n'.format(p_models, acc_train, auc_train, acc_test, auc_test))


if __name__ == "__main__":
    main()