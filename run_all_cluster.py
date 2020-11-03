import os
import subprocess
import json

import argparse

default_parameters = {
    "dataset": {
        "scenes": "../../thesis-data/Scenes",
        "thresholds": "../../thesis-data/previous_thresholds"
    },
    "cluster": {
        "estimators": ['l_mean', 'sobel', 'l_kolmogorov', 'l_variance'],
        "model": "data/models/kmeans_model.joblib",
        "nclusters": 4
    },
    "method": {
        "name": "svd_entropy_blocks",
        "params": [20, 20, 0, 20],
        "sequence": 5
    },
    "training": {
        "percent": 0.66, # percent of training part
        "epochs": 10,
        "batch_size": 64,
        "seqnorm": 1
    },
    "output": {
        "data": "cluster_data",
        "datasets": "clusters_datasets",
        "models": "clusters_models",
        "simulations": "clusters_simulation",
        "label_simulations": "clusters_svd_entropy_blocks"
    }
}


def main():

    parser = argparse.ArgumentParser(description="Run all clusters processing / training / simulation")

    parser.add_argument('--json', type=str, help='custom json parameter', default='')
    parser.add_argument('--simulations', type=int, help='simulations after training', choices=[0, 1], default=1)
    parser.add_argument('--only', type=int, help='simulations only', choices=[0, 1], default=0)

    args = parser.parse_args()

    p_json        = args.json
    p_simulations = bool(args.simulations)
    p_only        = bool(args.only)

    # 0. Initialisation with use of custom or not params
    if len(p_json) > 0:
        with open(p_json, 'r') as f:
            # extract parameters
            parameters = json.loads(f.read())
    else:
        parameters = default_parameters

    if not p_only:
        # 1. First step we need to prepare the file data using dataset

        # Parameter list of `prepare_data_file_cluster.py` script:
        """
        parser.add_argument('--dataset', type=str, help='folder dataset with all scenes', required=True)
        parser.add_argument('--cluster', type=str, help='clustering model to use', required=True)
        parser.add_argument('--nclusters', type=int, help='number of clusters', required=True)
        parser.add_argument('--estimators', type=str, help='list of estimators', default='l_mean,l_variance')
        parser.add_argument('--thresholds', type=str, help='file which contains all thresholds', required=True)
        parser.add_argument('--method', type=str, help='method name to used', choices=cfg.features_choices_labels, default=cfg.features_choices_labels[0])
        parser.add_argument('--params', type=str, help='param of the method used', default="", required=True)
        parser.add_argument('--imnorm', type=int, help="specify if image is normalized before computing something", default=0, choices=[0, 1])
        parser.add_argument('--output', type=str, help='output folder name with all clusters files', required=True)
        """
        command_str_data = "python prepare_data_file_cluster.py --dataset {0} --cluster {1} --nclusters {2} --estimators \"{3}\" --thresholds {4} " \
                        "--method {5} --params \"{6}\" --imnorm 0 --output {7}"

        command_data = command_str_data.format(
                            parameters['dataset']['scenes'],
                            parameters['cluster']['model'],
                            parameters['cluster']['nclusters'],
                            ','.join(parameters['cluster']['estimators']),
                            parameters['dataset']['thresholds'],
                            parameters['method']['name'],
                            ','.join(list(map(str, parameters['method']['params']))),
                            parameters['output']['data']
        )

        print('#########################################')
        print('#1. Running data preparation')
        print('#########################################')
        subprocess.call(command_data, shell=True)

        # 2. Prepare dataset from generated data file

        # Parameter list of `prepare_dataset_cluster.py` script:
        """
        parser.add_argument('--folder', type=str, help='folder with cluster file')
        parser.add_argument('--output', type=str, help='output folder data')
        parser.add_argument('--sequence', type=int, help='sequence length expected')
        parser.add_argument('--percent', type=float, help='percent of zones to select per scene for each cluster data file', default=0.75)
        """

        command_str_datasets = "python prepare_dataset_cluster.py --folder data/generated/{0} --sequence {1} --percent {2} --output {3}"

        command_datasets = command_str_datasets.format(
                        parameters['output']['data'],
                        parameters['method']['sequence'],
                        parameters['training']['percent'],
                        parameters['output']['datasets']
        )

        print('#########################################')
        print('#2. Preparation of datasets')
        print('#########################################')
        subprocess.call(command_datasets, shell=True)


        # 3. Training of each datasets

        # Parameter list of `train_lstm_cluster.py` script:

        """
        parser.add_argument('--data', type=str, help='data will all cluster data files')
        parser.add_argument('--nclusters', type=int, help='number of expected clusters')
        parser.add_argument('--output', type=str, help='output folder name')
        parser.add_argument('--epochs', type=int, help='number of expected epochs', default=30)
        parser.add_argument('--batch_size', type=int, help='expected batch size for training model', default=64)
        parser.add_argument('--seq_norm', type=int, help='normalization sequence by features', choices=[0, 1])
        """

        command_str_train = "python train_lstm_cluster.py --data data/datasets/{0} --nclusters {1} --epochs {2} --batch_size {3} " \
            "--seq_norm {4} --output {5}"

        command_train = command_str_train.format(
                        parameters['output']['datasets'],
                        parameters['cluster']['nclusters'],
                        parameters['training']['epochs'],
                        parameters['training']['batch_size'],
                        parameters['training']['seqnorm'],
                        parameters['output']['models']
        )
        
        print('#########################################')
        print('#3. Training each cluster model')
        print('#########################################')
        subprocess.call(command_train, shell=True)

    if p_simulations:
        # 4. Prediction using ensemble models

        # Parameter list of `prediction_scene_cluster.py` script:

        """
        parser.add_argument('--scene', type=str, help='folder path with all scenes', required=True)
        parser.add_argument('--cluster', type=str, help='clustering model to use', required=True)
        parser.add_argument('--nclusters', type=int, help='number of clusters', required=True)
        parser.add_argument('--estimators', type=str, help='list of estimators', default='l_mean,l_variance')
        parser.add_argument('--thresholds', type=str, help='file which contains all thresholds', required=True)
        parser.add_argument('--method', type=str, help='method name to used', choices=cfg.features_choices_labels, default=cfg.features_choices_labels[0])
        parser.add_argument('--params', type=str, help='param of the method used', default="", required=True)
        parser.add_argument('--sequence', type=int, help='sequence length expected', required=True)
        parser.add_argument('--seq_norm', type=int, help='normalization sequence by features', choices=[0, 1])
        parser.add_argument('--n_stop', type=int, help='n elements for stopping criteria', default=1)
        parser.add_argument('--models', type=str, help='folder with all models', required=True)
        parser.add_argument('--imnorm', type=int, help="specify if image is normalized before computing something", default=0, choices=[0, 1])
        parser.add_argument('--output', type=str, help='output folder name with predictions', required=True)
        """

        command_str_predict = "python prediction_scene_cluster.py --scene {0} --cluster {1} --nclusters {2} --estimators {3} " \
            "--thresholds {4} --method {5} --params {6} --sequence {7} --seq_norm {8} --n_stop 1 " \
            "--models data/saved_models/{9} --imnorm 0 --output {10} --label {11}"

        scenes_path = [ os.path.join(parameters['dataset']['scenes'], s) for s in sorted(os.listdir(parameters['dataset']['scenes'])) ]

        print('#########################################')
        print('#4. Simulation of scenes')
        print('#########################################')

        for scene in scenes_path:
            command_predict = command_str_predict.format(
                            scene,
                            parameters['cluster']['model'],
                            parameters['cluster']['nclusters'],
                            ','.join(parameters['cluster']['estimators']),
                            parameters['dataset']['thresholds'],
                            parameters['method']['name'],
                            ','.join(list(map(str, parameters['method']['params']))),
                            parameters['method']['sequence'],
                            parameters['training']['seqnorm'],
                            parameters['output']['models'],
                            parameters['output']['simulations'],
                            parameters['output']['label_simulations']
            )

            print('Simulation for {0}'.format(scene.split('/')[-1]))
            subprocess.call(command_predict, shell=True)

    
if __name__ == "__main__":
    main()