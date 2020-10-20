import os
import subprocess
import argparse


n_epochs = 30
features_list = ['svd_entropy_blocks', 'filters_statistics']

generated_datasets_folder = "data/datasets/{0}/{1}"
results_filename = "data/results/results_{0}.csv"

already_runned = []

for feature in features_list:

    if not os.path.exists(results_filename.format(feature)):
        os.system('touch {0}'.format(results_filename.format(feature)))

    with open(results_filename.format(feature), 'r') as f:
        
        lines = f.readlines()

        if len(lines) == 0:
            break

        del lines[0] # remove header

        for line in lines:
            name = line.split(';')[0]
            already_runned.append(name)


def main():
    
    parser = argparse.ArgumentParser(description="For each data files found, datasets are created and RNN model trained")

    parser.add_argument('--folder', type=str, help="data folder with data files", required=True)
    parser.add_argument('--dataset', type=str, help='folder dataset with all scenes', required=True)
    parser.add_argument('--selected_zones', type=str, help='list of train scenes used', required=True)
    parser.add_argument('--thresholds', type=str, help='thresholds files used', required=True)

    args = parser.parse_args()

    p_folder            = args.folder
    p_dataset           = args.dataset
    selected_zones_file = args.selected_zones
    thresholds_file     = args.thresholds

    # generate all data files into data/generated folder
    sv_list = {
        10: [(0, 10), (2, 10)],
        20: [(0, 20), (5, 20)],
        40: [(0, 40), (10, 40)],
        100: [(0, 100), (25, 100)],
    }

    for feature in features_list:

        feature_label = feature

        for norm in [0, 1]:

            if norm:
                feature_label += "_norm"

            param_label = ""
            
            if 'svd_entropy_blocks' in feature:

                for bsize in [10, 20, 40, 100]:

                    for SV in sv_list[bsize]:
                        begin, end = SV

                        param_label = "{0}_{1}_{2}_{3}".format(bsize, bsize, begin, end)
                        param_command = "{0},{1},{2},{3}".format(bsize, bsize, begin, end)

                        output_filename = feature_label + "_" + param_label

                        command_str = "python processing/prepare_data_file.py --dataset {0} --thresholds {1} --method {2} --params \"{3}\" --output {4}" \
                            .format(p_dataset, thresholds_file, feature_label, param_command, output_filename)
                        os.system(command_str)
            
            if 'filters_statistics' in feature:

                for norm in [0, 1]:

                    if norm:
                        feature_label += "_norm"

                    output_filename = feature_label + "_" + param_label

                    command_str = "python processing/prepare_data_file.py --dataset {0} --thresholds {1} --method {2} --params \"{3}\" --output {4}" \
                        .format(p_dataset, thresholds_file, feature_label, param_label, output_filename)
                    os.system(command_str)

    # get all data files
    data_files = sorted(os.listdir(p_folder))

    for data in data_files:
        
        data_filename = os.path.join(p_folder, data)

        for seq in range(3, 8):

            output_dataset_name = data + "_seq" + str(seq)

            if not os.path.exists(os.path.join('data', 'datasets', output_dataset_name)):

                command_dataset = "python processing/prepare_dataset_zones.py --data {0} --output {1} --sequence {2} --selected_zones {3}" \
                    .format(data_filename, output_dataset_name, seq, selected_zones_file)

                
                print("Prepare dataset with name {0}".format(output_dataset_name))
                # print(command_dataset)
                subprocess.call(command_dataset, shell=True)
            else:
                print("Dataset {0} already generated".format(output_dataset_name))

            for seqnorm in [0, 1]:

                # run pytorch RNN model with generated dataset file
                output_dataset_name_path = generated_datasets_folder.format(output_dataset_name, output_dataset_name)

                for b_size in [64, 128]:
                    
                     # specific output model name
                    output_model_name = output_dataset_name

                    if seqnorm:
                        output_model_name += "_seqnorm"

                    output_model_name += "_bsize" + str(b_size)

                    if output_model_name not in already_runned:
                        
                        command_run_model = "python train_lstm_weighted.py --train {0}.train --test {1}.test --output {2} --batch_size {3} --epochs {4} --seq_norm {5}" \
                            .format(output_dataset_name_path, output_dataset_name_path, output_model_name, b_size, n_epochs, seqnorm)

                        print("Start running model {0}".format(output_model_name))
                        # print(command_run_model)
                        subprocess.call(command_run_model, shell=True)
                    else:
                        print('{0} model already trained'.format(output_dataset_name))

if __name__ == "__main__":
    main()