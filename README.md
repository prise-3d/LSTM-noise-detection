# LSTM noise detection

## Description

Project developed in order to use LSTM (RNN) models for noise detection into synthesis images.

## Installation


```bash
git clone --recursive https://github.com/prise-3d/LSTM-noise-detection.git
```


```bash
pip install -r requirements.txt
```

## Prepare data and train model


### Precompute the whole expected features
First you need to generate data using thresholds file (file obtained from SIN3D app):

```bash
python processing/prepare_data_file.py --dataset /path/to/folder --thresholds file.csv --method svd  --params "0,200" --imnorm 0 --output output
```

- `--output`: save automatically output into `data/generated`.
- `--params`: associated params to selected method. 
- `--imnorm`: specified if image is normalized or not before computing method.

### Well compare models
In order to well compare models, you need to set the training and testing zones used for your dataset:

```bash
python processing/generate_selected_zones_file.py --dataset /path/to/folder --n_zones 12 --output file --thresholds file.csv
```

- `--output`: save automatically output into `data/learned_zones`

Each image is cut out into 16 zones, then you need to use the `n_zones` parameter to set you number of zones selected for training part.

The generated output file contains information for each scene about indices used for training and testing sets.

### Generate your dataset

Then, you can generate your dataset:

```bash
python processing/prepare_dataset_zones.py --data data/generated/output --selected_zones data/learned_zones/file --sequence 5 --output data/datasets/name
```

- `--data`: specify the output data folder path generated when precomputing features (saved into `data/generated`).
- `--selected_zones`: the previous output file generated in order to set (if specified `n_zones` is not used).
- `--sequence`: sliding window size to use as sequence input.
- `--n_zones`: number of zones to take if ramdomly zones choice.
- `--output`: save automatically output into `data/datasets`.

### Train your model

You can now use your dataset to train your model:

```bash
python train_lstm_weighted_v2.py --train data/datasets/dataset/dataset.train --test data/datasets/dataset/dataset.train --output modelv1 --seq_norm 1
```

- `--data`: specify the dataset name (without .train and .test generated extension) obtained from previous script.
- `--output`: save automatically output into `data/saved_models`.
- `--seq_norm`: set normalization of data by feature for sequence

## Simulations


### Obtained model simulation on scene

```bash
python display/display_thresholds_scene_file.py --model data/saved_models/modelv1.joblib --method svd --params "0,200" --sequence 5 --imnorm 1 --scene /path/to/scene --selected_zones data/learned_zones/file --thresholds filename.csv --save_thresholds simulation.csv --label_thresholds "Simulate modelv1"
```

- `--folder`: scene folder to simulate on.
- `--save_thresholds`: save estimated thresholds into file.
- `--label_thresholds`: label to use for this model into the saved file.


## Contributors

* [jbuisine](https://github.com/jbuisine)

## License

[MIT](LICENSE)