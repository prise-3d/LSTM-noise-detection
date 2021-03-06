# main imports
import argparse
import numpy as np
import pandas as pd
import os
import ctypes

from keras import backend as K
import matplotlib.pyplot as plt
from ipfml import utils

# dl imports
from keras.layers import Dense, Dropout, LSTM, Embedding, GRU, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import keras
from sklearn.metrics import roc_auc_score, accuracy_score
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split

import custom_config as cfg


def build_input(df):
    """Convert dataframe to numpy array input with timesteps as float array
    
    Arguments:
        df: {pd.Dataframe} -- Dataframe input
    
    Returns:
        {np.ndarray} -- input LSTM data as numpy array
    """

    arr = df.to_numpy()

    final_arr = []
    for v in arr:
        v_data = []
        for vv in v:
            #scaled_vv = np.array(vv, 'float') - np.mean(np.array(vv, 'float'))
            #v_data.append(scaled_vv)
            v_data.append(vv)
        
        final_arr.append(v_data)
    
    final_arr = np.array(final_arr, 'float32')            

    return final_arr

def build_label(x):
    index = list(x).index(max(x))

    output = []

    for i in range(len(x)):
        if index == i:
            output.append(1)
        else:
            output.append(0)

    return output


def create_model(input_shape):
    print ('Creating model...')
    model = Sequential()
    #model.add(Embedding(input_dim = 1000, output_dim = 50, input_length=input_length))
    model.add(LSTM(input_shape=input_shape, units=128, activation='tanh', recurrent_activation='sigmoid', dropout=0.4, return_sequences=True))
    model.add(LSTM(units=32, activation='tanh', recurrent_activation='sigmoid', dropout=0.4, return_sequences=True))
    model.add(LSTM(units=8, activation='tanh', dropout=0.4, recurrent_activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))

    print ('Compiling...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  #metrics=['accuracy', tf.keras.metrics.AUC()])
                  metrics=['accuracy'])

    return model


def main():

    parser = argparse.ArgumentParser(description="Read and compute training of LSTM model")

    parser.add_argument('--train', type=str, help='input train dataset')
    parser.add_argument('--test', type=str, help='input test dataset')
    parser.add_argument('--output', type=str, help='output model name')
    parser.add_argument('--epochs', type=int, help='number of expected epochs', default=30)
    parser.add_argument('--batch_size', type=int, help='expected batch size for training model', default=64)

    args = parser.parse_args()

    p_train        = args.train
    p_test         = args.test
    p_output       = args.output
    p_epochs       = args.epochs
    p_batch_size   = args.batch_size

    dataset_train = pd.read_csv(p_train, header=None, sep=';')
    dataset_test = pd.read_csv(p_test, header=None, sep=';')

    # getting weighted class over the whole dataset
    # line is composed of :: [scene_name; zone_id; image_index_end; label; data]
    noisy_df_train = dataset_train[dataset_train.iloc[:, 3] == 1]
    interval_df_train = dataset_train[dataset_train.iloc[:, 3] == 0]
    not_noisy_df_train = dataset_train[dataset_train.iloc[:, 3] == 2]
    nb_noisy_train = len(noisy_df_train.index)
    nb_interval_train = len(interval_df_train.index)
    nb_not_noisy_train = len(not_noisy_df_train.index)

    noisy_df_test = dataset_test[dataset_test.iloc[:, 3] == 1]
    interval_df_test = dataset_test[dataset_test.iloc[:, 3] == 2]
    not_noisy_df_test = dataset_test[dataset_test.iloc[:, 3] == 0]
    nb_noisy_test = len(noisy_df_test.index)
    nb_interval_test = len(interval_df_test.index)
    nb_not_noisy_test = len(not_noisy_df_test.index)

    noisy_samples = nb_noisy_test + nb_noisy_train
    interval_samples = nb_interval_test + nb_interval_train
    not_noisy_samples = nb_not_noisy_test + nb_not_noisy_train

    total_samples = noisy_samples + interval_samples + not_noisy_samples

    print('noisy', noisy_samples)
    print('interval', interval_samples)
    print('not_noisy', not_noisy_samples)
    print('total', total_samples)

    class_weight = {
        0: noisy_samples / float(total_samples),
        1: (not_noisy_samples / float(total_samples)),
        2: (interval_samples / float(total_samples)),
    }

    # shuffle data
    final_df_train = sklearn.utils.shuffle(dataset_train)
    final_df_test = sklearn.utils.shuffle(dataset_test)

    # split dataset into X_train, y_train, X_test, y_test
    X_train_all = final_df_train.loc[:, 4:].apply(lambda x: x.astype(str).str.split(' '))
    X_train_all = build_input(X_train_all)
    #y_train_all = final_df_train.loc[:, 3].apply(lambda x: build_label(x))
    y_train_all = tf.keras.utils.to_categorical(final_df_train.loc[:, 3], num_classes=3)

    X_test = final_df_test.loc[:, 4:].apply(lambda x: x.astype(str).str.split(' '))
    X_test = build_input(X_test)
    #y_test = final_df_test.loc[:, 3].apply(lambda x: build_label(x))
    y_test = tf.keras.utils.to_categorical(final_df_test.loc[:, 3], num_classes=3)

    input_shape = (X_train_all.shape[1], X_train_all.shape[2])
    print('Training data input shape', input_shape)
    model = create_model(input_shape)
    model.summary()

    # prepare train and validation dataset
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.3, shuffle=False)

    print("Fitting model with custom class_weight", class_weight)
    history = model.fit(X_train, y_train, batch_size=p_batch_size, epochs=p_epochs, validation_data=(X_val, y_val), verbose=1, shuffle=True, class_weight=class_weight)

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    # train_score, train_acc = model.evaluate(X_train, y_train, batch_size=1)

    # print(train_acc)
    # TODO: improve this part
    y_train_predict = [ build_label(x) for x in model.predict(X_train) ]
    y_val_predict = [ build_label(x) for x in model.predict(X_val) ]
    y_test_predict = [ build_label(x) for x in model.predict(X_test) ]

    # print(y_train_predict)
    # print(y_test_predict)

    auc_train = roc_auc_score(y_train, y_train_predict)
    auc_val = roc_auc_score(y_val, y_val_predict)
    auc_test = roc_auc_score(y_test, y_test_predict)

    acc_train = accuracy_score(y_train, y_train_predict)
    acc_val = accuracy_score(y_val, y_val_predict)
    acc_test = accuracy_score(y_test, y_test_predict)
    
    print('Train ACC:', acc_train)
    print('Train AUC', auc_train)
    print('Val ACC:', acc_val)
    print('Val AUC', auc_val)
    print('Test ACC:', acc_test)
    print('Test AUC:', auc_test)

    from sklearn.metrics import confusion_matrix

    y_test_predict_matrix = [ list(x).index(max(x)) for x in y_test_predict ]
    y_test_matrix = [ list(x).index(max(x)) for x in y_test ]
    output_matrix = confusion_matrix(y_test_matrix, y_test_predict_matrix, labels=["not noisy", "noisy", "interval"])
    print(output_matrix)

    # save model using h5
    if not os.path.exists(cfg.output_models):
        os.makedirs(cfg.output_models)

    model.save(os.path.join(cfg.output_models, p_output + '.h5'))

    # save model results
    if not os.path.exists(cfg.output_results_folder):
        os.makedirs(cfg.output_results_folder)

    results_filename_path = os.path.join(cfg.output_results_folder, cfg.results_filename)

    # write header if necessary
    if not os.path.exists(results_filename_path):
        with open(results_filename_path, 'w') as f:
            f.write('name;train_acc;val_acc;test_acc;train_auc;val_auc;test_auc;\n')

    with open(results_filename_path, 'a') as f:
        f.write(p_output + ';' + str(acc_train) + ';' + str(acc_val) + ';' + str(acc_test) + ';' \
             + str(auc_train) + ';' + str(auc_val) + ';' + str(auc_test) + '\n')


if __name__ == "__main__":
    main()