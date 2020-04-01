# main imports
import argparse
import numpy as np
import pandas as pd
import os

# dl imports
from keras.layers import Dense, Dropout, LSTM, Embedding, GRU
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn.metrics import roc_auc_score, accuracy_score
import tensorflow as tf
from keras import backend as K
import sklearn
from joblib import dump

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

    return np.array(final_arr)


def create_model(input_shape):
    print ('Creating model...')
    model = Sequential()
    #model.add(Embedding(input_dim = 1000, output_dim = 50, input_length=input_length))
    model.add(GRU(input_shape=input_shape, units=512, activation='sigmoid', recurrent_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(units=256, activation='sigmoid', recurrent_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    print ('Compiling...')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def main():

    parser = argparse.ArgumentParser(description="Read and compute training of LSTM model")

    parser.add_argument('--train', type=str, help='input train dataset')
    parser.add_argument('--test', type=str, help='input test dataset')
    parser.add_argument('--output', type=str, help='output model name')
   
    args = parser.parse_args()

    p_train        = args.train
    p_test         = args.test
    p_output       = args.output

    dataset_train = pd.read_csv(p_train, header=None, sep=';')
    dataset_test = pd.read_csv(p_test, header=None, sep=';')

    # getting weighted class over the whole dataset
    noisy_df_train = dataset_train[dataset_train.iloc[:, 0] == 1]
    not_noisy_df_train = dataset_train[dataset_train.iloc[:, 0] == 0]
    nb_noisy_train = len(noisy_df_train.index)
    nb_not_noisy_train = len(not_noisy_df_train.index)

    noisy_df_test = dataset_test[dataset_test.iloc[:, 0] == 1]
    not_noisy_df_test = dataset_test[dataset_test.iloc[:, 0] == 0]
    nb_noisy_test = len(noisy_df_test.index)
    nb_not_noisy_test = len(not_noisy_df_test.index)

    noisy_samples = nb_noisy_test + nb_noisy_train
    not_noisy_samples = nb_not_noisy_test + nb_not_noisy_train

    total_samples = noisy_samples + not_noisy_samples

    print('noisy', noisy_samples)
    print('not_noisy', not_noisy_samples)
    print('total', total_samples)

    class_weight = {
        0: noisy_samples / float(total_samples),
        1: (not_noisy_samples / float(total_samples)),
    }

    # shuffle data
    final_df_train = sklearn.utils.shuffle(dataset_train)
    final_df_test = sklearn.utils.shuffle(dataset_test)

    # split dataset into X_train, y_train, X_test, y_test
    X_train = final_df_train.loc[:, 1:].apply(lambda x: x.astype(str).str.split(' '))
    X_train = build_input(X_train)
    y_train = final_df_train.loc[:, 0].astype('int')

    X_test = final_df_test.loc[:, 1:].apply(lambda x: x.astype(str).str.split(' '))
    X_test = build_input(X_test)
    y_test = final_df_test.loc[:, 0].astype('int')

    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    input_shape = (X_train.shape[1], X_train.shape[2])
    print('Training data input shape', input_shape)
    model = create_model(input_shape)
    model.summary()

    print("Fitting model with custom class_weight", class_weight)
    hist = model.fit(X_train, y_train, batch_size=32, epochs=150, validation_split = 0.2, verbose = 1, shuffle=True, class_weight=class_weight)


    train_score, train_acc = model.evaluate(X_train, y_train, batch_size=1)

    print(train_acc)
    y_train_predict = model.predict_classes(X_train)
    y_test_predict = model.predict_classes(X_test)
    y_all_predict = model.predict_classes(X_all)

    print(y_train_predict)
    print(y_test_predict)

    auc_train = roc_auc_score(y_train, y_train_predict)
    auc_test = roc_auc_score(y_test, y_test_predict)
    auc_all = roc_auc_score(y_all, y_all_predict)

    acc_train = accuracy_score(y_train, y_train_predict)
    acc_test = accuracy_score(y_test, y_test_predict)
    acc_all = accuracy_score(y_all, y_all_predict)
    
    print('Train ACC:', acc_train)
    print('Train AUC', auc_train)
    print('Test ACC:', acc_test)
    print('Test AUC:', auc_test)
    print('All ACC:', acc_all)
    print('All AUC:', auc_all)


    # save model results
    if not os.path.exists(cfg.output_results_folder):
        os.makedirs(cfg.output_results_folder)

    results_filename = os.path.join(cfg.output_results_folder, cfg.results_filename)

    with open(results_filename, 'a') as f:
        f.write(p_output + ';' + str(acc_train) + ';' + str(auc_train) + ';' + str(acc_test) + ';' + str(auc_test) + '\n')

    # save model using joblib
    if not os.path.exists(cfg.output_models):
        os.makedirs(cfg.output_models)

    dump(model, os.path.join(cfg.output_models, p_output + '.joblib'))

if __name__ == "__main__":
    main()