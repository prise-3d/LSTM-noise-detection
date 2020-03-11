# main imports
import argparse
import numpy as np
import pandas as pd

# dl imports
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from keras import backend as K
import sklearn

# def auc(y_true, y_pred):
#     return roc_auc_score(y_true, y_pred)

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
            v_data.append(np.array(vv, 'float'))
        
        final_arr.append(v_data)

    return np.array(final_arr)


def create_model(input_shape):
    print ('Creating model...')
    model = Sequential()
    #model.add(Embedding(input_dim = 1000, output_dim = 50, input_length=input_length))
    model.add(LSTM(input_shape=input_shape, output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
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

    train_dataset = pd.read_csv(p_train, header=None, sep=';')
    test_dataset = pd.read_csv(p_test, header=None, sep=';')

    train_dataset = sklearn.utils.shuffle(train_dataset)
    test_dataset = sklearn.utils.shuffle(test_dataset)

    X_train = train_dataset.loc[:, 1:].apply(lambda x: x.str.split(' '))
    X_train = build_input(X_train)
    y_train = train_dataset.loc[:, 0].astype('int')

    X_test = test_dataset.loc[:, 1:].apply(lambda x: x.str.split(' '))
    X_test = build_input(X_test)
    y_test = test_dataset.loc[:, 0].astype('int')

    input_shape = (X_train.shape[1], X_train.shape[2])
    print('Training data input shape', input_shape)
    model = create_model(input_shape)
    model.summary()

    # print ('Fitting model...')
    hist = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split = 0.3, verbose = 1)

    score, acc = model.evaluate(X_test, y_test, batch_size=1)
    print('Test score:', score)
    print('Test accuracy:', acc)

if __name__ == "__main__":
    main()