import os
import sys

import glob

import pandas as pd
from keras.models import Model
from keras.layers import Conv1D, UpSampling1D, Input, Concatenate, BatchNormalization, Embedding
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
import numpy as np
import json
import requests
import keras

os.chdir(os.path.split(__file__)[0])
saved_files = glob.glob('saved/weights*.hdf5')
if saved_files:
    delete = input(('This will delete existing saved weight'
                    ' files, proceed? [y/n] '))
    while delete not in ['y', 'n']:
        delete = input(('This will delete existing saved weight'
                        ' files, proceed? [y/n] '))
    if delete == 'y':
        for f in saved_files:
            os.remove(f)
    else:
        print('Exiting.')
        exit()

if len(sys.argv) == 1:
    num_epochs = 100
else:
    num_epochs = int(sys.argv[1])

with open('training.txt', encoding='utf-8') as f:
    training_data = f.read()

training_df = pd.DataFrame([x.split() for x in training_data.split('\n') if x],
                           columns=['word', 'tag'])
training_df.iloc[:, 1] = training_df.iloc[:, 1].apply(int)
training_df['all_tags'] = 'NA'

ner = training_df
ner = ner[['word', 'all_tags', 'tag']]
ner['word'] += ' '

T = Tokenizer(char_level=True,
              filters='!"#$%&()*+-/:;<=>?@[\]^_`{|}~ ',
              lower=False)

T.fit_on_texts(ner['word'].values)
T.num_dims = T.texts_to_matrix('a').shape[1]

timesteps = 128
batch_size = 32

train_val_split = int(19 * ner.shape[0] / 20.)
# Add some padding so random sampling from 0 to train_val_split
# is guaranteed to always have enough characters
train_val_split -= np.where(ner['word'].str.len()[train_val_split::-1].cumsum() > timesteps)[0][0]
ner['cumsum'] = ner['word'].str.len().cumsum()

# ner_val_idxs = range(train_val_split, ner.shape[0] - timesteps, timesteps)
# x_val = np.array([ner.iloc[i:i+timesteps, 3:].values
#                   for i in ner_val_idxs])
# y_val = np.array([to_categorical(ner.iloc[i:i+timesteps, 2].values, 2)
#                   for i in ner_val_idxs])


def train_generator():
    while True:
        X = np.zeros((batch_size, timesteps))
        Y = np.zeros((batch_size, timesteps))
        for i in range(batch_size):
            start = np.random.randint(train_val_split - 1)
            stop = start + 1
            while ner['cumsum'][stop] - ner['cumsum'][start] < timesteps:
                stop += 1
            subset = ner.loc[start:stop, :]
            X[i] = np.argmax(T.texts_to_matrix(''.join(subset['word']))[:timesteps], axis=-1)
            Y[i] = np.concatenate([[x['tag']] * len(x['word'])
                                   for _, x in subset.iterrows()])[:timesteps]
        yield X, Y[:, :, np.newaxis]


def make_model():
    down_kwargs = {'strides': 2, 'padding': 'same', 'activation': 'relu'}
    up_kwargs = {'strides': 1, 'padding': 'same', 'activation': 'relu'}
    stable_kwargs = {'strides': 1, 'padding': 'same', 'activation': 'relu'}
    inp = Input(shape=(None,))
    k = 11
    filters = [8, 16, 32]
    down_layers = [Embedding(T.num_dims, 4)(inp)]
    for f in filters:
        x = Conv1D(f, k, **down_kwargs)(down_layers[-1])
        x = Conv1D(f, k, **stable_kwargs)(x)
        x = Conv1D(f, k, **stable_kwargs)(x)
        x = BatchNormalization(momentum=0.9)(x)
        down_layers.append(x)
    up_layers = [down_layers[-1]]
    for i, f in enumerate(filters[::-1]):
        x = UpSampling1D(size=2)(up_layers[i])
        x = Conv1D(f // 2, k, **up_kwargs)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Concatenate()([x, down_layers[-(i + 2)]])
        x = Conv1D(f, k, **stable_kwargs)(x)
        x = Conv1D(f, k, **stable_kwargs)(x)
        x = BatchNormalization(momentum=0.9)(x)
        up_layers.append(x)
    out = Conv1D(1, 1, strides=1, padding='same', activation='sigmoid')(up_layers[-1])
    model = Model(input=inp, output=out)

    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    print(model.summary(100))
    return model


checkpointer = ModelCheckpoint(filepath='./saved/weights-{epoch:02d}.hdf5',
                               monitor='val_auc',
                               mode='max',
                               verbose=1,
                               save_best_only=True)

with open('validation.txt', encoding='utf-8') as f:
    s = f.read()
val_words = [w for w in s.split('\n') if w]
assert not any(' ' in w for w in val_words), 'No words can have spaces!'


class OurAUC(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # Go to https://geo-extract-tester.herokuapp.com/ and download
        # the validation data (validation.txt).
        txt = ' '.join(val_words)
        txt_mat = T.texts_to_matrix(txt)
        space = T.texts_to_matrix(' ')[0]
        split_inds = np.where((txt_mat == space).all(axis=1))[0]
        txt_mat = np.argmax(txt_mat, axis=-1)
        n = 3
        while 2**n <= len(txt):
            n += 1
        n -= 1
        l = 2**n

        preds1 = self.model.predict(txt_mat[np.newaxis, :l])
        preds2 = self.model.predict(txt_mat[np.newaxis, len(txt) - l:])
        preds = np.concatenate([preds1[0, :], preds2[0, -(len(txt) - l):]])
        preds_per_word = [x.mean() for x in np.split(preds, split_inds)]

        with open('guesses-{epoch:02d}.txt'.format(epoch=epoch), 'w') as f:
            for prob in preds_per_word:
                f.write(str(prob) + '\n')

        with open('guesses-{epoch:02d}.txt'.format(epoch=epoch), 'rb') as f:
            url = 'https://geo-extract-tester.herokuapp.com/api/score'
            try:
                raw = requests.post(url, files={'file': f})
            except OSError:
                print('Lost connection, upload guesses directly to get AUC')
                auc = -np.inf
            else:
                r = json.loads(raw.text)
                if 'auc' not in r:
                    raise RuntimeError('Unexpected response:\n{}'.format(raw.text))
                auc = r['auc']
                print('AUC: {:.5f}, high score? {}'.format(auc, r['high_score']))

        logs['val_auc'] = auc


def train(model):
    our_auc = OurAUC()

    model.fit_generator(
        train_generator(),
        steps_per_epoch=250,
        epochs=num_epochs,
        callbacks=[our_auc, checkpointer],
        verbose=2
    )


def main():
    model = make_model()
    train(model)


if __name__ == '__main__':
    main()
