import os
import sys

os.chdir(os.path.split(__file__)[0])

import glob
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

from .... import utils
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import json
import requests
import keras

if len(sys.argv) == 1:
    num_epochs = 20
else:
    num_epochs = int(sys.argv[1])

glove = utils.load_vectorizer.load_glove('../../../data/glove.6B.50d.txt')
# ner = utils.load_data.load_ner_data('../../../data/')

with open('training.txt', encoding='utf-8') as f:
    training_data = f.read()

training_df = pd.DataFrame([x.split() for x in training_data.split('\n') if x],
                           columns=['word', 'tag'])
training_df.iloc[:, 1] = training_df.iloc[:, 1].apply(int)
training_df['all_tags'] = 'NA'

ner = training_df # pd.concat([training_df, ner]).reset_index(drop=True)
ner = ner[['word', 'all_tags', 'tag']]

ner = pd.concat([ner,
                 pd.DataFrame(ner['word'].str[0].str.isupper().values),
                 pd.DataFrame(glove.reindex(ner['word'].str.lower()).values)],
                axis='columns')
ner.fillna(value=0.0, inplace=True)

data_dim = 51
timesteps = 25 # only during training, testing can take arbitrary length.
num_classes = 2

train_val_split = int(19 * ner.shape[0] / 20.)

ner_train_idxs = range(0, train_val_split - timesteps, timesteps)
x_train = np.array([ner.iloc[i:i+timesteps, 3:].values
                    for i in ner_train_idxs])
y_train = np.array([to_categorical(ner.iloc[i:i+timesteps, 2].values, 2)
                    for i in ner_train_idxs])

ner_val_idxs = range(train_val_split, ner.shape[0] - timesteps, timesteps)
x_val = np.array([ner.iloc[i:i+timesteps, 3:].values
                  for i in ner_val_idxs])
y_val = np.array([to_categorical(ner.iloc[i:i+timesteps, 2].values, 2)
                  for i in ner_val_idxs])

model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(None, data_dim)))
model.add(LSTM(8, return_sequences=True))
model.add(TimeDistributed(Dense(2, activation='softmax')))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])
print(model.summary(100))

checkpointer = ModelCheckpoint(filepath='./saved/weights-{epoch:02d}.hdf5',
                               monitor='val_auc',
                               mode='max',
                               verbose=1,
                               save_best_only=True)

with open('validation.txt', encoding='utf-8') as f:
    s = f.read()
val_words = [w for w in s.split('\n') if w]

gloved_data = pd.concat(
    [pd.DataFrame([[w[0].isupper()] for w in val_words]),
     glove.reindex([w for w in val_words]).fillna(0).reset_index(drop=True)],
    axis='columns'
)


class OurAUC(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # Go to https://geo-extract-tester.herokuapp.com/ and download
        # the validation data (validation.txt).

        glove_time_size = 100
        preds_batched = []
        i = 0
        while gloved_data[i:i+glove_time_size].size:
            preds_batched.append(
                model.predict(np.expand_dims(gloved_data[i:i+glove_time_size],
                                             axis=0))[0][:, 1]
            )
            i += glove_time_size

        with open('guesses-{epoch:02d}.txt'.format(epoch=epoch), 'w') as f:
            for prob in [p for pred in preds_batched for p in pred]:
                f.write(str(prob) + '\n')

        with open('guesses-{epoch:02d}.txt'.format(epoch=epoch), 'rb') as f:
            url = 'https://geo-extract-tester.herokuapp.com/api/score'
            r = requests.post(url, files={'file': f})
            r = json.loads(r.text)
            auc = r['auc']
            print('AUC: {:.5f}, high score? {}'.format(auc, r['high_score']))

        os.remove('guesses-{epoch:02d}.txt'.format(epoch=epoch))
        logs['val_auc'] = auc


our_auc = OurAUC()

model.fit(x_train, y_train,
          epochs=num_epochs,
          validation_data=(x_val, y_val),
          callbacks=[our_auc, checkpointer],
          verbose=2)

idx = slice(501, 550)
pd.set_option('display.width', 200)
df_to_print = pd.DataFrame(
    model.predict(np.expand_dims(ner.iloc[idx, 3:].values, 0))[0][:, 1:],
    columns=['prob_geloc']
)
print(pd.concat([ner.iloc[idx, :3].reset_index(drop=True), df_to_print],
                axis='columns'))
