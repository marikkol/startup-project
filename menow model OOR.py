
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report



"""
single-frame model
"""


# load dataset
data = np.load('faces-embeddings4parts_.npz', allow_pickle=True)
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

print(trainX.shape, trainy.shape, testX.shape, testy.shape)

data = np.load('faces-embeddings4parts_1.npz', allow_pickle=True)
trainX1, trainy1, testX1, testy1 = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX1.shape[0], testX1.shape[0]))

print(trainX1.shape, trainy1.shape, testX1.shape, testy1.shape)

trainX = np.concatenate((trainX, trainX1), axis=0)
trainy = np.concatenate((trainy, trainy1), axis=0)
testX = np.concatenate((testX, testX1), axis=0)
testy = np.concatenate((testy, testy1), axis=0)

print(trainX.shape, trainy.shape, testX.shape, testy.shape)

"""
X shape:   (vids, imgs, embeddings)
y shape:   (vidslable, imgs)
"""


# # convert to object oriented (class single_frame_model)
# def single_frame_model(trainX, trainy, testX, testy, lable):

class MenowModel():


    def __init__(self, trainX, trainy, testX, testy, label='skin-type', model_type='single-frame'):

        # sanity check
        assert len(trainX)==len(trainy) and len(testX)==len(testy), "size of X and y hve to be the same"

        self.label = label
        self.model_type = model_type

        # Split the data into training, validation, and test sets
        trainX, valX, trainy, valy = train_test_split(trainX, trainy, test_size=0.2)
        print(len(trainX), 'training video examples')
        print(len(valX), 'validation video examples')
        print(len(testX), 'test video examples')

        # Create classes arr from the dictionary arr for the appropriate label
        if self.model_type == 'single-frame':
            self.trainX, self.valX, self.testX = trainX.reshape([-1, 128]), valX.reshape([-1, 128]), testX.reshape([-1, 128])
            self.trainy, self.valy, self.testy = trainy.reshape([-1, 1]), valy.reshape([-1, 1]), testy.reshape([-1, 1])
            if self.label == 'skin-type':
                self.trainy = np.array([int(xi[self.label]) for xi in self.trainy[:, 0]])
                self.valy = np.array([int(xi[self.label]) for xi in self.valy[:, 0]])
                self.testy = np.array([int(xi[self.label]) for xi in self.testy[:, 0]])
            # elif self.label == 'gender':
            #     self.yarr =
            # elif self.label == 'age':
            #     self.yarr =:
            print(f'Xarr shape: {self.trainX.shape}, yarr shape: {self.trainy.shape}')


    # Examine the class label imbalance
    def show_train_lable_dist(self):
        if self.label == 'skin-type':
            unique = np.unique(self.trainy)
            bincount = np.bincount(self.trainy)
            total = bincount.sum()
            print('Examples:')
            print(f'    Total: {total}')
            for label in unique:
                print(f'    type' + str(label) + f': {bincount[label]} ({(100 * bincount[label]/total):.2f}% of total)')
        # if self.label == "gender":
        # if self.label == "age":

    # Create an input pipeline using tf.data
    def create_datasets(self, shuffle=True, batch_size=64):
        ds_list = list()
        Xs = [self.trainX, self.valX, self.testX]
        ys = [self.trainy, self.valy, self.testy]
        for Xarr, yarr in zip(Xs,ys):
            print(f'Xarr shape: {Xarr.shape}, yarr shape: {yarr.shape}')
            if self.model_type == 'single-frame':
                if self.label == 'skin-type':
                    # one-hot representation
                    print(f'to cat yarr shape: {yarr.shape}')
                    yarr = yarr.reshape([-1, 1])
                    yarr = tf.keras.utils.to_categorical(yarr - 1)
                # elif self.label == 'gender':
                # elif self.label == 'age':
            elif self.model_type == 'late-fusion':
                continue

            print(Xarr.shape, yarr.shape)
            ds = tf.data.Dataset.from_tensor_slices((Xarr, yarr))
            if shuffle:
                ds = ds.shuffle(buffer_size=len(Xarr))
            ds = ds.batch(batch_size)
            ds = ds.prefetch(batch_size)
            ds_list.append(ds)

        return ds_list



    # Function that creates a simple neural network
    def make_model(self, metrics=['accuracy']):
        if self.model_type == 'single-frame':
            if self.label == 'skin-type':
                num_classes = 6
                embedding_input = tf.keras.Input(shape=(128,), name='embedding_input')
                x = tf.keras.layers.LayerNormalization(axis=1)(embedding_input)
                x = tf.keras.layers.Dense(64, activation="relu")(x)
                x = tf.keras.layers.Dropout(0.5)(x)
                # x = tf.keras.layers.LayerNormalization(axis=1)(x)
                x = tf.keras.layers.Dense(32, activation="relu")(x)
                x = tf.keras.layers.Dropout(0.5)(x)
                x = tf.keras.layers.Dense(num_classes)(x)
                output = tf.nn.sigmoid(x)
                model = tf.keras.Model(embedding_input, output)

        # elif model_type == 'late-fusion':
        #     # merge the list of feature inputs
        #     all_features = tf.keras.layers.concatenate(encoded_features)
        #     x = tf.keras.layers.Dense(32, activation="relu")(all_features)
        #     x = tf.keras.layers.Dropout(0.5)(x)
        #     x = tf.keras.layers.Dense(1)(x)
        #     output = tf.nn.sigmoid(x)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=metrics)

        return model


skintype_singleframe_model = MenowModel(trainX, trainy, testX, testy, label='skin-type', model_type='single-frame')

skintype_singleframe_model.show_train_lable_dist()


#Checking the format of the data the df_to_dataset function returns
train_ds, val_ds, test_ds = skintype_singleframe_model.create_datasets()
[(embedding_batch, label_batch)] = train_ds.take(1)
print('embedding batch shape:', embedding_batch.shape)
print('label batch shape:', label_batch.shape)
print(len(test_ds))


# def arr_to_dataset(Xarr, yarr, label='skin-type', model_type='single-frame',  shuffle=True, batch_size=32):
#
#     if model_type == 'single-frame':
#         Xarr = Xarr.reshape([-1, 128])
#         yarr = yarr.reshape([-1, 1])
#
#         if label == "skin-type":
#             #yarr = np.array([int(xi[label]) for xi in yarr[:,0]])
#             print(f'Xarr shape: {Xarr.shape}, yarr shape: {yarr.shape}')
#             unique = np.unique(yarr)
#             num_classes = len(unique)
#             print(f"num of classes: {num_classes}")
#             # one-hot representation
#             print(f'to cat yarr shape: {yarr.shape}')
#             yarr = tf.keras.utils.to_categorical(yarr - 1)
#
#     elif model_type == 'late-fusion':
#         yarr = yarr[:,0]
#     print(Xarr.shape, yarr.shape)
#     ds = tf.data.Dataset.from_tensor_slices((Xarr, yarr))
#     if shuffle:
#         ds = ds.shuffle(buffer_size=len(Xarr))
#     ds = ds.batch(batch_size)
#     ds = ds.prefetch(batch_size)
#     return ds
# batch_size = 64
# train_ds = arr_to_dataset(skintype_singleframe_model.trainX, skintype_singleframe_model.trainy)
# val_ds = arr_to_dataset(skintype_singleframe_model.valX, skintype_singleframe_model.valy, shuffle=False, batch_size=batch_size)
# test_ds = arr_to_dataset(skintype_singleframe_model.testX, skintype_singleframe_model.testy, shuffle=False, batch_size=batch_size)


# Create, compile, and train the model

# Model summary
model = skintype_singleframe_model.make_model()
print(model.summary())

# define the metrics
METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]

# train the model
EPOCHS = 60

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=val_ds)

# Produce plots of the model's loss on the training and validation set (useful to check for overfitting),
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# create an evaluation function to output all the needs metrics

def evaluate_preds(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels
    on a classification.
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds, average='micro')
    recall = recall_score(y_true, y_preds, average='micro')
    f1 = f1_score(y_true, y_preds, average='micro')
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2),
                   "recall": round(recall, 2),
                   "f1": round(f1, 2)}
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")

    return metric_dict

# Now we make predictions using the test data to see how the model performs
pred_y = model.predict(test_ds)
true_y = skintype_singleframe_model.testy
print(pred_y.shape, true_y.shape)
true_y = tf.keras.utils.to_categorical(true_y - 1)
print(pred_y.shape, true_y.shape)
evaluate_preds(np.argmax(true_y, axis=1), np.argmax(pred_y, axis=1))


# Create a Classification Report
print(classification_report(np.argmax(true_y, axis=1), np.argmax(pred_y, axis=1)))

conf_mat = confusion_matrix(np.argmax(true_y, axis=1), np.argmax(pred_y, axis=1))
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()



# # Apply the Keras preprocessing layers
# # perform feature-wise normalization of input features.
#
# def get_normalization_layer(vid_num, dataset):
#     normalizer = layers.Normalization(axis=None)
#     feature_ds = dataset.map(lambda x, y: x[vid_num])
#     normalizer.adapt(feature_ds)
#     return normalizer
#
# # Test the get_normalization_layer function
# vid_1 = train_features['age']
# layer = get_normalization_layer('age', train_ds)
# layer(age_col)