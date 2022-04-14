
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
from imblearn.over_sampling import SMOTE




"""
single-frame model
"""


# load dataset
trainX_arr = np.empty((0,5,128))
trainy_arr = np.empty((0,5))
testX_arr = np.empty((0,5,128))
testy_arr = np.empty((0,5))
for n in range(5):
    if n == 0:
        dataset_name = 'faces-embeddings4parts_.npz'
    else:
        dataset_name = 'faces-embeddings4parts_' + str(n) + '.npz'
    data = np.load(dataset_name, allow_pickle=True)
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print(dataset_name + ': train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
    trainX_arr = np.concatenate((trainX_arr, trainX), axis=0)
    trainy_arr = np.concatenate((trainy_arr, trainy), axis=0)
    testX_arr = np.concatenate((testX_arr, testX), axis=0)
    testy_arr = np.concatenate((testy_arr, testy), axis=0)
    print(trainX_arr.shape, trainy_arr.shape, testX_arr.shape, testy_arr.shape)
print(f'\nthe test frac.: {(100 * testX_arr.shape[0]/(trainX_arr.shape[0] + testX_arr.shape[0])):.2f}%')

"""
X shape:   (vids, imgs, embeddings)
y shape:   (vidslable, imgs)
"""


class MenowModel():


    def __init__(self, trainX, trainy, testX, testy, label='skin-type', model_type='single-frame', imbalance='None'):

        # sanity check
        assert len(trainX)==len(trainy) and len(testX)==len(testy), "size of X and y have to be the same"

        self.label = label
        self.model_type = model_type

        # Split the data into training, validation, and test sets
        trainX, valX, trainy, valy = train_test_split(trainX, trainy, test_size=0.2)
        print(len(trainX), 'training video examples')
        print(len(valX), 'validation video examples')
        print(len(testX), 'test video examples')

        num_frames_per_vid = trainX.shape[1]

        # Create classes arr from the dictionary arr for the appropriate label
        if self.model_type == 'single-frame':
            self.trainX, self.valX, self.testX = trainX.reshape([-1, 128]), valX.reshape([-1, 128]), testX.reshape([-1, 128])
            self.trainy, self.valy, self.testy = trainy.reshape([-1, 1]), valy.reshape([-1, 1]), testy.reshape([-1, 1])
            if self.label == 'skin-type':
                self.trainy = np.array([int(xi[self.label]) for xi in self.trainy[:, 0]])
                self.valy = np.array([int(xi[self.label]) for xi in self.valy[:, 0]])
                self.testy = np.array([int(xi[self.label]) for xi in self.testy[:, 0]])
                if imbalance == 'oversample':
                    oversample = SMOTE()
                    self.trainX, self.trainy = oversample.fit_resample(self.trainX, self.trainy)
            # elif self.label == 'gender':
            #     self.yarr =
            # elif self.label == 'age':
            #     self.yarr =:
            print(f'single-frame Xarr shape: {self.trainX.shape}, single-frame yarr shape: {self.trainy.shape}')

        elif self.model_type == 'late-fusion':
            self.trainX, self.valX, self.testX = trainX.reshape([-1, 128*num_frames_per_vid]), valX.reshape([-1, 128*num_frames_per_vid]), testX.reshape([-1, 128*num_frames_per_vid])
            self.trainy, self.valy, self.testy = trainy[:,0], valy[:,0], testy[:,0]
            if self.label == 'skin-type':
                self.trainy = np.array([int(xi[self.label]) for xi in self.trainy])
                self.valy = np.array([int(xi[self.label]) for xi in self.valy])
                self.testy = np.array([int(xi[self.label]) for xi in self.testy])
                if imbalance == 'oversample':
                    oversample = SMOTE()
                    self.trainX, self.trainy = oversample.fit_resample(self.trainX, self.trainy)
            print(f'late-fusion Xarr shape: {self.trainX.shape}, late-fusion yarr shape: {self.trainy.shape}')





    # Examine the class label imbalance
    def train_lable_dist(self):
        if self.label == 'skin-type':
            unique = np.unique(self.trainy)
            bincount = np.bincount(self.trainy)
            total = bincount.sum()
            print('Examples:')
            print(f'    Total: {total}')
            count_dict = {}
            for label in unique:
                print(f'    type' + str(label) + f': {bincount[label]} ({(100 * bincount[label]/total):.2f}% of total)')
                count_dict[str(label)] = bincount[label]
        # if self.label == "gender":
        # if self.label == "age":
        return count_dict

    # Create an input pipeline using tf.data
    def create_datasets(self, shuffle=True, batch_size=32):
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
                x = tf.keras.layers.Dense(60, activation="relu")(x)
                # x = tf.keras.layers.Dropout(0.5)(x)
                # x = tf.keras.layers.LayerNormalization(axis=1)(embedding_input)
                # x = tf.keras.layers.Dense(64, activation="relu")(x)
                # x = tf.keras.layers.Dropout(0.5)(x)
                x = tf.keras.layers.Dense(num_classes)(x)
                output = tf.nn.softmax(x)
                model = tf.keras.Model(embedding_input, output)

        elif self.model_type == 'late-fusion':
            num_classes = 6
            embedding_input = tf.keras.Input(shape=(128*5,), name='embedding_input')
            x = tf.keras.layers.LayerNormalization(axis=1)(embedding_input)
            x = tf.keras.layers.Dense(60, activation="relu")(x)
            # x = tf.keras.layers.Dropout(0.5)(x)
            # x = tf.keras.layers.LayerNormalization(axis=1)(embedding_input)
            # x = tf.keras.layers.Dense(64, activation="relu")(x)
            # x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(num_classes)(x)
            output = tf.nn.softmax(x)
            model = tf.keras.Model(embedding_input, output)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=metrics)

        return model


#skintype_singleframe_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='skin-type', model_type='single-frame')
skintype_latefusion_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='skin-type', model_type='late-fusion', imbalance='oversample')


# #Checking the format of the data the df_to_dataset function returns
# train_ds, val_ds, test_ds = skintype_singleframe_model.create_datasets()
# [(embedding_batch, label_batch)] = train_ds.take(1)
# print('embedding batch shape:', embedding_batch.shape)
# print('label batch shape:', label_batch.shape)
# print(len(test_ds))


def arr_to_dataset(Xarr, yarr, label='skin-type', model_type='late-fusion',  shuffle=True, batch_size=32):

    if model_type == 'single-frame':
        print(f'Xarr.shape: {Xarr.shape}, yarr.shape: {yarr.shape}')
        Xarr = Xarr.reshape([-1, 128])
        yarr = yarr.reshape([-1, 1])

        if label == "skin-type":
            #yarr = np.array([int(xi[label]) for xi in yarr[:,0]])
            print(f'Xarr shape: {Xarr.shape}, yarr shape: {yarr.shape}')
            unique = np.unique(yarr)
            num_classes = len(unique)
            print(f"num of classes: {num_classes}")
            # one-hot representation
            print(f'to cat yarr shape: {yarr.shape}')
            yarr = tf.keras.utils.to_categorical(yarr - 1)

    elif model_type == 'late-fusion':
        print(f'Xarr.shape: {Xarr.shape}, yarr.shape: {yarr.shape}')
        if label == "skin-type":
            print(f'Xarr shape: {Xarr.shape}, yarr shape: {yarr.shape}')
            unique = np.unique(yarr)
            num_classes = len(unique)
            print(f"num of classes: {num_classes}")
            # one-hot representation
            print(f'to cat yarr shape: {yarr.shape}')
            yarr = tf.keras.utils.to_categorical(yarr - 1)

    print(Xarr.shape, yarr.shape)
    ds = tf.data.Dataset.from_tensor_slices((Xarr, yarr))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(Xarr))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds
batch_size = 256
train_ds = arr_to_dataset(skintype_latefusion_model.trainX, skintype_latefusion_model.trainy)
val_ds = arr_to_dataset(skintype_latefusion_model.valX, skintype_latefusion_model.valy, shuffle=False, batch_size=batch_size)
test_ds = arr_to_dataset(skintype_latefusion_model.testX, skintype_latefusion_model.testy, shuffle=False, batch_size=batch_size)
[(embedding_batch, label_batch)] = train_ds.take(1)
print('embedding batch shape:', embedding_batch.shape)
print('label batch shape:', label_batch.shape)
print(len(test_ds))


# Create, compile, and train the model

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

# Model summary
model = skintype_latefusion_model.make_model()   # metrics=METRICS
print(model.summary())


def generate_class_weights(count_dict):
    n_samples = sum(list(count_dict.values()))
    weights_dict = {}
    for label in count_dict.keys():
        n_classes = len(count_dict.keys())
        weights_dict[int(label)-1] = (n_samples / (n_classes * count_dict[label]))
    return weights_dict


count_dict = skintype_latefusion_model.train_lable_dist()

class_weights = generate_class_weights(count_dict)
print(class_weights)


# train the model
EPOCHS = 200

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
    validation_data=val_ds,
#    class_weight=class_weights
)

# Produce plots of the model's loss on the training and validation set (useful to check for overfitting),
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

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
true_y = skintype_latefusion_model.testy
print(pred_y.shape, true_y.shape)
true_y = tf.keras.utils.to_categorical(true_y - 1)
print(pred_y.shape, true_y.shape)
evaluate_preds(np.argmax(true_y, axis=1), np.argmax(pred_y, axis=1))


# Create a Classification Report
print(classification_report(np.argmax(true_y, axis=1), np.argmax(pred_y, axis=1)))

print(f'true_y shape: {np.argmax(true_y, axis=1).shape}')
print(f'pred_y shape: {np.argmax(true_y, axis=1).shape}')


conf_mat = confusion_matrix(np.argmax(true_y, axis=1), np.argmax(pred_y, axis=1))
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()



