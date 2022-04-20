from datetime import datetime
import itertools
import io

import tensorflow as tf
import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE



# load dataset
trainX_arr = np.empty((0,5,128))
trainy_arr = np.empty((0,5))
testX_arr = np.empty((0,5,128))
testy_arr = np.empty((0,5))
for n in range(10):
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
print(f'the test frac.: {(100 * testX_arr.shape[0]/(trainX_arr.shape[0] + testX_arr.shape[0])):.2f}%\n')

"""
X shape:   (vids, imgs, embeddings)
y shape:   (vidslable, imgs)
"""



class MenowModel():


    def __init__(self, trainX, trainy, testX, testy, label='skin-type', model_type='late-fusion', imbalance='oversample'):

        # sanity check
        assert len(trainX)==len(trainy) and len(testX)==len(testy), "size of X and y have to be the same"

        self.label = label
        self.model_type = model_type
        self.imbalance = imbalance

        num_frames_per_vid = trainX.shape[1]

        # Create classes arr from the dictionary arr for the appropriate label
        if self.model_type == 'single-frame':
            self.trainX, self.testX = trainX.reshape([-1, 128]), testX.reshape([-1, 128])
            self.trainy, self.testy = trainy.reshape([-1, 1]), testy.reshape([-1, 1])
            if self.label == 'skin-type':
                # no missing values
                self.trainy = np.array([int(xi[self.label]) for xi in self.trainy[:, 0]])
                self.testy = np.array([int(xi[self.label]) for xi in self.testy[:, 0]])
                if self.imbalance == 'oversample':
                    oversample = SMOTE()
                    self.trainX, self.trainy = oversample.fit_resample(self.trainX, self.trainy)
            elif self.label == 'gender':
                self.trainy = np.array([xi[self.label] for xi in self.trainy[:, 0]])
                self.testy = np.array([xi[self.label] for xi in self.testy[:, 0]])
                print(f'num of missing genders: train-{np.count_nonzero(self.trainy == "N/A")}, '
                      f'test-{np.count_nonzero(self.testy == "N/A")}')
                # deleting missing values (only for testy)
                missing_idx = np.where(self.testy == "N/A")[0]
                self.testy = np.delete(self.testy, missing_idx)
                self.testX = np.delete(self.testX, missing_idx, 0)
                self.trainy = np.array([0 if xi == "Male" else 1 for xi in self.trainy])
                self.testy = np.array([0 if xi == "Male" else 1 for xi in self.testy])
            elif self.label == 'age':
                self.trainy = np.array([xi[self.label] for xi in self.trainy[:, 0]])
                self.testy = np.array([xi[self.label] for xi in self.testy[:, 0]])
                print(f'num of missing genders: train-{np.count_nonzero(self.trainy == "N/A")}, '
                      f'test-{np.count_nonzero(self.testy == "N/A")}')
                # deleting missing values (only for testy)
                missing_idx = np.where(self.testy == "N/A")[0]
                self.testy = np.delete(self.testy, missing_idx)
                self.testX = np.delete(self.testX, missing_idx, 0)
                self.trainy = np.array([int(xi) for xi in self.trainy])
                self.testy = np.array([int(xi) for xi in self.testy])
            print(f'single-frame shapes:\n trainX shape: {self.trainX.shape}, trainy shape: {self.trainy.shape}\n '
                  f'testX shape: {self.testX.shape}, testy shape: {self.testy.shape}\n')

        elif self.model_type == 'late-fusion':
            self.trainX, self.testX = trainX.reshape([-1, 128 * num_frames_per_vid]), testX.reshape(
                [-1, 128 * num_frames_per_vid])
            self.trainy, self.testy = trainy[:, 0], testy[:, 0]
            if self.label == 'skin-type':
                # no missing values
                self.trainy = np.array([int(xi[self.label]) for xi in self.trainy])
                self.testy = np.array([int(xi[self.label]) for xi in self.testy])
                if imbalance == 'oversample':
                    oversample = SMOTE()
                    self.trainX, self.trainy = oversample.fit_resample(self.trainX, self.trainy)
            elif self.label == 'gender':
                self.trainy = np.array([xi[self.label] for xi in self.trainy])
                self.testy = np.array([xi[self.label] for xi in self.testy])
                print(f'num of missing genders: train-{np.count_nonzero(self.trainy == "N/A")}, '
                      f'test-{np.count_nonzero(self.testy == "N/A")}')
                # deleting missing values (only for testy)
                missing_idx = np.where(self.testy == "N/A")[0]
                self.testy = np.delete(self.testy, missing_idx)
                self.testX = np.delete(self.testX, missing_idx, 0)
                self.trainy = np.array([0 if xi == "Male" else 1 for xi in self.trainy])
                self.testy = np.array([0 if xi == "Male" else 1 for xi in self.testy])
                print(self.testy)
            elif self.label == 'age':
                self.trainy = np.array([xi[self.label] for xi in self.trainy])
                self.testy = np.array([xi[self.label] for xi in self.testy])
                print(f'num of missing ages: train-{np.count_nonzero(self.trainy == "N/A")}, '
                      f'test-{np.count_nonzero(self.testy == "N/A")}')
                # deleting missing values (only for testy)
                missing_idx = np.where(self.testy == "N/A")[0]
                self.testy = np.delete(self.testy, missing_idx)
                self.testX = np.delete(self.testX, missing_idx, 0)
                print(f'num of missing ages: train-{np.count_nonzero(self.trainy == "N/A")}, '
                      f'test-{np.count_nonzero(self.testy == "N/A")}')
                self.trainy = np.array([int(xi) for xi in self.trainy])
                self.testy = np.array([int(xi) for xi in self.testy])
            print(f'late-fusion shapes:\n trainX shape: {self.trainX.shape}, trainy shape: {self.trainy.shape}\n '
                  f'testX shape: {self.testX.shape}, testy shape: {self.testy.shape}\n')

        unique = np.unique(self.trainy)
        self.num_classes = len(unique)
        print(f"num of classes: {self.num_classes}")

        if self.label == 'skin-type':
            self.class_names = ['Type I', 'Type II', 'Type III', 'Type IV', 'Type V', 'Type VI']


    # Examine the class label imbalance
    def train_lable_dist(self, show_dis=False):
        if self.label == 'skin-type' or self.label == 'gender':
            unique = np.unique(self.trainy)
            bincount = np.bincount(self.trainy)
            total = bincount.sum()
            print('Examples:')
            print(f'    Total: {total}')
            count_dict = {}
            for label in unique:
                print(
                    f'    type' + str(label) + f': {bincount[label]} ({(100 * bincount[label] / total):.2f}% of total)')
                count_dict[str(label)] = bincount[label]
            return count_dict

        if self.label == "age" and show_dis == True:
            sns.displot(data=self.trainy, kde=True)
            plt.show()
            return


    # Create an input pipeline using tf.data
    def create_datasets(self, shuffle=True, batch_size=32):

        print(f'trainX.shape: {self.trainX.shape}, trainy.shape: {self.trainy.shape}')
        print(f'testX.shape: {self.testX.shape}, testy.shape: {self.testy.shape}')

        if self.label == "skin-type":
            # one-hot representation
            trainy = tf.keras.utils.to_categorical(self.trainy - 1)
            testy = tf.keras.utils.to_categorical(self.testy - 1)

        elif self.label == 'gender':
            # one-hot representation
            trainy = tf.keras.utils.to_categorical(self.trainy)
            testy = tf.keras.utils.to_categorical(self.testy)

        elif self.label == 'age':
            trainy = self.trainy
            testy = self.testy

        print(f'trainX_ds.shape: {self.trainX.shape}, trainy_ds.shape: {trainy.shape}')
        print(f'testX_ds.shape: {self.testX.shape}, testy_ds.shape: {testy.shape}')
        train_ds = tf.data.Dataset.from_tensor_slices((self.trainX, trainy))
        test_ds = tf.data.Dataset.from_tensor_slices((self.testX, testy))
        if shuffle:
            train_ds = train_ds.shuffle(buffer_size=len(self.trainX))
        train_ds = train_ds.batch(batch_size)
        test_ds = test_ds.batch(batch_size)
        train_ds = train_ds.prefetch(batch_size)
        test_ds = test_ds.prefetch(batch_size)

        return train_ds, test_ds


    # Function that creates a simple neural network
    def make_model(self, learning_rate=1e-5, metrics=['accuracy']):

        if self.model_type == 'single-frame':
            if self.label == 'skin-type':
                embedding_input = tf.keras.Input(shape=(128,), name='embedding_input')
                x = tf.keras.layers.LayerNormalization(axis=1)(embedding_input)
                x = tf.keras.layers.Dense(60, activation="relu")(x)
                # x = tf.keras.layers.Dropout(0.5)(x)
                # x = tf.keras.layers.LayerNormalization(axis=1)(embedding_input)
                # x = tf.keras.layers.Dense(64, activation="relu")(x)
                # x = tf.keras.layers.Dropout(0.5)(x)
                x = tf.keras.layers.Dense(self.num_classes)(x)
                output = tf.nn.softmax(x)
                model = tf.keras.Model(embedding_input, output)

            elif self.label == 'gender':
                embedding_input = tf.keras.Input(shape=(128,), name='embedding_input')
                x = tf.keras.layers.LayerNormalization(axis=1)(embedding_input)
                x = tf.keras.layers.Dense(60, activation="relu")(x)
                x = tf.keras.layers.Dropout(0.5)(x)
                # x = tf.keras.layers.LayerNormalization(axis=1)(embedding_input)
                # x = tf.keras.layers.Dense(64, activation="relu")(x)
                # x = tf.keras.layers.Dropout(0.5)(x)
                x = tf.keras.layers.Dense(self.num_classes)(x)
                output = tf.nn.softmax(x)
                model = tf.keras.Model(embedding_input, output)

            elif self.label == 'age':
                embedding_input = tf.keras.Input(shape=(128,), name='embedding_input')
                x = tf.keras.layers.LayerNormalization(axis=1)(embedding_input)
                x = tf.keras.layers.Dense(60, activation="relu")(x)
                x = tf.keras.layers.Dropout(0.5)(x)
                x = tf.keras.layers.LayerNormalization(axis=1)(x)
                x = tf.keras.layers.Dense(64, activation="relu")(x)
                # x = tf.keras.layers.Dropout(0.5)(x)
                output = tf.keras.layers.Dense(1)(x)
                model = tf.keras.Model(embedding_input, output)

        elif self.model_type == 'late-fusion':
            if self.label == 'skin-type':
                embedding_input = tf.keras.Input(shape=(128 * 5,), name='embedding_input')
                x = tf.keras.layers.LayerNormalization(axis=1)(embedding_input)
                x = tf.keras.layers.Dense(60, activation="relu")(x)
                x = tf.keras.layers.Dropout(0.5)(x)
                # x = tf.keras.layers.LayerNormalization(axis=1)(embedding_input)
                # x = tf.keras.layers.Dense(64, activation="relu")(x)
                # x = tf.keras.layers.Dropout(0.5)(x)
                x = tf.keras.layers.Dense(self.num_classes)(x)
                output = tf.nn.softmax(x)
                model = tf.keras.Model(embedding_input, output)

            elif self.label == 'gender':
                embedding_input = tf.keras.Input(shape=(128 * 5,), name='embedding_input')
                x = tf.keras.layers.LayerNormalization(axis=1)(embedding_input)
                x = tf.keras.layers.Dense(60, activation="relu")(x)
                x = tf.keras.layers.Dropout(0.5)(x)
                # x = tf.keras.layers.LayerNormalization(axis=1)(embedding_input)
                # x = tf.keras.layers.Dense(64, activation="relu")(x)
                # x = tf.keras.layers.Dropout(0.5)(x)
                x = tf.keras.layers.Dense(self.num_classes)(x)
                output = tf.nn.softmax(x)
                model = tf.keras.Model(embedding_input, output)

            elif self.label == 'age':
                embedding_input = tf.keras.Input(shape=(128 * 5,), name='embedding_input')
                x = tf.keras.layers.LayerNormalization(axis=1)(embedding_input)
                x = tf.keras.layers.Dense(60, activation="relu")(x)
                x = tf.keras.layers.Dropout(0.5)(x)
                x = tf.keras.layers.LayerNormalization(axis=1)(x)
                x = tf.keras.layers.Dense(64, activation="relu")(x)
                # x = tf.keras.layers.Dropout(0.5)(x)
                output = tf.keras.layers.Dense(1)(x)
                model = tf.keras.Model(embedding_input, output)

        if self.label == 'skin-type' or self.label == 'gender':
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=metrics)
        elif self.label == 'age':
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss=tf.keras.losses.MeanAbsoluteError())

        return model


    # function that converts matplotlib plot a PNG image (for cm_callback)
    def _plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image


    # function that calculates the confusion matrix (for cm_callback)
    def _plot_confusion_matrix(self, cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
          cm (array, shape = [n, n]): a confusion matrix of integer classes
          class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        # Compute the labels from the normalized confusion matrix.
        labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure


    # train the model
    def fit(self, model, train_ds, val_ds, NAME, EPOCHS=90, show_plots=True):

        def generate_class_weights(count_dict):
            n_samples = sum(list(count_dict.values()))
            weights_dict = {}
            for label in count_dict.keys():
                n_classes = len(count_dict.keys())
                if self.label == 'skin-type':
                    weights_dict[int(label) - 1] = (n_samples / (n_classes * count_dict[label]))
                elif self.label == 'gender':
                    weights_dict[int(label)] = (n_samples / (n_classes * count_dict[label]))
            return weights_dict

        count_dict = self.train_lable_dist()
        if self.label == 'skin-type' or self.label == 'gender':
            class_weights = generate_class_weights(count_dict)
            print(class_weights)

        # Callback to stop training when a monitored metric has stopped improving.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=1,
            patience=10,
            mode='min',
            restore_best_weights=True)

        logdir = 'logs_bs_lr1/{}'.format(NAME) + "/image/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # Define the basic TensorBoard callback.
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
        file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')

        class_names = self.class_names
        def log_confusion_matrix(epoch, logs):
            # Use the model to predict the values from the validation dataset
            y_preds = model.predict(val_ds)
            y_true = self.testy
            if self.label == "skin-type":
                y_true = tf.keras.utils.to_categorical(y_true - 1)
            elif self.label == "gender":
                y_true = tf.keras.utils.to_categorical(y_true)
            if self.label == 'skin-type' or self.label == 'gender':
                y_true, y_preds = np.argmax(y_true, axis=1), np.argmax(y_preds, axis=1)
            # Calculate the confusion matrix
            cm = confusion_matrix(y_true, y_preds)
            # Log the confusion matrix as an image summary
            figure = self._plot_confusion_matrix(cm, class_names=class_names)
            cm_image = self._plot_to_image(figure)
            # Log the confusion matrix as an image summary
            with file_writer_cm.as_default():
                tf.summary.image("Confusion Matrix", cm_image, step=epoch)

        # Define the per-epoch callback
        cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

        if self.imbalance == 'class_weights':
            history = model.fit(
                train_ds,
                epochs=EPOCHS,
                callbacks=[early_stopping, tensorboard_callback, cm_callback],
                validation_data=val_ds,
                class_weight=class_weights)

        else:
            history = model.fit(
                train_ds,
                epochs=EPOCHS,
                callbacks=[early_stopping, tensorboard_callback, cm_callback],
                validation_data=val_ds)

        # Produce plots of the model's loss on the training and validation set (useful to check for overfitting)
        if show_plots == True:
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            if self.label == 'skin-type' or self.label == 'gender':
                acc = history.history['accuracy']
                val_acc = history.history['val_accuracy']
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
            elif self.label == 'age':
                epochs_range = range(len(loss))
                plt.plot(epochs_range, loss, label='Training MAE')
                plt.plot(epochs_range, val_loss, label='Validation MAE')
                plt.legend(loc='upper right')
                plt.title('Training and Validation Loss')
                plt.show()

        return history


    def evaluate(self, model, plot_cm=True):

        y_preds = model.predict(val_ds)
        y_true = self.testy
        if self.label == "skin-type":
            y_true = tf.keras.utils.to_categorical(y_true - 1)
        elif self.label == "gender":
            y_true = tf.keras.utils.to_categorical(y_true)

        if self.label == 'skin-type' or self.label == 'gender':
            y_true, y_preds = np.argmax(y_true, axis=1), np.argmax(y_preds, axis=1)

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

            # Create a Classification Report
            print(classification_report(y_true, y_preds))

            print(f'true_y shape: {y_true.shape}')
            print(f'pred_y shape: {y_preds.shape}')

            if plot_cm == True:
                conf_mat = confusion_matrix(y_true, y_preds)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(conf_mat, annot=True, fmt='d')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                plt.show()

        elif self.label == 'age':
            print(y_true)
            print(type(y_true), y_true.shape)
            print(y_preds)
            print(type(y_preds), y_preds.shape)

            eval = model.evaluate(val_ds)
            print(eval)
            # check the error distribution
            # error = y_preds - y_true
            # sns.displot(data=error, kind="kde")
            # plt.show()

            error = y_preds - y_true
            plt.hist(error, bins=25)
            plt.xlabel('Prediction Error [MPG]')
            plt.ylabel('Count')




#skintype_singleframe_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='skin-type', model_type='single-frame')
skintype_latefusion_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='skin-type', model_type='late-fusion')
#gender_singleframe_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='gender', model_type='single-frame')
#gender_latefusion_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='gender', model_type='late-fusion')
#age_singleframe_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='age', model_type='single-frame')
#age_latefusion_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='age', model_type='late-fusion')



# #Checking the format of the data the df_to_dataset function returns
# train_ds, val_ds = skintype_latefusion_model.create_datasets()
# [(embedding_batch, label_batch)] = train_ds.take(1)
# print('embedding batch shape:', embedding_batch.shape)
# print('label batch shape:', label_batch.shape)
# print(len(val_ds))

# # Create, compile, and train the model
# # define the metrics
# METRICS = [
#     tf.keras.metrics.TruePositives(name='tp'),
#     tf.keras.metrics.FalsePositives(name='fp'),
#     tf.keras.metrics.TrueNegatives(name='tn'),
#     tf.keras.metrics.FalseNegatives(name='fn'),
#     tf.keras.metrics.BinaryAccuracy(name='accuracy'),
#     tf.keras.metrics.Precision(name='precision'),
#     tf.keras.metrics.Recall(name='recall'),
#     tf.keras.metrics.AUC(name='auc'),
#     tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
# ]


for bs in [32,64,128]:
    for lr in [1e-2,1e-3,1e-4,1e-5]:
        train_ds, val_ds = skintype_latefusion_model.create_datasets(batch_size=32)

        model = skintype_latefusion_model.make_model(learning_rate=lr)   # metrics=METRICS
        print(model.summary())

        skintype_latefusion_model.fit(model, train_ds, val_ds, NAME=str(bs)+"_"+str(lr), show_plots=False)

        skintype_latefusion_model.evaluate(model, plot_cm=False)






