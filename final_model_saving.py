
import sys
import numpy as np
import tensorflow as tf

sys.path.append(r'C:\Users\USER1\Desktop\MeNow Project')
from menow_model import MenowModel

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


# saving the final models using SavedModel format, that saves the model architecture, weights, and the traced
# Tensorflow subgraphs of the call functions. This enables Keras to restore both built-in layers as well as custom
# objects (the NeighborsAccuracy()&TrueAccuracy() custom validation metrics).


gender_latefusion_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='gender', model_type='late-fusion')
age_latefusion_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='age', model_type='late-fusion')
reg_skintype_latefusion_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='reg_skin-type', model_type='late-fusion')

models = [reg_skintype_latefusion_model, gender_latefusion_model, age_latefusion_model]
names = ['SkinType_model','Gender_model','Age_model']
best_bs = [64,32,32]
best_lr = [0.0001,0.0001,0.001]
for mod,bs,lr,name in zip(models,best_bs,best_lr,names):
    train_ds, val_ds = mod.create_datasets(batch_size=bs)

    model = mod.make_model(learning_rate=lr)  # metrics=METRICS
    print(model.summary())

    mod.fit(model, train_ds, val_ds, NAME=str(bs) + "_" + str(lr), show_plots=False)

    # Creating a SavedModel folder `name_model`.
    model.save(name)