
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


#skintype_singleframe_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='skin-type', model_type='single-frame')
#skintype_latefusion_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='skin-type', model_type='late-fusion')
#gender_singleframe_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='gender', model_type='single-frame')
#gender_latefusion_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='gender', model_type='late-fusion')
#age_singleframe_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='age', model_type='single-frame')
age_latefusion_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='age', model_type='late-fusion')
#reg_skintype_latefusion_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='reg_skin-type', model_type='late-fusion')


tf.random.set_seed(1234)

NeighborsAcc_scores = list()
TrueAcc_scores = list()
MAE_scores = list()
bs_lr = list()
for bs in [32,64,128]:
    for lr in [1e-2,1e-3,1e-4,1e-5]:
        train_ds, val_ds = age_latefusion_model.create_datasets(batch_size=bs)

        model = age_latefusion_model.make_model(learning_rate=lr)   # metrics=METRICS
        print(model.summary())

        age_latefusion_model.fit(model, train_ds, val_ds, NAME=str(bs)+"_"+str(lr), show_plots=False)

        TrueAcc_score, NeighborsAcc_score, MAE = age_latefusion_model.evaluate(model, val_ds, plot_cm=False)
        TrueAcc_score = TrueAcc_score.numpy()
        NeighborsAcc_score = NeighborsAcc_score.numpy()

        bs_lr.append(str(bs) + '_' + str(lr))
        TrueAcc_scores.append(TrueAcc_score)
        NeighborsAcc_scores.append(NeighborsAcc_score)
        MAE_scores.append(MAE)


print(f"The True-Accuracy scores: {TrueAcc_scores}\n"
      f"The Neighbors-Accuracy scores: {NeighborsAcc_scores}\n"
      f"The MAE scores: {MAE_scores}\n"
      f"bs_lr: {bs_lr}")
