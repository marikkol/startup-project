
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

sys.path.append(r'C:\Users\USER1\Desktop\MeNow Project')
from menow_model import MenowModel, NeighborsAccuracy, TrueAccuracy

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


# k-fold cross validation

# creating a a total Dataset by concatenating the train_ds dataset with val_ds dataset, and splitting up into k-folds.
def create_Kfolds(train_ds, val_ds, num_folds=5):
    tot_ds = train_ds.concatenate(val_ds)
    ds_size = sum(1 for _ in tot_ds)
    tot_ds = tot_ds.shuffle(ds_size)
    fold_size = int(ds_size/num_folds)
    folds = []
    for fold in range(num_folds):
        if fold==0:
            fold_ds = tot_ds.take(fold_size)
        elif fold==(num_folds-1):
            fold_ds = tot_ds.skip(fold_size).skip(fold_size)
        else:
            fold_ds = tot_ds.skip(fold_size).take(fold_size)
        folds.append(fold_ds)
    return folds


# reconstruct the models identically.
SkinType_model = tf.keras.models.load_model('SkinType_model',
                 custom_objects={"NeighborsAccuracy": NeighborsAccuracy, "TrueAccuracy": TrueAccuracy})
Gender_model = tf.keras.models.load_model('Gender_model')
Age_model = tf.keras.models.load_model('Age_model')


reg_skintype_latefusion_model = MenowModel(trainX_arr, trainy_arr, testX_arr, testy_arr, label='reg_skin-type', model_type='late-fusion')
train_ds, val_ds = reg_skintype_latefusion_model.create_datasets(batch_size=64)

folds_list  = create_Kfolds(train_ds, val_ds)

repeats = 30
neighbors_scoers = list()
true_scoers = list()
for i in range(repeats):
    run_neighbors_scoers = list()
    run_true_scoers = list()
    for i in range(len(folds_list)):
        l = folds_list.copy()
        val = l.pop(i)
        if len(l)==4:
            fold0_ds, fold1_ds, fold2_ds, fold3_ds = l[0], l[1], l[2], l[3]
            ds1 = fold0_ds.concatenate(fold1_ds)
            ds2 = fold2_ds.concatenate(fold3_ds)
            train = ds1.concatenate(ds2)
        model = reg_skintype_latefusion_model.make_model(learning_rate=0.0001)
        print(model.summary())
        reg_skintype_latefusion_model.fit(model, train_ds, val_ds, NAME=str(64) + "_" + str(0.0001), show_plots=False)
        true_skill, neighbors_skill, _ = reg_skintype_latefusion_model.evaluate(model, val_ds, plot_cm=False)
        true_skill = true_skill.numpy()
        neighbors_skill = neighbors_skill.numpy()
        run_neighbors_scoers.append(neighbors_skill)
        run_true_scoers.append(true_skill)
    neighbors_mean = np.mean(run_neighbors_scoers)
    true_mean = np.mean(run_true_scoers)
    print(f"The True-Accuracy run scores: {run_true_scoers}, the mean: {true_mean}\n"
          f"The Neighbors-Accuracy run scores: {run_neighbors_scoers}, the mean: {neighbors_mean}\n")
    true_mean = np.round(true_mean, 2)
    neighbors_mean = np.round(neighbors_mean, 2)
    true_scoers.append(true_mean)
    neighbors_scoers.append(neighbors_mean)

print(f"The True-Accuracy scores: {true_scoers}\n"
      f"The Neighbors-Accuracy scores: {neighbors_scoers}\n")

true_scoers = np.array(true_scoers)
neighbors_scoers = np.array(neighbors_scoers)
np.savez_compressed('Evaluation_SkinType_ModelStability.npz', true_scoers, neighbors_scoers)

# plot results
data = np.load('Evaluation_SkinType_ModelStability.npz', allow_pickle=True)
d = {'true_scoers': data['arr_0']*100, 'neighbors_scoers': data['arr_1']*100}
df = pd.DataFrame(data=d)

means = np.array([np.mean(df.true_scoers.values), np.mean(df.neighbors_scoers.values)])
means = np.round(means, 2)
stds = np.array([np.std(df.true_scoers.values), np.std(df.neighbors_scoers.values)])
stds = np.round(stds, 2)

sns.set(style="darkgrid")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
fig.suptitle("SkinType Evaluation: 30-repeats, 5-Fold")
sns.histplot(data=df, x='true_scoers', kde=True, ax=ax[0])
ax[0].set_title("mean: "+str(means[0]) + " ,std: "+str(stds[0]))
sns.histplot(data=df, x='neighbors_scoers', kde=True, ax=ax[1])
ax[1].set_title("mean: "+str(means[1]) + " ,std: "+str(stds[1]))
plt.show()

