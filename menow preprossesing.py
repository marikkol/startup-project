
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import os
import random
from PIL import Image
from mtcnn.mtcnn import MTCNN


""" no video sepperation version"""


# create image label DataFrame
def image_label_df(json_path, imgs_path, imgs_per_person=5):

    # json to dataframe
    df = pd.read_json(json_path, orient='records')
    df = df.T

    # create df containing the participants index to the images in a string column
    participants = os.listdir(imgs_path)
    col = []
    for i in range(len(participants)):
        all_imgs = os.listdir(imgs_path + "\\" + participants[i])
        rand_imgs = random.sample(all_imgs, imgs_per_person)
        col.append(rand_imgs)
    d = {'participants': participants, 'imgs': col}
    imgs_df = pd.DataFrame(data=d)
    imgs_df.participants = imgs_df.participants.apply(lambda x: int(x))
    imgs_df = imgs_df.set_index('participants')

    # new df of filename col and skin-type(label) col
    img_label_df = pd.concat([df, imgs_df], axis=1, join="inner")
    img_label_df = img_label_df.explode('imgs')
    #img_label_df.label = img_label_df.label.apply(lambda x: x['skin-type'])     להכליל את הלייבלים, לשימוש של שלושתם
    img_label_df.imgs = np.array([str(x) + "\\" for x in img_label_df.index]) + img_label_df.imgs
    img_label_df = img_label_df[['imgs', 'label']]

    return img_label_df



json_path = "C:\\Users\\USER1\\Desktop\\MeNow Project\\CC_annotations\\CasualConversations.json"
imgs_path = "C:\\Users\\USER1\\Desktop\\MeNow Project\\part1"

img_label_df = image_label_df(json_path, imgs_path, 10)
print(img_label_df)


## Detect Faces


# extract a single face from a given photograph
def extract_face(filename):
    required_size = (160, 160)
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    print(filename)
    if len(results)==0:
        print("no face found")
        return "no face found"
    # extract the bounding box from the first face
    print(results)
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


# load images and extract faces for all images in a directory
def load_faces(img_label_df, plot_example):
    faces = list()
    # enumerate files
    for filename in img_label_df.imgs.values:
        # path
        img_path = imgs_path + "\\" + filename
        # get face
        face = extract_face(img_path)
        # store
        faces.append(face)
    if plot_example==True:
        # display 14 faces from the extracted faces
        i = 1
        # enumerate files
        for filename in img_label_df.imgs.values[:14]:
            # path
            img_path = imgs_path + "\\" + filename
            # get face
            face_arr = extract_face(img_path)
            if face_arr == "no face found":
                face_arr = np.zeros((160, 160, 3))
            print(i, face_arr.shape)
            # embedding = get_embedding(facenet_model, face)
            # print(embedding.shape)
            # plot
            plt.subplot(2, 7, i)
            plt.axis('off')
            plt.imshow(face_arr)
            i += 1
        plt.show()
    return faces


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(img_label_df, plot_example=False):
    X = load_faces(img_label_df, plot_example)
    X = np.asarray(X)
    print(f"X shape: {X.shape}")
    y = img_label_df.label.values
    # drop files were no face have found
    idxs = np.where(X == "no face found")
    idxs = idxs[0]
    print(f"num of pic were no face found: {len(idxs)}")
    X = np.delete(X, idxs, axis=(len(X.shape)-1))
    y = np.delete(y, idxs)
    return X, y

# train test split
def train_test_split(df,fruc):
    participants = np.unique(df.index)
    num_parti = len(participants)
    print(f"the num of participants: {num_parti}")
    train_df = df[df.index < (num_parti*fruc)]
    test_df = df[df.index > (num_parti*fruc)]
    return train_df, test_df

train_df, test_df = train_test_split(img_label_df, 0.7)
print(f"train size: {len(train_df)}, test size: {len(test_df)}")

# load train dataset
trainX, trainy = load_dataset(train_df, plot_example=False)
print(trainX[0].shape, trainy.shape)
print(trainX)
# load test dataset
testX, testy = load_dataset(test_df)
print(len(testX))
print(f"final train size: {len(trainy)}, final test size: {len(testy)}")
# save arrays to one file in compressed format
np.savez_compressed('faces-dataset.npz', trainX, trainy, testX, testy)



## Create Face Embeddings


# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

# load the face dataset
data = np.load('faces-dataset.npz', allow_pickle=True)
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
# load the facenet model
facenet_model = load_model('facenet_keras.h5')
# summarize input and output shape
print(facenet_model.inputs)
print(facenet_model.outputs)
# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(facenet_model, face_pixels)
    newTrainX.append(embedding)
newTrainX = np.asarray(newTrainX)
print(newTrainX.shape)
# convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX:
    embedding = get_embedding(facenet_model, face_pixels)
    newTestX.append(embedding)
newTestX = np.asarray(newTestX)
print(newTestX.shape)
# save arrays to one file in compressed format
np.savez_compressed('faces-embeddings.npz', newTrainX, trainy, newTestX, testy)

