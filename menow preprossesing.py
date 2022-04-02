import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import os
from PIL import Image
from mtcnn.mtcnn import MTCNN


""" video sepperation version"""

# הבעיות עם השייפ - בגלל שזה אריי של ליסטים - לתקן בפונקציות


#pip install git+https://github.com/keras-team/keras-preprocessing.git

# create image label DataFrame
def image_label_df(json_path, imgs_path):

    # json to dataframe
    df = pd.read_json(json_path, orient='records')
    df = df.T

    # create df containing the participants index to the images in a string column
    participants = os.listdir(imgs_path)
    col = []
    for i in range(len(participants)):
        imgs = os.listdir(imgs_path + "\\" + participants[i])
        col.append(imgs)
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
imgs_path = "C:\\Users\\USER1\\Desktop\\MeNow Project\\Sampled Frames Dataset"

img_label_df = image_label_df(json_path, imgs_path)



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
def load_faces(img_label_df, plot_example, num_face_per_video=5):
    faces = list()    # (videos_list, img_list(10), 160 ,160 , 3)
    participants = np.unique(img_label_df.index)
    idx_les_5 = list()    # the indexs to drop from y - drop videos that not found 5 faces in them.
    idx = 0
    for par in participants:
        par_df = img_label_df[img_label_df.index == par]
        vid1_df = par_df[:100]
        vid2_df = par_df[100:]
        for vid_df in [vid1_df,vid2_df]:
            vid_faces = list()
            filename_list = list()
            while len(vid_faces) < num_face_per_video:
                filename = np.random.choice(vid_df.imgs.values, 1)[0]
                if filename not in filename_list:
                    img_path = imgs_path + "\\" + filename
                    # get face
                    face = extract_face(img_path)
                    if face == "no face found":
                        filename_list.append(filename)
                        if len(filename_list) > (len(vid_faces) + 3):
                            break        # give up this video
                        else:
                            continue     # keep trying find faces

                    filename_list.append(filename)
                    vid_faces.append(face)
            if len(vid_faces) == 5:
                faces.append(vid_faces)
            else:
                idx_les_5.append(idx)
            idx += 1

    if plot_example==True:
        # display 10 faces from the extracted faces
        i = 1
        for face in faces[0]:
            print(i, face.shape)
            # plot
            plt.subplot(2, 5, i)
            plt.axis('off')
            plt.imshow(face)
            i += 1
        plt.show()

    return faces, idx_les_5


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(img_label_df, plot_example=False):
    X, idx_les_5 = load_faces(img_label_df, plot_example)
    print(f"the indexs to drop from y: {idx_les_5}")
    X = np.asarray(X)
    y = img_label_df.label.values
    num_y_vid = X.shape[0] + len(idx_les_5)
    y = np.reshape(y, (num_y_vid, -1))     # (videos_arr, labels(100))
    if len(idx_les_5) > 0:
        y = np.delete(y, np.asarray(idx_les_5), 0)
    # Check all values in an array are equal to its first element
    result = np.all(y[10] == y[10][0])
    if result:
        print('All Values in Array are same')
    else:
        print('All Values in Array are not same')
    return X, y[:,:X.shape[1]]


# train test split
def train_test_split(df,fruc):
    participants = np.unique(df.index)
    num_parti = len(participants)
    print(f"the num of participants: {num_parti}")
    train_df = df[df.index < (num_parti*fruc)]
    test_df = df[df.index > (num_parti*fruc)]
    return train_df, test_df

train_df, test_df = train_test_split(img_label_df, 0.75)
print(f"train size: {len(train_df)}, test size: {len(test_df)}")


# load train dataset
trainX, trainy = load_dataset(train_df, plot_example=True)
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset(test_df)
print(f"final train size: {len(trainy)} videos, final test size: {len(testy)} videos")
# save arrays to one file in compressed format
np.savez_compressed('faces-dataset191.npz', trainX, trainy, testX, testy)

"""
trainX, testX - is a 5 dimetional array (videos_arr, img_arr(5), 160 ,160 , 3)
trainy, testy - is a 5 dimetional array (videos_arr, img_arr(5), 160 ,160 , 3)
"""


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
data = np.load('faces-dataset191.npz', allow_pickle=True)
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)


# load the facenet model
facenet_model = load_model('facenet_keras.h5')
# summarize input and output shape
print(facenet_model.inputs)
print(facenet_model.outputs)


# convert each face in the train set to an embedding
def faces_to_embeddings(X_arr):
    new_X_arr = list()
    for vid_imgs in X_arr:
        vid_embedding = list()
        for face_pixels in vid_imgs:
            embedding = get_embedding(facenet_model, face_pixels)
            vid_embedding.append(embedding)
        vid_embedding = np.asarray(vid_embedding)
        new_X_arr.append(vid_embedding)
    new_X_arr = np.asarray(new_X_arr)
    return new_X_arr

newTrainX = faces_to_embeddings(trainX)
print(newTrainX.shape)
newTestX = faces_to_embeddings(testX)
print(newTestX.shape)

# save arrays to one file in compressed format
np.savez_compressed('faces-embeddings191.npz', newTrainX, trainy, newTestX, testy)
