import os
import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Input, Conv2D, MaxPool2D, Flatten


from keras.utils import to_categorical

def detect_face(img):
    img = img[70:195,78:172]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (50, 50))
    return img

def print_progress(val, val_len, folder, bar_size=20):
    progr = "#"*round((val)*bar_size/val_len) + " "*round((val_len - (val))*bar_size/val_len)
    if val == 0:
        print("", end = "\n")
    else:
        print("[%s] (%d samples)\t label : %s \t\t" % (progr, val+1, folder), end="\r")

dataset_folder = "dataset/"

names = []
images = []
for folder in os.listdir(dataset_folder):
    files = os.listdir(os.path.join(dataset_folder, folder))[:150]
    if len(files) < 50 :
        continue
    for i, name in enumerate(files):
        if name.find(".jpg") > -1 :
            img = cv2.imread(os.path.join(dataset_folder + folder, name))
            img = detect_face(img) # detect face using mtcnn and crop to 100x100
            if img is not None :
                images.append(img)
                names.append(folder)

                print_progress(i, len(files), folder)

print("banyak sample :", len(names))


def img_augmentation(img):
    h, w = img.shape
    center = (w // 2, h // 2)
    M_rot_5 = cv2.getRotationMatrix2D(center, 5, 1.0)
    M_rot_neg_5 = cv2.getRotationMatrix2D(center, -5, 1.0)
    M_rot_10 = cv2.getRotationMatrix2D(center, 10, 1.0)
    M_rot_neg_10 = cv2.getRotationMatrix2D(center, -10, 1.0)
    M_trans_3 = np.float32([[1, 0, 3], [0, 1, 0]])
    M_trans_neg_3 = np.float32([[1, 0, -3], [0, 1, 0]])
    M_trans_6 = np.float32([[1, 0, 6], [0, 1, 0]])
    M_trans_neg_6 = np.float32([[1, 0, -6], [0, 1, 0]])
    M_trans_y3 = np.float32([[1, 0, 0], [0, 1, 3]])
    M_trans_neg_y3 = np.float32([[1, 0, 0], [0, 1, -3]])
    M_trans_y6 = np.float32([[1, 0, 0], [0, 1, 6]])
    M_trans_neg_y6 = np.float32([[1, 0, 0], [0, 1, -6]])

    imgs = []
    imgs.append(cv2.warpAffine(img, M_rot_5, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_rot_neg_5, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_rot_10, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_rot_neg_10, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_3, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_3, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_6, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_6, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_y3, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_y3, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_y6, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_y6, (w, h), borderValue=(255, 255, 255)))
    imgs.append(cv2.add(img, 10))
    imgs.append(cv2.add(img, 30))
    imgs.append(cv2.add(img, -10))
    imgs.append(cv2.add(img, -30))
    imgs.append(cv2.add(img, 15))
    imgs.append(cv2.add(img, 45))
    imgs.append(cv2.add(img, -15))
    imgs.append(cv2.add(img, -45))

    return imgs

# plt.imshow(images[0], cmap="gray")
# tes augmentasi
# img_test = images[0]
#
# augmented_image_test = img_augmentation(img_test)
#
# plt.figure(figsize=(15,10))
# for i, img in enumerate(augmented_image_test):
#     plt.subplot(4,5,i+1)
#     plt.imshow(img, cmap="gray")
# plt.show()

# buat augmentasi ke seluruh image
augmented_images = []
augmented_names = []
for i, img in enumerate(images):
    try :
        augmented_images.extend(img_augmentation(img))
        augmented_names.extend([names[i]] * 20)
    except :
        print(i)
# total image setelah di augmentasi
print(len(augmented_images), len(augmented_names))

unique, counts = np.unique(names, return_counts = True)

for item in zip(unique, counts):
    print(item)

# mengimbangkan data
def print_data(label_distr, label_name):
    plt.figure(figsize=(12, 6))

    my_circle = plt.Circle((0, 0), 0.7, color='white')
    plt.pie(label_distr, labels=label_name, autopct='%1.1f%%')
    plt.gcf().gca().add_artist(my_circle)
    plt.show()

unique = np.unique(names)
label_distr = {i: names.count(i) for i in names}.values()
#print_data(label_distr, unique)

# reduce sample size per-class using numpy random choice
n = 1000

def randc(labels, l):
    return np.random.choice(np.where(np.array(labels) == l)[0], n, replace=True)

mask = np.hstack([randc(names, l) for l in np.unique(names)])

names = [names[m] for m in mask]
images = [images[m] for m in mask]

label_distr = {i:names.count(i) for i in names}.values()
#print_data(label_distr, unique)

#print(len(names))

# Encoding Label & Categoricalization
le = LabelEncoder()

le.fit(names)

labels = le.classes_

name_vec = le.transform(names)

categorical_name_vec = to_categorical(name_vec)

print("number of class :", len(labels))
# print(labels)
# print(name_vec)
#
# print(categorical_name_vec)

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(np.array(images, dtype=np.float32),   # input data
                                                    np.array(categorical_name_vec),       # target/output data
                                                    test_size=0.15,
                                                    random_state=42)

# print(x_train.shape, y_train.shape, x_test.shape,  y_test.shape)

# Reshape Data
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

print((x_train.shape, x_test.shape))

# # Buat CNN Model
def cnn_model(input_shape):
    model = Sequential()

    model.add(Conv2D(64,
                     (3, 3),
                     padding="valid",
                     activation="relu",
                     input_shape=input_shape))
    model.add(Conv2D(64,
                     (3, 3),
                     padding="valid",
                     activation="relu",
                     input_shape=input_shape))

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(128,
                     (3, 3),
                     padding="valid",
                     activation="relu"))
    model.add(Conv2D(128,
                     (3, 3),
                     padding="valid",
                     activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(len(labels)))  # equal to number of classes
    model.add(Activation("softmax"))

    model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
#
# # Training CNN Model
input_shape = x_train[0].shape

EPOCHS = 10
BATCH_SIZE = 32

model = cnn_model(input_shape)

history = model.fit(x_train,
                    y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_split=0.15   # 15% of train dataset will be used as validation set
                    )


# # Evaluasi Model
# def evaluate_model_(history):
#     names = [['accuracy', 'val_accuracy'],
#              ['loss', 'val_loss']]
#     for name in names:
#         fig1, ax_acc = plt.subplots()
#         plt.plot(history.history[name[0]])
#         plt.plot(history.history[name[1]])
#         plt.xlabel('Epoch')
#         plt.ylabel(name[0])
#         plt.title('Model - ' + name[0])
#         plt.legend(['Training', 'Validation'], loc='lower right')
#         plt.grid()
#         plt.show()
# #
# #
# evaluate_model_(history)

model.save("model-cnn-facerecognition.h5")

# predict test data
y_pred=model.predict(x_test)

# plot confusion matriks
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#     plt.figure(figsize=(8, 8))
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()
#
#
# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
# np.set_printoptions(precision=2)
#
#
# # Plot non-normalized confusion matrix
# plot_confusion_matrix(cnf_matrix, classes=labels,normalize=False,
#                       title='Confusion matrix')
#
#
# print(classification_report(y_test.argmax(axis=1),
#                             y_pred.argmax(axis=1),
#                             target_names=labels))



def draw_ped(img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):

    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img,
                  (x0, y0 + baseline),
                  (max(xt, x0 + w), yt),
                  color,
                  2)
    cv2.rectangle(img,
                  (x0, y0 - h),
                  (x0 + w, y0 + baseline),
                  color,
                  -1)
    cv2.putText(img,
                label,
                (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
                cv2.LINE_AA)
    return img
#
#
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# --------- load Keras CNN model -------------
model = load_model("model-cnn-facerecognition.h5")
print("[INFO] finish load model...")

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:

            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (50, 50))
            face_img = face_img.reshape(1, 50, 50, 1)

            result = model.predict(face_img)
            idx = result.argmax(axis=1)
            confidence = result.max(axis=1) * 100
            if confidence > 80:
                label_text = "%s (%.2f %%)" % (labels[idx], confidence)
            else:
                label_text = "N/A"
            frame = draw_ped(frame, label_text, x, y, x + w, y + h, color=(0, 255, 255), text_color=(50, 50, 50))

        cv2.imshow('Detect Face', frame)
    else:
        break
    if cv2.waitKey(10) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
