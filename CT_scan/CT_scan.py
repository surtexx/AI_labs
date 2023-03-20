import cv2
import imutils
import numpy as np
# from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
import csv
from sklearn.metrics import f1_score

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        leftmost = tuple(c[c[:, :, 0].argmin()][0])
        rightmost = tuple(c[c[:, :, 0].argmax()][0])
        topmost = tuple(c[c[:, :, 1].argmin()][0])
        bottommost = tuple(c[c[:, :, 1].argmax()][0])

        new_image = image[topmost[1]:bottommost[1], leftmost[0]:rightmost[0]]
        return new_image
    else:
        return image


train_images_names = [f"0{i}{j}{k}{l}{m}" for i in range(2) for j in range(10) for k in range(10) for l in range(10) for m in range(10)][1:15001]
validation_images_names = [f"0{x}" for x in range(15001, 17001)]
test_images_names = [f"0{x}" for x in range(17001, 22150)]
labels = np.array([int(x) for x in "\n".join(open("train_labels.txt").read().split(",")[1:]).split("\n")[2::2]])

train_images = np.array([cv2.resize(preprocess_image(cv2.imread(f"./data/data/{img}.png")), dsize=(64, 64), interpolation=cv2.INTER_CUBIC)/255 for img in train_images_names], dtype=object)
train_images = train_images.reshape(train_images.shape[0], -1)
train_labels = labels[:15000]

validation_images = np.array([cv2.resize(preprocess_image(cv2.imread(f"./data/data/{img}.png")), dsize=(64, 64), interpolation=cv2.INTER_CUBIC)/255 for img in validation_images_names], dtype=object)
validation_images = validation_images.reshape(validation_images.shape[0], -1)
validation_labels = labels[15000:]

# test_images = np.array([cv2.resize(preprocess_image(cv2.imread(f"./data/data/{img}.png")), dsize=(64,64), interpolation=cv2.INTER_CUBIC)/255 for img in test_images_names])
# test_images = test_images.reshape(test_images.shape[0], -1)

classifier = OneVsRestClassifier(SVC(kernel='linear', C=1, gamma=0.1, probability=True))
classifier.fit(train_images, train_labels)
validation_prediction = classifier.predict(validation_images)
validation_score = f1_score(validation_labels, validation_prediction, average='macro')
print(validation_score)

# test_prediction = classifier.predict(test_images)
# output = [{'id': img, 'class': prediction} for (img, prediction) in zip(test_images_names, test_prediction)]
# with open("submission.csv", "w", newline='') as out:
#     writer = csv.DictWriter(out, fieldnames=['id', 'class'])
#     writer.writeheader()
#     writer.writerows(output)
