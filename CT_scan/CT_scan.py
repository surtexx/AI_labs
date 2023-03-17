from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


train_images_names = [f"0{i}{j}{k}{l}{m}.png" for i in range(2) for j in range(10) for k in range(10) for l in range(10) for m in range(10)][1:15001]
validation_images_names = [f"0{x}.png" for x in range(15001, 17001)]
test_images_names = [f"0{x}.png" for x in range(17001, 22150)]
labels = np.array([int(x) for x in "\n".join(open("train_labels.txt").read().split(",")[1:]).split("\n")[2::2]])

train_images = np.array([np.asarray(Image.open(f"./data/data/{img}").convert('L')) for img in train_images_names])
train_images = train_images.reshape(train_images.shape[0], -1)
train_labels = labels[:15000]

validation_images = np.array([np.asarray(Image.open(f"./data/data/{img}").convert('L')) for img in validation_images_names])
validation_images = validation_images.reshape(validation_images.shape[0], -1)

test_images = np.array([np.asarray(Image.open(f"./data/data/{img}").convert('L')) for img in test_images_names])
test_images = test_images.reshape(test_images.shape[0], -1)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(train_images, train_labels)
predictie = classifier.predict(validation_images)
