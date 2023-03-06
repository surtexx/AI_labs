import numpy as np
import matplotlib.pyplot as plt

# 1.


class KnnClassifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors=3, metric='l2'):
        distances = []
        for i in range(len(self.train_images)):
            if metric == 'l1':
                dist = np.sum(np.abs(self.train_images[i] - test_image))
            elif metric == 'l2':
                dist = np.sqrt(np.sum(np.square(self.train_images[i] - test_image)))
            distances.append((dist, self.train_labels[i]))
        distances.sort(key=lambda x: x[0])
        neighbors = [distances[i][1] for i in range(num_neighbors)]
        counts = np.bincount(neighbors)
        return np.argmax(counts)

    def classify_images(self, test_images, num_neighbours=3, metric='l2'):
        return np.array([self.classify_image(image) for image in test_images])

    def accuracy(self, test_images, test_labels):
        classified_images = np.array([self.classify_image(image) for image in test_images])
        np.savetxt('predictii_3nn_l2_mnist.txt', classified_images)
        return np.mean(test_labels == classified_images) * 100

    def accuracy_nn(self, test_images, test_labels, num_neighbours=3, metric='l2'):
        classified_images = np.array([self.classify_image(image, num_neighbours, metric) for image in test_images])
        return np.mean(test_labels == classified_images) * 100


train_images = np.loadtxt('./data/train_images.txt')
test_images = np.loadtxt('./data/test_images.txt')
train_labels = np.loadtxt('./data/train_labels.txt')
test_labels = np.loadtxt('./data/test_labels.txt')

# 2.

classifier = KnnClassifier(train_images, train_labels)
img = np.reshape(test_images[0], (28, 28))
plt.imshow(img.astype(np.uint8), cmap='gray')
plt.show()
print(classifier.classify_image(test_images[0]))

# 3.

print(classifier.accuracy(test_images, test_labels))

# 4. a)
nums_neighbours = [1, 3, 5, 7, 9]
accuracies = [classifier.accuracy_nn(test_images, test_labels, i, 'l2') for i in nums_neighbours]
np.savetxt('acuratete_l2.txt', np.array(accuracies))
plt.plot(nums_neighbours, accuracies)
plt.ylabel('accuracy')
plt.xlabel('number of neighbours')
plt.show()

# b)
nums_neighbours = [1, 3, 5, 7, 9]
accuracies_l2 = [classifier.accuracy_nn(test_images, test_labels, i, 'l2') for i in nums_neighbours]
plt.plot(nums_neighbours, accuracies_l2)
plt.ylabel('accuracy')
plt.xlabel('number of neighbours')
accuracies_l1 = [classifier.accuracy_nn(test_images, test_labels, i, 'l1') for i in nums_neighbours]
np.savetxt('acuratete_l1.txt', np.array(accuracies_l1))
plt.plot(nums_neighbours, accuracies_l1)
plt.legend(['l2', 'l1'])
plt.show()

# c)
nums_neighbours = [1, 3, 5, 7, 9]
plt.ylabel('accuracy')
plt.xlabel('number of neighbours')
accuracies_l1 = [classifier.accuracy_nn(test_images, test_labels, i, 'l1') for i in nums_neighbours]
plt.plot(nums_neighbours, accuracies_l1)
plt.show()
