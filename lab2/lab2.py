import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

multime = [(160, 'F'), (165, 'F'), (155, 'F'), (172, 'F'), (175, 'B'), (180, 'B'), (177, 'B'), (190, 'B')]

# 1.
p_fata = len([x for x in multime if x[1] == 'F'])/len(multime)
p_interval = len([x for x in multime if 170 < x[0] < 181])/len(multime)
p_cond = len([x for x in multime if 170 < x[0] < 181 and x[1] == 'F']) / len([x for x in multime if x[1] == 'F'])
print(p_cond * p_fata / p_interval)
p_baiat = len([x for x in multime if x[1] == 'F'])/len(multime)
p_cond1 = len([x for x in multime if 170 < x[0] < 181 and x[1] == 'B']) / len([x for x in multime if x[1] == 'B'])
print(p_cond1 * p_baiat / p_interval)

# 2.
bins = np.linspace(start=0, stop=256, num=6)
train_images = np.loadtxt('./data/train_images.txt')
train_images = np.digitize(train_images, bins) - 1
train_labels = np.loadtxt('./data/train_labels.txt', 'int')

# 3.
bins = np.linspace(start=0, stop=256, num=6)
test_images = np.digitize(np.loadtxt('./data/test_images.txt'), bins) - 1
test_labels = np.loadtxt('./data/test_labels.txt', 'int')
train_images = np.loadtxt('./data/train_images.txt')
train_images = np.digitize(train_images, bins) - 1
train_labels = np.loadtxt('./data/train_labels.txt', 'int')
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(train_images, train_labels)
naive_bayes_model.predict(test_images)
print(naive_bayes_model.score(test_images, test_labels))

# 4.
for num_bins in [3, 5, 7, 9, 11]:
    bins = np.linspace(start=0, stop=256, num=num_bins + 1)
    train_images = np.loadtxt('./data/train_images.txt')
    train_images = np.digitize(train_images, bins) - 1
    train_labels = np.loadtxt('./data/train_labels.txt', 'int')
    test_images = np.digitize(np.loadtxt('./data/test_images.txt'), bins) - 1
    test_labels = np.loadtxt('./data/test_labels.txt', 'int')
    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(train_images, train_labels)
    predict = naive_bayes_model.predict(test_images)
    # print(naive_bayes_model.score(test_images, test_labels))

# 5.
bins = np.linspace(start=0, stop=256, num=5)
train_images = np.loadtxt('./data/train_images.txt')
train_images = np.digitize(train_images, bins) - 1
train_labels = np.loadtxt('./data/train_labels.txt', 'int')
test_images = np.digitize(np.loadtxt('./data/test_images.txt'), bins) - 1
test_labels = np.loadtxt('./data/test_labels.txt', 'int')
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(train_images, train_labels)
predict = naive_bayes_model.predict(test_images)
wrong_images = test_images[predict != test_labels]
wrong_pred = predict[predict != test_labels]
for i in range(10):
    img = wrong_images[i, :]
    img = np.reshape(img, (28,28))
    plt.imshow(img.astype(np.uint8), cmap='gray')
    plt.show()

# 6.
bins = np.linspace(start=0, stop=256, num=5)
train_images = np.loadtxt('./data/train_images.txt')
train_images = np.digitize(train_images, bins) - 1
train_labels = np.loadtxt('./data/train_labels.txt', 'int')
test_images = np.digitize(np.loadtxt('./data/test_images.txt'), bins) - 1
test_labels = np.loadtxt('./data/test_labels.txt', 'int')
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(train_images, train_labels)
predict = naive_bayes_model.predict(test_images)


def confusion_matrix(y_true, y_pred):
    matrix = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            matrix[i, j] = np.sum((y_true == test_labels[i]) & (y_pred == test_labels[j]))
    return matrix


print(confusion_matrix(predict, test_labels))
