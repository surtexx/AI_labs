import numpy as np
import matplotlib.pyplot as plt

train_labels = np.load('data/training_labels.npy', allow_pickle=True)
train_sentences = np.load('data/training_sentences.npy', allow_pickle=True)
test_labels = np.load('data/test_labels.npy', allow_pickle=True)
test_sentences = np.load('data/test_sentences.npy', allow_pickle=True)

# 2.


def normalize_data(train_data, test_data, type=None):
    if type is None:
        return train_data, test_data
    elif type == 'standard':
        medie = np.mean(train_data, axis=0)
        deviatie = np.std(train_data, axis=0)
        return (train_data - medie) / deviatie, (test_data - medie) / deviatie
    elif type == 'l1':
        suma_train = np.sum(train_data, axis=0)
        suma_test = np.sum(test_data, axis=0)
        return train_data / suma_train, test_data / suma_test
    elif type == 'l2':
        norm_train = np.linalg.norm(train_data, axis=0)
        norm_test = np.linalg.norm(test_data, axis=0)
        return train_data / norm_train, test_data / norm_test
    else:
        raise Exception('Invalid normalization type')


# 3. si 4.


class BagOfWords:
    def __init__(self):
        self.vocabulary = {}
        self.order = []

    def build_vocabulary(self, data):
        index = 0
        for sentence in data:
            for word in sentence:
                if word not in self.vocabulary:
                    self.vocabulary[word] = index
                    self.order.append(word)
                    index += 1

    def get_features(self, data):
        features = np.zeros((data.shape[0], len(self.vocabulary)))
        for i in range(data.shape[0]):
            for word in data[i]:
                if word in self.vocabulary:
                    features[i, self.vocabulary[word]] += 1
        return features


train_s = BagOfWords()
test_s = BagOfWords()
train_s.build_vocabulary(train_sentences)
test_s.build_vocabulary(test_sentences)

# 5.
normalized_data = normalize_data(train_s.get_features(train_sentences), test_s.get_features(test_sentences), 'l2')
