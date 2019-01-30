import numpy as np
from functools import reduce

class NaiveBayesianForText:
    def fit(self, X, y):
        data_number = X.shape[0]
        word_set = reduce(np.union1d, X)
        word_set_len = len(word_set)

        word_vec = np.zeros((data_number, word_set_len))
        for i in range(data_number):
            _, indexes, _ = np.intersect1d(word_set, X[i], return_indices=True)
            word_vec[i, indexes] = 1
        
        self.__classes = np.unique(y)
        classes_number = len(self.__classes)
        self.__p_classes = np.zeros(classes_number)
        for i in range(classes_number):
            self.__p_classes[i] = np.mean(y == self.__classes[i])
        
        word_of_classes = np.ones((classes_number, word_set_len))
        word_of_classes_total = np.full(classes_number, classes_number)
        for i in range(data_number):
            for j in range(classes_number):
                if y[i] == self.__classes[j]:
                    word_of_classes[j] += word_vec[i]
                    word_of_classes_total[j] += np.sum(word_vec[i])

        self.__p_word_of_classes = word_of_classes / word_of_classes_total.reshape((-1, 1))

    def predict(self, X):
        data_number = X.shape[0]
        vocab_set = reduce(np.union1d, X)
        vocab_set_len = len(vocab_set)

        word_vec = np.zeros((data_number, vocab_set_len))
        for i in range(data_number):
            _, indexes, _ = np.intersect1d(vocab_set, X[i], return_indices=True)
            word_vec[i, indexes] = 1

        p_class_of_doc = word_vec.dot(np.log(self.__p_word_of_classes).T) + np.log(self.__p_classes)

        return self.__classes[np.argmax(p_class_of_doc, axis=1)]