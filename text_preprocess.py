import numpy as np
from functools import reduce

class Tfidf:
    @property
    def word_dictionary(self):
        return self.__word_dictionary

    def __create_dictionary(self, corpus):
        return reduce(np.union1d, corpus)

    def __create_idf(self, corpus):
        data_number = corpus.shape[0]
        word_dictionary_len = len(self.__word_dictionary)

        word_number = np.zeros(word_dictionary_len)
        for i in range(word_dictionary_len):
            for j in range(data_number):
                if self.__word_dictionary[i] in corpus[j]:
                    word_number[i] += 1
        return np.log(data_number / (word_number + 1))

    def fit_transform(self, X):
        self.__word_dictionary = self.__create_dictionary(X)
        self.__idf = self.__create_idf(X)
        
        return self.transform(X)

    def transform(self, X):
        data_number = X.shape[0]
        word_dictionary_len = len(self.__word_dictionary)

        tf = np.zeros((data_number, word_dictionary_len))
        for i in range(data_number):
            for word in X[i]:
                tf[i, np.flatnonzero(self.__word_dictionary == word)] += 1 / len(X[i])

        return tf * self.__idf