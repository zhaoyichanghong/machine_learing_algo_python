import numpy as np
from functools import reduce

class Tfidf:
    @property
    def word_dictionary(self):
        return self.__word_dictionary

    def __create_idf(self, corpus):
        word_number = np.zeros_like(self.__word_dictionary, dtype=int)
        for row in corpus:
            _, indexes, _ = np.intersect1d(self.word_dictionary, row, return_indices=True)
            word_number[indexes] += 1

        return np.log(len(corpus) / (word_number + 1))

    def fit_transform(self, X):
        '''
        Parameters
        ----------
        X : shape (corpus_number, text_length)
            Training corpus

        Returns
        -------
        Tf-idf matrix : shape (corpus_number, dictionary_length)
                        Tf-idf-weighted document-term matrix
        '''
        self.__word_dictionary = reduce(np.union1d, X)
        self.__idf = self.__create_idf(X)
        
        return self.transform(X)

    def transform(self, X):
        '''
        Parameters
        ----------
        X : shape (corpus_number, text_length)
            Predicting corpus

        Returns
        -------
        Tf-idf matrix : shape (corpus_number, dictionary_length)
                        Tf-idf-weighted document-term matrix
        '''  
        tf = np.zeros((len(X), len(self.__word_dictionary)))
        for i in range(len(X)):
            for j in range(len(self.__word_dictionary)):
                tf[i, j] = X[i].count(self.__word_dictionary[j])

        return tf * self.__idf