import numpy as np
from functools import reduce

class Tfidf:
    @property
    def word_dictionary(self):
        return self.__word_dictionary

    def __create_idf(self, corpus):
        n_words = np.zeros_like(self.__word_dictionary, dtype=int)
        for row in corpus:
            _, indexes, _ = np.intersect1d(self.word_dictionary, row, return_indices=True)
            n_words[indexes] += 1

        return np.log(len(corpus) / (n_words + 1))

    def fit_transform(self, X):
        '''
        Parameters
        ----------
        X : shape (n_corpus, text_length)
            Training corpus

        Returns
        -------
        Tf-idf matrix : shape (n_corpus, dictionary_length)
                        Tf-idf-weighted document-term matrix
        '''
        self.__word_dictionary = reduce(np.union1d, X)
        self.__idf = self.__create_idf(X)
        
        return self.transform(X)

    def transform(self, X):
        '''
        Parameters
        ----------
        X : shape (n_corpus, text_length)
            Predicting corpus

        Returns
        -------
        Tf-idf matrix : shape (n_corpus, dictionary_length)
                        Tf-idf-weighted document-term matrix
        '''  
        tf = np.zeros((len(X), len(self.__word_dictionary)))
        for i in range(len(X)):
            for j in range(len(self.__word_dictionary)):
                tf[i, j] = X[i].count(self.__word_dictionary[j])

        return tf * self.__idf