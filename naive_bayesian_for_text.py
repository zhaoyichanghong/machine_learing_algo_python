import numpy as np
import text_preprocess

class NaiveBayesianForText:
    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : shape (corpus_number, text_length)
            Training corpus
        y : shape (corpus_number,)
            Target values
        '''
        self.__classes = np.unique(y)
        classes_number = len(self.__classes)
        self.__p_classes = [np.mean(y == label) for label in self.__classes]

        self.__model = text_preprocess.Tfidf()
        word_vector = self.__model.fit_transform(X)

        word_of_classes = np.ones((classes_number, len(self.__model.word_dictionary)))
        word_of_classes_total = np.full(classes_number, classes_number)
        for i in range(classes_number):
            word_of_classes[i] += np.sum(word_vector[np.flatnonzero(y == self.__classes[i])], axis=0)
            word_of_classes_total[i] += np.sum(word_of_classes[i])

        self.__p_word_of_classes = word_of_classes / word_of_classes_total.reshape((-1, 1))

    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (corpus_number, text_length)
            Predicting corpus

        Returns
        -------
        y : shape (corpus_number,)
            Predicted class label per sample
        '''
        data_number = len(X)

        word_vector = np.zeros((data_number, len(self.__model.word_dictionary)))
        for i in range(data_number):
            _, indexes, _ = np.intersect1d(self.__model.word_dictionary, X[i], return_indices=True)
            word_vector[i, indexes] = 1

        p_class_of_doc = word_vector.dot(np.log(self.__p_word_of_classes).T) + np.log(self.__p_classes)

        return self.__classes[np.argmax(p_class_of_doc, axis=1)]