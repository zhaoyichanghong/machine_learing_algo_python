import numpy as np
import text_preprocess

class NaiveBayesianForText:
    def fit(self, X, y):
        data_number = X.shape[0]

        self.__classes = np.unique(y)
        classes_number = len(self.__classes)
        self.__p_classes = np.zeros(classes_number)
        for i in range(classes_number):
            self.__p_classes[i] = np.mean(y == self.__classes[i])

        self.__model = text_preprocess.Tfidf()
        word_vector = self.__model.fit_transform(X)
        word_vector_len = word_vector.shape[1]

        word_of_classes = np.ones((classes_number, word_vector_len))
        word_of_classes_total = np.full(classes_number, classes_number)
        for i in range(data_number):
            for j in range(classes_number):
                if y[i] == self.__classes[j]:
                    word_of_classes[j] += word_vector[i]
                    word_of_classes_total[j] += np.sum(word_vector[i])

        self.__p_word_of_classes = word_of_classes / word_of_classes_total.reshape((-1, 1))

    def predict(self, X):
        data_number = X.shape[0]
        word_dictionary_len = len(self.__model.word_dictionary)

        word_vec = np.zeros((data_number, word_dictionary_len))
        for i in range(data_number):
            _, indexes, _ = np.intersect1d(self.__model.word_dictionary, X[i], return_indices=True)
            word_vec[i, indexes] = 1

        p_class_of_doc = word_vec.dot(np.log(self.__p_word_of_classes).T) + np.log(self.__p_classes)

        return self.__classes[np.argmax(p_class_of_doc, axis=1)]