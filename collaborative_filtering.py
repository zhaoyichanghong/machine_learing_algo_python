import numpy as np
import matplotlib.pyplot as plt

class CollaborativeFiltering:
    def fit(self, X, y, dimension, learning_rate, epochs):
        '''
        Parameters
        ----------
        X : shape (data_number, 2)
            Training data, column 1 is user id, column 2 is item id
        y : shape (data_number, 1)
            Rating
        learning_rate : learning rate
        epochs : The number of epochs
        '''
        data_number = X.shape[0]
        user_id = X[:, 0]
        item_id = X[:, 1]
        
        self.__user_items = np.unique(user_id)
        self.__item_items = np.unique(item_id)

        user_number = len(self.__user_items)
        item_number = len(self.__item_items)

        self.__user_vector = np.random.uniform(size=(user_number, dimension))
        self.__user_bias = np.zeros((user_number, 1))
        self.__item_vector = np.random.uniform(size=(item_number, dimension))
        self.__item_bias = np.zeros((item_number, 1))

        loss = []
        for _ in range(epochs):
            index = np.random.randint(0, data_number)

            user_index = np.flatnonzero(self.__user_items == user_id[index])
            item_index = np.flatnonzero(self.__item_items == item_id[index])

            r = (self.__user_vector[user_index].dot(self.__item_vector[item_index].T) + self.__user_bias[user_index] + self.__item_bias[item_index] - y[index])

            loss.append(r.ravel() ** 2)

            user_vector_new = self.__user_vector[user_index] - learning_rate * r * self.__item_vector[item_index]
            self.__user_bias[user_index] -= learning_rate * r
            item_vector_new = self.__item_vector[item_index] - learning_rate * r * self.__user_vector[user_index]
            self.__item_bias[item_index] -= learning_rate * r
            
            self.__user_vector[user_index] = user_vector_new
            self.__item_vector[item_index] = item_vector_new
            
        plt.plot(loss)
        plt.show()

    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (data_number, 2)
            Predicting data, column 1 is user id, column 2 is item id

        Returns
        -------
        y : shape (data_number, 1)
            Predicted rating per sample.
        '''
        data_number = X.shape[0]
        user_id = X[:, 0]
        item_id = X[:, 1]

        y = np.zeros((data_number, 1))
        for i in range(data_number):
            user_index = np.flatnonzero(self.__user_items == user_id[i])
            item_index = np.flatnonzero(self.__item_items == item_id[i])
            y[i] = self.__user_vector[user_index].dot(self.__item_vector[item_index].T) + self.__user_bias[user_index] + self.__item_bias[item_index]

        return y