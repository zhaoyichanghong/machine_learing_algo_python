import numpy as np
import matplotlib.pyplot as plt

class CollaborativeFiltering:
    def fit(self, X, y, dimension, learning_rate, epochs):
        data_number = X.shape[0]
        user_id = X[:, 0]
        movie_id = X[:, 1]
        
        user_number = len(np.unique(user_id))
        movie_number = len(np.unique(movie_id))

        self.__user_vector = np.random.uniform(size = (user_number, dimension))
        self.__movie_vector = np.random.uniform(size = (dimension, movie_number))

        self.__user_items = np.unique(user_id)
        self.__movie_items = np.unique(movie_id)

        loss = []

        for _ in range(epochs):
            index = np.random.randint(0, data_number)

            user_index = np.flatnonzero(self.__user_items == user_id[index])
            movie_index = np.flatnonzero(self.__movie_items == movie_id[index])

            r = (self.__user_vector[user_index].dot(self.__movie_vector[:, movie_index]) - y[index])[0][0]

            loss.append(r ** 2)

            user_vector_new = self.__user_vector[user_index] - learning_rate * r * self.__movie_vector[:, movie_index].T
            movie_vector_new = self.__movie_vector[:, movie_index] - learning_rate * r * self.__user_vector[user_index].T
            
            self.__user_vector[user_index] = user_vector_new
            self.__movie_vector[:, movie_index] = movie_vector_new
            
        plt.plot(loss)
        plt.show()

    def predict(self, X):
        data_number = X.shape[0]
        user_id = X[:, 0]
        movie_id = X[:, 1]

        y = np.zeros((data_number, 1))

        for i in range(data_number):
            user_index = np.flatnonzero(self.__user_items == user_id[i])
            movie_index = np.flatnonzero(self.__movie_items == movie_id[i])
            y[i] = self.__user_vector[user_index].dot(self.__movie_vector[:, movie_index])

        return y