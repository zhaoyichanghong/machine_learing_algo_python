import numpy as np

def xavier_normal(input_size, output_size, shape):
    return np.random.normal(scale=np.sqrt(2 / (input_size + output_size)), size=shape)

def xavier_uniform(input_size, output_size, shape):
    return np.random.uniform(low=-np.sqrt(6 / (input_size + output_size)), high=np.sqrt(6 / (input_size + output_size)), size=shape)

def he_normal(input_size, output_size, shape):
    return np.random.normal(scale=np.sqrt(2 / (input_size)), size=shape)

def he_uniform(input_size, output_size, shape):
    return np.random.uniform(low=-np.sqrt(6 / (input_size)), high=np.sqrt(6 / (input_size)), size=shape)
