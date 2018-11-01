import numpy as np
import metrics

def lda(X, y, component_number):
    feature_number = X.shape[1]
    classes = np.unique(y)
    class_number = len(classes)

    mean = np.mean(X, axis=0)
    means = np.zeros((class_number, feature_number))
    s_b = 0
    s_w = 0
    for i in range(class_number):
        items = np.where(y == classes[i])[0]
        number = len(items)
        means[i] = np.mean(X[items], axis=0)
        
        s_b += number * (means[i] - mean).reshape((-1, 1)).dot((means[i] - mean).reshape((1, -1)))

        diff1 = (X[items] - means[i])[:,:,np.newaxis]
        diff2 = np.transpose(diff1, axes=(0, 2, 1))
        s_w += np.tensordot(diff1, diff2, axes=(0, 0)).reshape((feature_number, feature_number))

        '''
        for item in items:
            s_w += (X[item] - means[i]).reshape((-1, 1)).dot((X[item] - means[i]).reshape((1, -1)))
        '''
    
    eig_values, eig_vectors = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))
    eig_vectors = eig_vectors[:, np.argsort(-eig_values)]

    pc = X.dot(eig_vectors[:, :component_number])
    if component_number == 2:
        metrics.scatter_feature(pc, y)

    return pc