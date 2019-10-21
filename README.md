# Contents
- [Supervised Learning](#supervised-learning)
  - [Classification](#classification)
  - [Regression](#regression)
  - [Classification and Regression](#classification-and-regression)
  - [Dimensionality Reduction](#dimensionality-reduction)
- [Unsupervised Learning](#unsupervised-learning)
  - [Clustering](#clustering)
  - [Dimensionality Reduction](#dimensionality-reduction) 
  - [Abnormal Detection](#abnormal-detection)
  - [Others](#others)
 - [Tools](#tools)
 
# Supervised Learning

## [neural_network.py](neural_network.py)
    neural network algorithm including following components
    layers: conv1d, conv2d, mean pool, max pool, drop out, batch normalization, rnn, flatten, dense, tanh, relu
    loss: binary crossentropy, categorical crossentropy, mse, categorical_hinge

## Classification

### [pla.py](pla.py)
    perceptron algorithm
    
### [pocket.py](pocket.py)
    pocket algorithm
    
### [logistic_regression.py](logistic_regression.py)
    logistic regression algorithm including gradient descent, newton
    
### [softmax_regression.py](softmax_regression.py)
    softmax regression algorithm

### [learning_vector_quantization.py](learning_vector_quantization.py)
    learning vector quantization algorithm
    
### [knn.py](knn.py)
    k-NearestNeighbor algorithm

### [gaussian_discriminant_analysis.py](gaussian_discriminant_analysis.py)
    gaussian discriminant analysis algorithm

### [svm.py](svm.py)
    support vector machine algorithm including following components
    kernel: linear, rbf
    solver: smo, quadratic programming

### [discrete_adaboost.py](discrete_adaboost.py)
    discrete adaboost algorithm

### [real_adaboost.py](real_adaboost.py)
    real adaboost algorithm

### [gentle_adaboost.py](gentle_adaboost.py)
    gentle adaboost algorithm

### [naive_bayesian_for_text.py](naive_bayesian_for_text.py)
    naive bayesian algorithm for text classification

### [decision_tree_id3.py](decision_tree_id3.py)
    decision tree id3 algorithm

### [decision_tree_c45.py](decision_tree_c45.py)
    decision tree c45 algorithm

## Regression

### [linear_regression.py](linear_regression.py)
    linear regression algorithm including gradient descent, newton, equation
    
### [linear_regression_locally_weight.py](linear_regression_locally_weight.py)
    locally weight linear regression algorithm
    
### [collaborative_filtering.py](collaborative_filtering.py)
    collaborative filtering algorithm

## Classification and Regression

### [rbf_network.py](rbf_network.py)
    rbf network algorithm

### [decision_tree_cart.py](decision_tree_cart.py)
    decision tree cart algorithm

### [random_forest.py](random_forest.py)
    random_forest algorithm including bagging, random features, oob verification, feature selection

### [gbdt.py](gbdt.py)
    gradient boost decision tree algorithm

## Dimensionality Reduction

### [linear_discriminant_analysis.py](linear_discriminant_analysis.py)
    linear discriminant analysis algorithm with "eigen" solver
 
# Unsupervised Learning

## Clustering

### [k_means.py](k_means.py)
    k-means algorithm
    
### [k_means_plus.py](k_means_plus.py)
    k-means++ algorithm
    
### [bisecting_kmeans.py](bisecting_kmeans.py)
    bisecting k-means algorithm
    
### [k_median.py](k_median.py)
    k-median algorithm

### [k_mediods.py](k_mediods.py)
    k-mediods algorithm

### [fuzzy_c_means.py](fuzzy_c_means.py)
    fuzzy c means algorithm
    
### [gaussian_mixed_model.py](gaussian_mixed_model.py)
    gaussian mixed model algorithm

### [agnes.py](agnes.py)
    agnes clustering algorithm
    
### [diana.py](diana.py)
    diana clustering algorithm

### [dbscan.py](dbscan.py)
    dbscan clustering algorithm

### [spectral_clustering.py](spectral_clustering.py)
    spectral clustering algorithm
    
## Dimensionality Reduction

### [pca.py](pca.py)
    principal Component Analysis algorithm including whiten, zero-phase component analysis whiten, kernel pca

### [mltidimensional_scaling.py](mltidimensional_scaling.py)
    mltidimensional scaling algorithm
    
### [locally_linear_embedding.py](locally_linear_embedding.py)
    locally linear embedding algorithm

## abnormal detection

### [support_vector_data_description.py](support_vector_data_description.py)
    support vector data description algorithm

### [isolate_forest.py](isolate_forest.py)
    isolate forest algorithm

## Others

### [ica.py](ica.py)
    independent component analysis algorithm
    
# Tools

## [image_preprocess.py](image_preprocess.py)
    image preprocess algorithms including rgb2gray, histogram of oriented gradient

## [text_preprocess.py](text_preprocess.py)
    text preprocess algorithms including tf-idf
    
## [distance.py](distance.py)
    distance algorithms including euclidean distance, manhattan distance, chebyshev distance, mahalanobis distance, cosine distance

## [kernel.py](kernel.py)
    kernel function including linear, rbf
    
## [preprocess.py](preprocess.py)
    preprocess algorithm including min-max scaler, z-score scaler, one-hot encoder, bagging
    
## [regularizer.py](regularizer.py)
    regularizer algorithm including L1, L2, elastic-net
    
## [metrics.py](metrics.py)
    scores including accuracy, precision, recall, f-score, R2 score, confusion matrix, pr curve, roc curve, auc, silhouette coefficient, 2d feature scatter, learning curve, information value

## [optimizer.py](optimizer.py)
    optimizer algorithm including following components
    gradient descent: momentum，nesterov, adagrad，rmsprop，adam

## [weights_initializer.py](weights_initializer.py)
    weights initialization algorithm
# ---------------------------------------------------
# nnet.py
    neural network algorithm including following components
    layers: conv1d, conv2d, max pool, mean pool, flatten, dense, dropout, batch normalization, rnn, relu, selu, tanh
    loss: binary crossentropy, categorical crossentropy, mse, categorical hinge
