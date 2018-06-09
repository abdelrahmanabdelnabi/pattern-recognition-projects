# Pattern Recognition Course HWs and Assigments
This repository contains my solutions to the HWs and assignments of CC474 Pattern Recognition Spring 2018 at Alexandria University.

The [course webpage](https://sites.google.com/view/ssp-pr-torki/home) has the lectures, HWs, and assignments.

## Assignments

### Face Recognition
- Facial recognition on the ORL faces database
- Implemented PCA and LDA
- Evaluating classification accuracy with a KNN classifier for different values of explained variance
- Plotting of eigen faces

### NLP and Sentiment Analysis (this project is in a separate [repository](https://github.com/abdelrahmanabdelnabi/Sentiment-Analysis))
Sentiment Analysis on the IMDB movie reviews dataset
- data preprocessing (lemmatization, stop-word removal, tokenization, ...)
- Evaluation of several text vectorization techniques
    * Bag of Words
    * TF-IDF
    * pretrained Glove embeddings weighted with IDF
    * word embeddings generated from the dataset
    * Doc2Vec
    * Doc2VecC
- Comparing classification results using several classifiers
- Analysis of the effects of different pre-processing techniques, classifier choice, and vectorization techniques

### Modulation Recognition
Deep learning for modulation scheme recognition and classification on a synthetic dataset, generated with GNU Radio, consisting of 11 modulations with variable-SNR.
- Implemented a baseline fully connected neural network using Keras
- Grid searched for the hyper paramters values
- Implemented a convolutional neural network
- Tried different combinations of input features (raw signal, derivative or signal, integral of signal, and all their combinations)
- Regularized with early stopping
- Compared results of CNN against baseline
- plots of confusion matrices and accuracy vs SNR

### Image Segmentation
- Implemented Spectral Clustering on the 5-NN graph of an image
- Tested against KMeans clustering
- Analysed the effect of encoding the spatial layout
- Compared results of different values of K using F1 score

***

## HWs
1. Principal Component Analysis (PCA)
    - implementation of PCA for dimensionality reduction.
    - consine similarity, euclidean distance, and analysis of expalained variance.
2. Linear Discriminant Analysis (LDA)
    - implementation of LDA
3. Probabilistic Classification
    - implementation of Naive Bayes Classifier
    - testing on the ORL faces database
    - written problems on Full and Naive Bayes
4. Decision Trees
    - Implementation of Decision Tree Classifer
    - Testing the ORL faces database
    - Comparing results with Sci-kit Learns implementation
5. SVM, Ensemble Methods, Linear Regression
    - Written problems on SVM
    - Using Sci-kit learn's SVM implementation
    - Written problems on linear regression and comparing results with Sci-kit learn
6. Clustering with KMeans
    - Implementation of KMeans
    - Written problems on KMeans
    - Experimenting with different distance functions
7. Normalized Cut and Similarity Graphs
    - Implementation of Normailzed Cut Spectral Clustering
    - Experimenting with different similarity matrices
8. Clustering Evaluation
    - External Evaluation measures (Purity, Conditional Entropy, Pairwise Rand and Jaccard Indices)
    - Internal Measures (BetaCV, normalized Cut) (Not Finished)
