# Classic ML Algorithms for Text & Image Classification and Segmentation

This repository contains implementations of foundational machine learning algorithms applied to both text and image data. Each section focuses on implementing the algorithms from scratch using Python, with a focus on understanding the inner workings of classification and segmentation models. This project was developed as part of the "Deep Learning for Computer Vision" course at GUC.

## 📘 Task 1: Naïve Bayes for Text Classification

Objective: Implement a Naïve Bayes classifier from scratch to classify IMDB movie reviews as positive or negative.

### Key Steps:

#### Data Preprocessing:

Tokenization

Lowercasing

Stopword removal

Non-alphabetic character removal

#### Feature Extraction:

Construct frequency tables for word counts in positive and negative reviews

Compute probabilities: P(positive), P(negative), P(word|positive), P(word|negative)

#### Classification:

Implement classifier based on Naïve Bayes probability formula

Classify new reviews

#### Evaluation:

Test classifier on unseen data and calculate accuracy

Report classification results

## 🖼️ Task 2: Naïve Bayes for Color Image Classification

Objective: Implement a binary Naïve Bayes classifier for color image classification using the BSDS300 dataset.

### Key Steps:

#### Data Representation:

Flatten RGB images into feature vectors

Use gray-level thresholding to generate binary ground truth labels

#### Implementation:

BayesModel(dta, gt) to train model

BayesPredict(BM, td) to make predictions

ConfMtrx(gt, lbl) to generate confusion matrix

#### Evaluation:

Visualize predicted vs ground truth segmentations

Report confusion matrix

## 🧠 Task 3: EM Algorithm for Image Segmentation

Objective: Apply the Expectation-Maximization algorithm to segment foreground and background in an image.

### Key Steps:

#### Using scikit-learn's EM/GMM:

Load and reshape image

Apply GMM for clustering

Visualize segmented image

#### Implement EM from Scratch:

E-step: Assign pixel probabilities

M-step: Update parameters

Repeat until convergence

#### Output:

Segmented binary image separating object and background

## 🎨 Task 4: Image Segmentation using K-Means

Objective: Segment images using K-means clustering with pixel features.

### Key Steps:

Feature Construction:

Extract RGB and spatial (x, y) coordinates

#### Clustering:

Apply K-means for various values of k (2, 5, 10, 15)

Assign cluster colors and visualize results

## 🐶🐘 Task 5: KNN for Image Classification

Objective: Build a K-Nearest Neighbors (KNN) classifier to distinguish between "elephant" and "bus" images from a filtered CIFAR-100 dataset.

### Key Steps:

#### Feature Extraction:

Divide image into 4x4 grid and calculate mean RGB for each block (16x3 vector)

#### KNN Implementation:

Compute Euclidean distance

Assign label by majority voting

#### Evaluation:

Measure and report classification accuracy
