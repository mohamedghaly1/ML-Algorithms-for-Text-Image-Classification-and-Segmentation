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

![image](https://github.com/user-attachments/assets/5d824e5f-61de-46ff-a8fa-f22d6a86d2a5)

![image](https://github.com/user-attachments/assets/e2416218-0555-4807-b77f-9f153a760b7b)



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

![image](https://github.com/user-attachments/assets/564122d6-08a8-4f2c-ba97-6affc78fd995)


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

![image](https://github.com/user-attachments/assets/12bc5166-596e-4ffc-a3d2-46a8334a2860)
![image](https://github.com/user-attachments/assets/5079a3fd-f2e9-4039-9b26-0aad51d0f661)



## 🎨 Task 4: Image Segmentation using K-Means

Objective: Segment images using K-means clustering with pixel features.

### Key Steps:

Feature Construction:

Extract RGB and spatial (x, y) coordinates

#### Clustering:

Apply K-means for various values of k (2, 5, 10, 15)

Assign cluster colors and visualize results

![image](https://github.com/user-attachments/assets/09695ad8-af63-4791-89b4-fb2cbc51140b)


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

#### Accuracy:

KNN Classifier Accuracy: 70.50%

Details of 5 classification results:
Sample 1: True Label: Bus,	Predicted: Elephant
Sample 2: True Label: Bus,	Predicted: Bus
Sample 3: True Label: Elephant,	Predicted: Bus
Sample 4: True Label: Elephant,	Predicted: Elephant
Sample 5: True Label: Elephant,	Predicted: Elephant

![image](https://github.com/user-attachments/assets/d88ee966-4644-4a2d-ac1b-28a1decc5b81)




