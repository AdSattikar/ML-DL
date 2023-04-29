# ML-DL
Repo for machine Learning and Deep Learning Projects


[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository contains several machine learning classifiers implemented in Python and examples of how to evaluate their performance using various metrics.

## Table of Contents

- [Breast Cancer Classifier](#breast-cancer-classifier)
- [Handwritten Digits Classifier](#handwritten-digits-classifier)
- [Sentiment Analysis Classifier](#sentiment-analysis-classifier)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Breast Cancer Classifier

In this project,a breast cancer classifier is implemented using the ID3 algorithm for decision tree induction. The classifier takes in a set of features and outputs whether the breast tumor is benign or malignant. The dataset used for training and testing the classifier is the [Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)). 

## Handwritten Digits Classifier

In this project,a handwritten digits classifier is implemented using DCGAN and TensorFlow on the MNIST dataset. The generator and discriminator networks of the DCGAN were trained using TensorFlow. The classifier takes in a handwritten digit image and outputs the predicted digit is real or fake. 

## Sentiment Analysis Classifier

In this project,a Naive Bayes classifier is implemented for sentiment analysis of app reviews on the Google Play Store. The classifier takes in a review and outputs whether it is positive or negative. The dataset used for training and testing the classifier is the [Google Play Store Apps dataset](https://www.kaggle.com/lava18/google-play-store-apps).

## Performance Metrics

In this project, the performance of an MLP Regressor model is evaluated using various metrics such as R-squared, mean squared error (MSE), and mean absolute error (MAE). 

## Installation

To use the classifiers and metrics in this repository, you will need to install Python and the following libraries:

- NumPy
- Pandas
- SciPy
- Scikit-learn
- TensorFlow


## Contributing

If you would like to contribute to this repository, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
