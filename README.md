# MNIST Sign language dataset using fully connected neural networks and convolutional neural network
## A course work grouped project for the course HPI Deep Learning 2023
This project aim to implement 2 machine learning models on the data set MNIST sign language.
## Data
The original MNIST image dataset of handwritten digits is a popular benchmark for image-based machine learning methods but researchers have renewed efforts to update it and develop drop-in replacements that are more challenging for computer vision and original for real-world applications. The dataset format is patterned to match closely with the classic MNIST. Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions). The training data (27,455 cases) and test data (7172 cases) are approximately half the size of the standard MNIST but otherwise similar with a header row of label, pixel1, pixel2 ... pixel784 which represent a single 28x28 pixel image with grayscale values between 0-255. The train and test data have been concatenated and can be retrieved by selecting the first 27,455 rows for train and the last 7172 for test.

Data set link : https://www.openml.org/search?type=data&status=active&id=45082

Download the dataset and put the train and test data .csv files into the `data` folder.
## Installation Instructions
Requirements:
1. Python 3.8
1. Pandas and Seaborn library
    `pip install pandas`
    `pip install seaborn`
1. skorch library
    `pip install seaborn`
1. Pytorch : https://pytorch.org/get-started/locally/
2. Optuna : `pip install optuna`
3. or use `pip install -r requirements.txt`
## Usage
Simply run the python script and provide architecture to be used

To use a normal fully connected neural network:

`python main.py -fc`

To use a convolutional neural network:

`python main.py -cnn`

To use a grid search with the convolutional neural network:

`python main.py -gs`

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Contact
phan@uni-potsdam.de

hugo.jorge@campus.fct.unl.pt
