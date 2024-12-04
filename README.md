# git-repo
# Heart Disease Prediction
The aim of this project is to build a ML model to analyse the features and predict the presence or absence of Heart Disease based on various features and demographic informaton.
The project involved analysis of the heart disease patient dataset with proper data processing. Then, different models were trained and and predictions are made with different algorithms Decision Tree, Logistic Regression and MLP Classifier. 

This repository contains the Python code, dataset and instructions to run the heart disease prediction model.

# Prerequisites
To run this project, you need:
Python 3.8 and above

# Installing
### Libraries 
pandas - For data manipulation and analysis, including loading and cleaning the dataset.
numpy - Provides support for numerical computations and array manipulations.
matplotlib - For creating static, interactive, and visually appealing data visualizations.
seaborn - Built on top of Matplotlib, it simplifies complex visualizations like histograms and count plots.
sklearn (Scikit-learn) - For machine learning tasks such as splitting datasets, OneHotEncoding, StandardScaler, building models and evaluating performance.

### IDE 
For this project I used Jupyter environment to write, debug and run Python code efficiently.

### Ensure the following before running the code:
- Python 3.8 or above version is installed.
- Install the libraries using pip.
- Place the dataset file (heart_disease.csv) in the project directory.
- Use an IDE to run the Python script.
- 
### Steps to follow
1. Clone the repository: In Gitbash terminal write git clone and paste the repository link
   git clone https://github.com/shil-ks/git-repo.git
2. Navigate into the cloned repository directory:
   cd git-repo
3. After successfully downloading the repository, there should be a folder with the name of the repository on your local machine.
4. Use the IDE Anaconda Prompt - Jupyter notebook and set the directory
5. Install Pre-requisites and Libraries mentioned above.
6. Ensure the dataset heart_disease.csv path is correct 
8. Cells --> Run all or you can run cell by cell to see and understand the ouptut for the code.


### Dataset   
The dataset I've used for this project is taken from open source kaggle.
It consist of 18 attributes of which 17 are indenpendent variables and one dependant variable (which is the trarget variable), heart_disease has the values No/Yes.
Dataset Link: https://www.kaggle.com/datasets/luyezhang/heart-2020-cleaned/data

### Understanding the data:
To understand the data better we use functions such as 
.head() - to display top 5 rows of the dataset
.isnull().sum() - To check if the dataset has any null values
.info() - to get information about the dataset
.describe() - to get descriptive information about the dataset.

### Exploratory Data Analysis
Performed EDA to draw meaningful insights from the data. 

### Data Pre-Processing

Re-arranged the columns in the data 
Printed the unique values to understand how further data preprocessing to be done.
The data has categorical variables and in order to make the data Machine Learning ready, I used OneHotEncoder and StandardScalar to convert the categorical data into binary matrix form and normalized numerical variables to ensure a uniform scale.
Split the data into training and testing set using train_test_split.

# Testing - Breakdown

This data is further trained using ML algorithms 
Logistic Regression (Scikit-learn)
DecisionTree Classifier (Scikit-learn)
MLP Classifier - (neural_network)

The following metrics were used to evaluate model performance:

Confusion Matrix: Provides the number of true positives, true negatives, false positives, and false negatives.
Accuracy: Percentage of correct predictions out of total predictions.
AUC (Area Under Curve): Measures the ability of the model to distinguish between classes.
Classification Report: Includes precision, recall, and F1-score for detailed analysis.

![Metrics Evaluation Summary](https://github.com/shil-ks/git-repo/blob/main/images/Metrics%20Evaluation%20Summary.png)

Upon evaluting all these metrices, decided that Logistic Regression is the most consistent and interpretable model for this dataset, given its balance of high accuracy of 91.39% and reasonable AUC. Therefore, I used Logistic Regression Model to make predictions on test data.

# Deployment
This project is currently not deployed.
One could use Streamlit.app,  FlaskAPI for deployment.

# Author
Shilpa Kumaraswamy 
Email ID - shilpak2705@gmail.com

# License
This project is licensed under the [MIT License](https://github.com/shil-ks/git-repo/blob/main/LICENSE). See the LICENSE file for details

# Acknowledgements
This project is used for Introduction to Data Analysis by Prof. Ziyad Mohamed at Durham College
Kaggle for providing the dataset




