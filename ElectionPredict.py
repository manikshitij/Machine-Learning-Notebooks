# Machine Learning Algorithm #2 : Logistic Regression (Binary Classification)
# Author : Anuradha C

# Program to predict winner of an election between 2 candidates, depending on voter data
# Using Logistic Regression ML Algorithm

import pandas as pd
import numpy as np
import collections

# check deprecation
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

DATA_SET_PATH = 'C://Users/anu/workspace/Python/ML_BD/Election.csv'
 
def main():
    """
    Logistic Regression classifier main
    :return:
    """
    # Load the data set for training and testing the logistic regression classifier
    dataset = pd.read_csv(DATA_SET_PATH)
    print("---------------------------------------------------------")
    print("Election Result Prediction - Logistic Regression")
    print("---------------------------------------------------------")
    print ("Number of Observations in data set :: ", len(dataset))

	# Get the first observation
    print("Sample of the top few rows in the data set ::")
    print(dataset.head())

    headers = dataset_headers(dataset)
    print("Data set headers :: {headers}".format(headers=headers))
	
    #Only the following 5 independent X variables are considered
    training_features = ['TVnews', 'PID', 'age', 'educ', 'income']
    #The last column in the data is the dependent Y result
    target = 'vote'
    
	# Train , Test data split. Randomly allot 70% data sets to trial
    train_x, test_x, train_y, test_y = train_test_split(dataset[	training_features], dataset[target], train_size=0.7)
    
	# See how strong is the relation between education and vote
    print("Target_frequencies for the parameter educ :: ", feature_target_frequency_relation(dataset, [training_features[3], target]))

    # Size of training data
    print("train_x size :: ", train_x.shape)
    print("train_y size :: ", train_y.shape)

    # Size of testing data
    print("test_x size :: ", test_x.shape)
    print("test_y size :: ", test_y.shape)

    # Number of votes per candidate , 1s and 0s
    print("Vote count in training data ::", count_votes(train_y))
    print("Vote count in test data ::", count_votes(test_y))
	
	# These are the 5 independent factors that affect poll result
    print ("5 independent variables chosen for regression ::" , training_features)
	
    # Training Logistic regression model
    trained_logistic_regression_model = train_logistic_regression(train_x, train_y)
	
    predict_y = trained_logistic_regression_model.predict(test_x)
    print("Prediction ::", predict_y)
    print("Total number of voters in prediction", predict_y.shape)
    print("Vote count in prediction ::", count_votes(predict_y))
    
    train_accuracy = model_accuracy(trained_logistic_regression_model, train_x, train_y)

    # Testing the logistic regression model
    test_accuracy = model_accuracy(trained_logistic_regression_model, test_x, test_y)

    print("Train Accuracy :: ", train_accuracy)
    print("Test Accuracy :: ", test_accuracy)

# End of Main Function


def count_votes(dataset):
    """
    To get count of y values in the data set 
    :param dataset: loaded dataset into pandas DataFrame
    :return: count of 1s and 0s
    """
    counter=collections.Counter(dataset)
    return counter

def dataset_headers(dataset):
    """
    To get the dataset header names
    :param dataset: loaded dataset into pandas DataFrame
    :return: list of header names
    """
    return list(dataset.columns.values)

def unique_observations(dataset, header, method=1):
    """
    To get unique observations in the loaded pandas DataFrame column
    :param dataset:
    :param header:
    :param method: Method to perform the unique (default method=1 for pandas and method=0 for numpy )
    :return:
    """
    try:
        if method == 0:
            # With Numpy
            observations = np.unique(dataset[[header]])
        elif method == 1:
            # With Pandas
            observations = pd.unique(dataset[header].values.ravel())
        else:
            observations = None
            print("Wrong method type, Use 1 for pandas and 0 for numpy")
    except Exception as e:
        observations = None
        print("Error: {error_msg} /n Please check the inputs once..!".format(error_msg=e.message))
    return observations	
	
	
def feature_target_frequency_relation(dataset, f_t_headers):
 
    """
    To get the frequency relation between targets and the unique feature observations
    :param dataset:
    :param f_t_headers: feature and target header
    :return: feature unique observations dictionary of frequency count dictionary
    """
 
    feature_unique_observations = unique_observations(dataset, f_t_headers[0])
    unique_targets = unique_observations(dataset, f_t_headers[1])
 
    frequencies = {}
    for feature in feature_unique_observations:
        frequencies[feature] = {unique_targets[0]: len(
            dataset[(dataset[f_t_headers[0]] == feature) & (dataset[f_t_headers[1]] == unique_targets[0])]),
            unique_targets[1]: len(
                dataset[(dataset[f_t_headers[0]] == feature) & (dataset[f_t_headers[1]] == unique_targets[1])])}
    return frequencies

def train_logistic_regression(train_x, train_y):
    """
    Training logistic regression model with train dataset features(train_x) and target(train_y)
    :param train_x:
    :param train_y:
    :return:
    """

    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(train_x, train_y)
	
    print ("y = m1*x1 + m2*x2 + m3*x3 + m4*x4 + m5*x5 + c")
    print ("Co-efficients m1-m5 :: ", logistic_regression_model.coef_ )

    return logistic_regression_model


def model_accuracy(trained_model, features, targets):
    """
    Get the accuracy score of the model
    :param trained_model:
    :param features:
    :param targets:
    :return:
    """
    accuracy_score = trained_model.score(features, targets)
    return accuracy_score

	
if __name__ == "__main__":
    main()