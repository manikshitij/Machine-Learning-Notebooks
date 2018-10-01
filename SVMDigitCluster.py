from sklearn import datasets

# Import `train_test_split`
from sklearn.cross_validation import train_test_split

# Import the `svm` model
from sklearn import svm

# Import `digits dataset` from `sklearn`
# Load in the `digits` data
digits = datasets.load_digits()

# Split the data into training and test sets 
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(digits.data, digits.target, digits.images, test_size=0.25, random_state=42)

# Create the SVC model 
svc_model = svm.SVC(gamma=0.001, C=100., kernel='linear')

# Fit the data to the SVC model
svc_model.fit(X_train, y_train)

# Train and score a new classifier with the grid search parameters
score = svm.SVC(C=10, kernel='rbf', gamma=0.001).fit(X_train, y_train).score(X_test, y_test)

# Predict the labels for `X_test`
y_pred=svc_model.predict(X_test)

# Print out the first 100 instances of `y_pred`
print("First 100 instances of predicted y")
print(y_pred[:100])

# Print out the first 100 instances of `y_test`
print("First 100 instances of test data y")
print(y_test[:100])

#Print model accuracy by comparing the y test and y prediction values 
print("Accuracy : ", 'Accuracy:{0:f}'.format(score))