#-------------------------------------------------------------------------
# AUTHOR: Wesley Kwan
# FILENAME: knn
# SPECIFICATION: Performs LOO-CV for 1NN
# FOR: CS 4200- Assignment #2
# TIME SPENT: 45 mins
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

wrong_predictions = 0
num_predictions = 0
#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]
    #--> add your Python code here
    # X =
    X = []
    for instance in db:
        X.append([float(i) for i in instance[0:2]])
    X.pop(i)

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]
    #--> add your Python code here
    # Y =
    Y = []
    for instance in db:
        if instance[2] == '+':
            Y.append(1)
        else:
            Y.append(2)
    Y.pop(i)

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    #testSample =
    testSample = db[i]

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([[float(i) for i in testSample[0:2]]])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if testSample[2] == '+' and class_predicted != 1:
        wrong_predictions+=1
    elif testSample[2] == '-' and class_predicted != 2:
        wrong_predictions+=1
    num_predictions+=1

#print the error rate
#--> add your Python code here
error_rate = wrong_predictions / num_predictions
print('error rate of LOO-CV for 1NN: %.2f' % error_rate)





