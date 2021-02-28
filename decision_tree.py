#-------------------------------------------------------------------------
# AUTHOR: Wesley Kwan
# FILENAME: decision_tree
# SPECIFICATION: Trains and tests decision trees using different training sets
# FOR: CS 4200- Assignment #2
# TIME SPENT: 45 mins
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    # X =
    value_dictionary = {'Young':1, 'Prepresbyopic':2, 'Presbyopic':3,
                        'Myope':1, 'Hermetrope':2, 'Hypermetrope':3,
                        'No':1, 'Yes':2,
                        'Reduced':1, 'Normal':2}
    
    for instance in dbTraining:
        instance_transform = []
        for attribute_value in instance[0:4]:
            instance_transform.append(value_dictionary[attribute_value])
        X.append(instance_transform)

    #transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =
    for instance in dbTraining:
        if instance[len(instance)-1] == 'Yes':
            Y.append(1)
        else:
            Y.append(2)

    accuracy_list = []
    #loop your training and test tasks 10 times here
    for i in range (10):

       #fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
       clf = clf.fit(X, Y)

       #read the test data and add this data to dbTest
       #--> add your Python code here
       # dbTest =
       dbTest = []
       with open('contact_lens_test.csv', 'r') as csvfile:
           reader = csv.reader(csvfile)
           for i, row in enumerate(reader):
               if i > 0: #skipping the header
                   dbTest.append(row)
       
       true_positive = 0
       true_negative = 0
       num_classifications = 0
       for data in dbTest:
           #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
           #class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
           data_transform = []
           for attribute_value in data[0:4]:
               data_transform.append(value_dictionary[attribute_value])
           class_predicted = clf.predict([data_transform])[0]

           #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
           if data[4] == 'Yes' and class_predicted == 1:
               true_positive+=1
           elif data[4] == 'No' and class_predicted == 2:
               true_negative+=1
           num_classifications+=1

       #find the lowest accuracy of this model during the 10 runs (training and test set)
       #--> add your Python code here
       accuracy_list.append((true_positive + true_negative) / num_classifications)

    #print the lowest accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that:
         #final accuracy when training on contact_lens_training_1.csv: 0.2
         #final accuracy when training on contact_lens_training_2.csv: 0.3
         #final accuracy when training on contact_lens_training_3.csv: 0.4
    #--> add your Python code here
    lowest_accuracy = min(accuracy_list)
    print('final accuracy when training on %s: %.3f' % (ds, lowest_accuracy))




