#-------------------------------------------------------------------------
# AUTHOR: Wesley Kwan
# FILENAME: naive_bayes
# SPECIFICATION: Uses Naive Bayes strategy to classify test instances
# FOR: CS 4200- Assignment #2
# TIME SPENT: 60 mins
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB

#reading the training data
#--> add your Python code here
import csv

dbTraining = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            dbTraining.append(row)

#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# X =
value_dictionary = {'Sunny':1, 'Overcast':2, 'Rain':3,
                    'Hot':1, 'Mild':2, 'Cool':3,
                    'High':1, 'Normal':2,
                    'Weak':1, 'Strong':2}

X = []
for instance in dbTraining:
    instance_transform = []
    for attribute_value in instance[1:5]:
        instance_transform.append(value_dictionary[attribute_value])
    X.append(instance_transform)

#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =
Y = []
for instance in dbTraining:
    if instance[len(instance)-1] == 'Yes':
        Y.append(1)
    else:
        Y.append(2)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the data in a csv file
#--> add your Python code here
dbTest = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            dbTest.append(row)

dbTest_transform = []
for instance in dbTest:
    instance_transform = []
    for attribute_value in instance[1:5]:
        instance_transform.append(value_dictionary[attribute_value])
    dbTest_transform.append(instance_transform)

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions.
#--> add your Python code here
#-->predicted = clf.predict_proba([[3, 1, 2, 1]])[0]
for i, instance in enumerate(dbTest_transform):
    if i < len(dbTest):
        class_probability = clf.predict_proba([instance])[0]
        class_confidence = class_probability[1]
        class_predicted = 'No'
        if class_probability[0] > class_probability[1]:
            class_confidence = class_probability[0]
            class_predicted = 'Yes'
        if max(class_probability) > 0.75:
            print('%-15s%-15s%-15s%-15s%-15s%-15s%.2f' % (dbTest[i][0], dbTest[i][1], dbTest[i][2], 
                                                          dbTest[i][3], dbTest[i][4], class_predicted, 
                                                          class_confidence))


