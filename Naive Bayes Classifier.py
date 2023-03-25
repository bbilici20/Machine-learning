import numpy as np
import pandas as pd
import math


X = np.genfromtxt("hw01_data_points.csv", delimiter = ",", dtype = str)
y = np.genfromtxt("hw01_class_labels.csv", delimiter = ",", dtype = int)



# STEP 3
# first 50000 data points should be included to train
# remaining 43925 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    X_train = X[:50000]
    X_test = X[50000:]
    y_train = y[:50000]
    y_test = y[50000:]
    # your implementation ends above
    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# STEP 4
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    count1 = 0
    count2 = 0
    for i in y:
        if (i == 1):
            count1 += 1
        elif (i == 2):
            count2 += 1
    class_priors = np.array([count1/50000,count2/50000])
    # your implementation ends above
    
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 5
# assuming that there are K classes and D features
# should return four numpy arrays with shape (K, D)
def estimate_nucleotide_probabilities(X, y):
    # your implementation starts below
    a,b = estimate_prior_probabilities(y)
    countA1 = 0
    countC1 = 0
    countG1 = 0
    countT1 = 0
    countA2 = 0
    countC2 = 0
    countG2 = 0
    countT2 = 0
    
    pAcd1 = []
    pAcd2 = []
    pCcd1 = []
    pCcd2 = []
    pGcd1 = []
    pGcd2 = []
    pTcd1 = []
    pTcd2 = []
    
    for m in range(0,7):
        for n in range(0,50000):
            if(X[n][m] == 'A'):
                if (y[n]==1):
                    countA1 +=1
                else:
                    countA2 +=1
            elif(X[n][m] == 'C'):
                if (y[n]==1):
                    countC1 +=1
                else:
                    countC2 +=1
            elif(X[n][m] == 'G'):
                if (y[n]==1):
                    countG1 +=1
                else:
                    countG2 +=1
            elif(X[n][m] == 'T'):
                if (y[n]==1):
                    countT1 +=1
                else:
                    countT2 +=1
        
        pAcd1.append(countA1/a/50000)
        pAcd2.append(countA2/b/50000)
        pCcd1.append(countC1/a/50000)
        pCcd2.append(countC2/b/50000)
        pGcd1.append(countG1/a/50000)
        pGcd2.append(countG2/b/50000)
        pTcd1.append(countT1/a/50000)
        pTcd2.append(countT2/b/50000)
        
        countA1 = 0
        countC1 = 0
        countG1 = 0
        countT1 = 0
        countA2 = 0
        countC2 = 0
        countG2 = 0
        countT2 = 0
    
    pAcd = np.array([pAcd1,pAcd2])
    pCcd = np.array([pCcd1,pCcd2])
    pGcd = np.array([pGcd1,pGcd2])
    pTcd = np.array([pTcd1,pTcd2])
    # your implementation ends above
    return(pAcd, pCcd, pGcd, pTcd)

pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)



# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    # your implementation starts below
    score_values1 = []
    product1 = 1
    product2 = 1
    for m in range(0,len(X)):
            for n in range(0,7):
                if(X[m][n]=='A'):
                    product1 *= pAcd[0][n]
                elif(X[m][n]=='C'):
                    product1 *= pCcd[0][n]
                elif(X[m][n]=='G'):
                    product1 *= pGcd[0][n]
                elif(X[m][n]=='T'):
                    product1 *= pTcd[0][n]
            product1 = math.log(product1)
            product1 += math.log(class_priors[0])
                
            for n in range(0,7):
                if(X[m][n]=='A'):
                    product2 *= pAcd[1][n]
                elif(X[m][n]=='C'):
                    product2 *= pCcd[1][n]
                elif(X[m][n]=='G'):
                    product2 *= pGcd[1][n]
                elif(X[m][n]=='T'):
                    product2 *= pTcd[1][n]
            product2 = math.log(product2)
            product2 += math.log(class_priors[1])
            score_values1.append([product1,product2])
            product1 = 1
            product2 = 1
    
    score_values = np.array(score_values1)
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)



# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    class1_true = 0
    class1_false = 0
    class2_true = 0
    class2_false = 0
    for m in range(0,len(y_truth)):
            if(scores[m][0]>scores[m][1]):
                if(y_truth[m]==1):
                    class1_true +=1
                else:
                    class1_false +=1
            elif(scores[m][0]<scores[m][1]):
                if(y_truth[m]==2):
                    class2_true +=1
                else:
                    class2_false +=1
                
           
    confusion_matrix = np.array([[class1_true,class1_false],[class2_false,class2_true]])            
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
