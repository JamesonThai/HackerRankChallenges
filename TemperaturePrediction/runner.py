# Enter your code here. Read input from STDIN. Print output to STDOUT
import os
import sys
import pandas as pd
import re
import sklearn
from sklearn.impute import SimpleImputer
from sklearn import linear_model as lm
from sklearn.linear_model import LogisticRegression as lr
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import sklearn.ensemble as skensem
import numpy as np
import warnings
# Reference https://www.hackerrank.com/challenges/temperature-predictions/forum

def LinearRegressionRun(X,YTMax, YTMin):
    predictions = []
    modelMax = lm.LinearRegression()
    modelMax.fit(X.loc[:,X.columns != 'tmax'], X.loc[:,'tmax'])
    for i,j in enumerate(modelMax.predict(YTMax)):
        # <class 'int'> <class 'numpy.float64'> for i,j
        predictions.append((YTMax.index[i],round(j,1)))
    modelMin = lm.LinearRegression()
    modelMin.fit(X.loc[:,X.columns != 'tmin'], X.loc[:,'tmin'])
    for i,j in enumerate(modelMin.predict(YTMin)):
        predictions.append((YTMin.index[i], round(j,1)))
    predictions.sort()
    return predictions

def LogisticRegressionRun(X, YTMax, YTMin):
    predictions = []
    modelMax = lr(solver = 'lbfgs', multi_class = 'multinomial')
    modelMax.fit(X.loc[:,X.columns != 'tmax'], X.loc[:,'tmax'])
    for i,j in enumerate(modelMax.predict(YTMax)):
        # print(type(i),type(j))
        predictions.append((YTMax.index[i],round(np.float64(j),1)))
    modelMin = lr(solver = "lbfgs", multi_class = "multinomial")
    modelMin.fit(X.loc[:,X.columns != 'tmin'], X.loc[:,'tmin'])
    for i,j in enumerate(modelMin.predict(YTMin)):
        predictions.append((YTMin.index[i], round(np.float64(j),1)))
    predictions.sort()
    return predictions

def linARDRun(X,YTMax, YTMin):
    predictions = []
    modelMax = lm.ARDRegression()
    modelMax.fit(X.loc[:,X.columns != 'tmax'], X.loc[:,'tmax'])
    for i,j in enumerate(modelMax.predict(YTMax)):
        predictions.append((YTMax.index[i],round(j,1)))
    modelMin = lm.ARDRegression()
    modelMin.fit(X.loc[:,X.columns != 'tmin'], X.loc[:,'tmin'])
    for i,j in enumerate(modelMin.predict(YTMin)):
        predictions.append((YTMin.index[i],round(j,1)))
    predictions.sort()
    return predictions

def ADABoostRegressorRun(X,YTMax, YTMin):
    predictions = []
    modelMax = skensem.AdaBoostRegressor(n_estimators=100, loss = "exponential")
    modelMax.fit(X.loc[:,X.columns != 'tmax'], X.loc[:,'tmax'])
    for i,j in enumerate(modelMax.predict(YTMax)):
        predictions.append((YTMax.index[i],round(j,1)))
    modelMin = skensem.AdaBoostRegressor(n_estimators=100, loss = "exponential")
    modelMin.fit(X.loc[:,X.columns != 'tmin'], X.loc[:,'tmin'])
    for i,j in enumerate(modelMin.predict(YTMin)):
        predictions.append((YTMin.index[i],round(j,1)))
    predictions.sort()
    return predictions

def baggingRegressorRun(X, YTMax, YTMin):
    predictions = []
    modelMax = skensem.BaggingRegressor(n_estimators=100)
    modelMax.fit(X.loc[:,X.columns != 'tmax'], X.loc[:,'tmax'])
    for i,j in enumerate(modelMax.predict(YTMax)):
        predictions.append((YTMax.index[i],round(j,1)))
    modelMin = skensem.BaggingRegressor(n_estimators=100)
    modelMin.fit(X.loc[:,X.columns != 'tmin'], X.loc[:,'tmin'])
    for i,j in enumerate(modelMin.predict(YTMin)):
        predictions.append((YTMin.index[i],round(j,1)))
    predictions.sort()
    return predictions

def ExtraTreesRegressorRun(X,YTMax, YTMin):
    predictions = []
    modelMax = skensem.ExtraTreesRegressor(n_estimators=100)
    modelMax.fit(X.loc[:,X.columns != 'tmax'], X.loc[:,'tmax'])
    for i,j in enumerate(modelMax.predict(YTMax)):
        predictions.append((YTMax.index[i],round(j,1)))
    modelMin = skensem.ExtraTreesRegressor(n_estimators=100)
    modelMin.fit(X.loc[:,X.columns != 'tmin'], X.loc[:,'tmin'])
    for i,j in enumerate(modelMin.predict(YTMin)):
        predictions.append((YTMin.index[i],round(j,1)))
    predictions.sort()
    return predictions 

def GradientBoostingRegressorRun(X,YTMax, YTMin):
    predictions = []
    modelMax = skensem.GradientBoostingRegressor(n_estimators=64)
    modelMax.fit(X.loc[:,X.columns != 'tmax'], X.loc[:,'tmax'])
    for i,j in enumerate(modelMax.predict(YTMax)):
        predictions.append((YTMax.index[i],round(j,1)))
    modelMin = skensem.GradientBoostingRegressor(n_estimators=64)
    modelMin.fit(X.loc[:,X.columns != 'tmin'], X.loc[:,'tmin'])
    for i,j in enumerate(modelMin.predict(YTMin)):
        predictions.append((YTMin.index[i],round(j,1)))
    predictions.sort()
    return predictions 

def RandomForestRegressorRun(X,YTMax, YTMin):
    predictions = []
    modelMax = skensem.RandomForestRegressor(n_estimators=100)
    modelMax.fit(X.loc[:,X.columns != 'tmax'], X.loc[:,'tmax'])
    for i,j in enumerate(modelMax.predict(YTMax)):
        predictions.append((YTMax.index[i],round(j,1)))
    modelMin = skensem.RandomForestRegressor(n_estimators=100)
    modelMin.fit(X.loc[:,X.columns != 'tmin'], X.loc[:,'tmin'])
    for i,j in enumerate(modelMin.predict(YTMin)):
        predictions.append((YTMin.index[i],round(j,1)))
    predictions.sort()
    return predictions 

def evaluator(result):
    MAX_SCORE = 100 
    expected = []
    length = len(result)
    totalSum = 0
    with open('expected.txt', "r") as expectedFile:
        for line in expectedFile:
            expected.append(line.rstrip())
    listOfDifferences = []
    # Compare being that for each line compare their difference 
    for k in range(len(result)):
        predictedVal = result[k][1]
        dif = predictedVal - float(expected[k])
        totalSum += abs(dif)
    average = totalSum / length
    return (1 - (average/5)) * (MAX_SCORE)


def main(temperatures):
    months = {"January":1, "February":2, "March":3, "April":4, "May":5, "June":6, "July":7, "August":8, "September":9, "October":10, "November":11, "December":12}
    # Parse and make dataframe
    yearList,monthList, tmaxList,tminList   = [], [], [], []
    count = 0
    for line in temperatures: 
        row = re.split(r'\t+', line)
        yearList.append(row[0].rstrip())
        monthList.append(row[1].rstrip())
        if "Missing" in row[2]: tmaxList.append(np.nan)
        else:   tmaxList.append(row[2].rstrip())
        if "Missing" in row[3]: tminList.append(np.nan)
        else:   tminList.append(row[3].rstrip())
        if "Missing" in row[2] or "Missing" in row[3]:  count += 1
    
    # Create PandasDataFrame
    d = {"yyyy": yearList,"month": monthList, "tmax": tmaxList, "tmin": tminList}
    df = pd.DataFrame(d)

    # Create missing max and missing Min dataframes with those columns gone
    missingTMax = df[df['tmax'].isnull()].loc[:,df.columns != 'tmax']
    missingTMin = df[df['tmin'].isnull()].loc[:,df.columns != 'tmin']

    # Remove missing/NaN rows from main dataframe 
    dfNoNull = df.dropna()
    """
        Linear regression Result:                       84.3
        Logistic Regression Result:
                lbfgs, mul                              77.52
                lblin, ovr                              74.84
                lbfgs, ovr                              72.34
                sag, ovr                                10
                new, ovr                                DNR       
        ARD Result:                                     84.46
                normalized                              84.46    
        Boosting values tend to change, will try to record highest or get max
        ADA Result:
            linear Loss:                                85.92
            Square Loss:                                85.32
            Expone Loss:                                85.66
        BAG RESULT:                                     85.58
        ETR Result:                                     86.04

    """

    # convert months to digits
    # Note there is a warning in this as we're trying to set on a copy of a slience from a df
    dfNoNull.loc[:,'month'] = dfNoNull.month.map(months)
    missingTMax.loc[:,'month']= missingTMax.month.map(months)
    missingTMin.loc[:,'month'] = missingTMin.month.map(months)
    # Setting Datasets
    train = dfNoNull
    testTMax = missingTMax
    testTMin = missingTMin

    # Models
    resultLin = LinearRegressionRun(train, testTMax, testTMin)
    resultLog = LogisticRegressionRun(train, testTMax, testTMin)
    resultARD = linARDRun(train, testTMax, testTMin)
    resultADA = ADABoostRegressorRun(train,testTMax, testTMin)
    resultBAG = baggingRegressorRun(train,testTMax, testTMin)
    resultETR = ExtraTreesRegressorRun(train, testTMax, testTMin)
    resultGBR = GradientBoostingRegressorRun(train, testTMax, testTMin)
    resultRFR = RandomForestRegressorRun(train, testTMax, testTMin)

    Listing = []
    # Finetuning GBR
    # i = 0.01
    # constraint = 0.1
    # GrowthRate = 0.01
    # while i < constraint:
    #     Label = "GBR Evaluation lr = " + str(i) + ":"
    #     resultGBR = GradientBoostingRegressorRun(train, testTMax, testTMin, i)
    #     Listing.append((Label, resultGBR))
    #     i += GrowthRate
    
    Listing.append(("Linear Evaluation", resultLin))
    Listing.append(("Logistic Evaluation", resultLog))
    Listing.append(("ARD Evaluation", resultARD))
    Listing.append(("ADA Evaluation", resultADA))
    Listing.append(("BAG Evaluation", resultBAG))
    Listing.append(("ETR Evaluation", resultETR))
    Listing.append(("GBR Evaluation", resultGBR))
    Listing.append(("RFR Evaluation", resultRFR))


    Results = []
    # Run Evaluator
    for item in Listing:
        evalResult = evaluator(item[1])
        Results.append((item[0], evalResult))

    Results.sort(key=lambda model: model[1], reverse = True)
    # Print Evaluations in ranked order
    for item in Results:
        print(item[0], item[1])

    return resultGBR


if __name__ == '__main__':
    # For Debug purposes in warnings
    # warnings.filterwarnings(action='once')
    warnings.filterwarnings('ignore')
    # f = open(os.environ['OUTPUT_PATH'], 'w')
    # n = int(input())
    # Just to get it to pass once
    # input()
    temperatures = []

    # for _ in range(n):
        # temperature_item = str(input())
        # temperatures.append(temperature_item)

    # From reading files
    with open("input.txt", "r") as inputFile:
        next(inputFile)
        next(inputFile)
        for line in inputFile:
            temperatures.append(line)
    result = main(temperatures)

    # f.write('\n'.join(map(str, result)))
    # f.write('\n')
    # f.close()
