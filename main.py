from preProcessing.dataPreprocessing import dataPreProcessor
from models.naiveBayes import NaiveBayes
from models.knn import KNN
from models.svm import SVM
from models.decisionTree import DecisionTree
from models.randomForest import RandomForest
import pandas as pd


if __name__ == "__main__":
    dataFrame=dataPreProcessor()
    dataFrame.initializeDataFrame()
    dataFrame.dataCleaning()
    dataFrameModified=dataFrame.applyPreProcessing()
    naiveBayes=NaiveBayes(dataFrameModified)
    #knn=KNN(dataFrameModified)
    #svm=SVM(dataFrameModified)
    #decisionTree=DecisionTree(dataFrameModified)
    
    #randomForest=RandomForest(dataFrameModified)

    
