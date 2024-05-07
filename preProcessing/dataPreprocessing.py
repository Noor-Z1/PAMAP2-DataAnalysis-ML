from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


class dataPreProcessor():
     
    def __init__(self):
        self.dataFrame=pd.DataFrame()
    
    def initializeDataFrame(self):

        list_of_files = ['PAMAP2_Dataset/Protocol/subject101.dat',
                        'PAMAP2_Dataset/Protocol/subject102.dat',
                        'PAMAP2_Dataset/Protocol/subject103.dat',
                        'PAMAP2_Dataset/Protocol/subject104.dat',
                        'PAMAP2_Dataset/Protocol/subject105.dat',
                        'PAMAP2_Dataset/Protocol/subject106.dat',
                        'PAMAP2_Dataset/Protocol/subject107.dat',
                        'PAMAP2_Dataset/Protocol/subject108.dat',
                        'PAMAP2_Dataset/Protocol/subject109.dat' ]

        subjectID = [1,2,3,4,5,6,7,8,9]

        activityIDdict = {0: 'transient',
                    1: 'lying',
                    2: 'sitting',
                    3: 'standing',
                    4: 'walking',
                    5: 'running',
                    6: 'cycling',
                    7: 'Nordic_walking',
                    9: 'watching_TV',
                    10: 'computer_work',
                    11: 'car driving',
                    12: 'ascending_stairs',
                    13: 'descending_stairs',
                    16: 'vacuum_cleaning',
                    17: 'ironing',
                    18: 'folding_laundry',
                    19: 'house_cleaning',
                    20: 'playing_soccer',
                    24: 'rope_jumping' }

        colNames = ["timestamp", "activityID","heartrate"]

        IMUhand = ['handTemperature', 
                'handAcc16_1', 'handAcc16_2', 'handAcc16_3', 
                'handAcc6_1', 'handAcc6_2', 'handAcc6_3', 
                'handGyro1', 'handGyro2', 'handGyro3', 
                'handMagne1', 'handMagne2', 'handMagne3',
                'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4']

        IMUchest = ['chestTemperature', 
                'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 
                'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3', 
                'chestGyro1', 'chestGyro2', 'chestGyro3', 
                'chestMagne1', 'chestMagne2', 'chestMagne3',
                'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4']

        IMUankle = ['ankleTemperature', 
                'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 
                'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3', 
                'ankleGyro1', 'ankleGyro2', 'ankleGyro3', 
                'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
                'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4']

        columns = colNames + IMUhand + IMUchest + IMUankle  #all columns in one list

        for file in list_of_files:
            procData = pd.read_table(file, header=None, sep='\s+')
            procData.columns = columns
            procData['subject_id'] = int(file[-5])
            self.dataFrame = self.dataFrame._append(procData, ignore_index=True)

        self.dataFrame.reset_index(drop=True, inplace=True)

        
        #subject_data = self.dataFrame[self.dataFrame['subject_id'] == 9]

        # Shuffle the subject data
        #subject_data = subject_data.sample(frac=1)  # Shuffle all rows

        #print(subject_data)

    def dataCleaning(self):
        self.dataFrame = self.dataFrame.drop(['handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4',
                                             'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4',
                                             'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4'],
                                             axis = 1) 
        self.dataFrame = self.dataFrame.drop(self.dataFrame[self.dataFrame.activityID == 0].index)
        self.dataFrame = self.dataFrame.apply(pd.to_numeric, errors = 'ignore') 
        self.dataFrame = self.dataFrame.interpolate()

    def applyPreProcessing(self):
        self.dataFrame.reset_index(drop = True, inplace = True)

        for i in range(0,4):
            self.dataFrame["heartrate"].iloc[i]=100
            #dropping 3D-acceleration data (ms-2), scale: ±6g columns as the readme file recommends considering 3D-acceleration data (ms-2), scale: ±16g data
        columnsToDrop= ["timestamp","handAcc6_1","handAcc6_2","handAcc6_3","chestAcc6_1","chestAcc6_2","chestAcc6_3","ankleAcc6_1","ankleAcc6_2","ankleAcc6_3"]
        self.dataFrame=self.dataFrame.drop(columnsToDrop,axis=1)


        # Create a RobustScaler object
        scaler = RobustScaler()

        # Separate features (excluding potential ID columns like 'subject_id')
        self.dataFrame.iloc[:,1:-1] = scaler.fit_transform(self.dataFrame.iloc[:,1:-1])

        print("heeere now!")
        print(self.dataFrame.head(5))


        """
        features = self.dataFrame.columns[self.dataFrame.columns != 'activityID']  # Assuming 'activityID' is the target
        X = self.dataFrame[features]  # Features for PCA

        # Create a PCA object
        pca = PCA(n_components=X.shape[1])  # Set n_components to match feature count

        # Fit PCA to the data
        pca.fit(X)

        var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

        plt.title("PCA Variance against num of Componmnets")
        plt.ylabel("Variance %")
        plt.xlabel("Number of componments")
        l = plt.axhline(94, color="red")

        plt.plot(var1)
        plt.grid()
        plt.show()
        """

        dataCollectionCopy=self.dataFrame.copy()


        tempColumnsToDrop=["activityID","subject_id"]
        dataCollectionTransformed=self.dataFrame.drop(tempColumnsToDrop,axis=1).values

        pca = PCA(n_components=25)
        newTransformedDataFrame=pca.fit_transform(dataCollectionTransformed)
        newTransformedDataFrame = pd.DataFrame(newTransformedDataFrame)

        combined_df = pd.DataFrame({"subject_id": dataCollectionCopy["subject_id"], "activityID": dataCollectionCopy["activityID"]})

        for i in range(newTransformedDataFrame.shape[1]):
            combined_df[f"PC{i+1}"] = newTransformedDataFrame.iloc[:, i]


        #print(combined_df.head())
        #print(combined_df.describe())

        return combined_df

     
