import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

class pipeline():
    def __init__(self,data):
        self.y = data['price']
        self.X = data.drop(['price'],axis=1)
        self.categorical = ['bike_name','city','brand']
        self.numerical = ['kms_driven','owner','age','power']
        
    def show(self):
        print(self.X.head())

    def TrainPipe(self):
        self.saveData()
        self.encodeTrain()
        self.Train()
        self.saveMethods()

    def encodeTrain(self):
        self.X = self.X.replace(['First Owner','Second Owner','Third Owner','Fourth Owner Or More'], [1, 2, 3, 4])
        OHE = OneHotEncoder()
        Xmini = pd.DataFrame(OHE.fit_transform(self.X[self.categorical]).toarray())
        self.encoder = OHE
        self.X = pd.concat([self.X[self.numerical], Xmini], axis = 1)

    def Train(self):
        print('Training Model...')
        model = RandomForestRegressor(n_estimators=200)
        model.fit(self.X,self.y)
        self.model = model
        print('Model Trained Succesfully')
        
    def saveData(self):
        self.X['bike_name'].to_csv('bike_name.csv',index=False)
        self.X['city'].to_csv('city.csv',index=False)
        self.X['brand'].to_csv('brand.csv',index=False)
        print('Categories Saved Succesfully!')
        
    def saveMethods(self):
        dump(self.model, 'model.joblib')
        print('Model Dumped Succesfully!')
        dump(self.encoder, 'encoder.joblib')
        print('Encoder Dumped Succesfully!')


if __name__ == '__main__':
    d = pd.read_csv('Used_Bikes.csv')
    
    Pipe = pipeline(d)
    Pipe.TrainPipe()