import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from joblib import load
import streamlit as st
from PIL import Image

class testPipeline():
    def __init__(self,Cat,Num):
        self.categorical = ['bike_name','city','brand']
        self.numerical = ['kms_driven','owner','age','power']
        self.Cat = Cat
        self.Num = Num
        self.encoder = load('encoder.joblib')
        self.model = load('model.joblib')


    def seeFuture(self):
        a = pd.DataFrame(self.encoder.transform(pd.DataFrame(np.array(self.Cat).reshape(-1,len(self.Cat)))).toarray())
        a = pd.concat([pd.DataFrame(np.array(self.Num).reshape(-1,len(self.Num))),a],axis=1)
        st.success('You can expect around '+str(self.model.predict(a)[0])+'₹')
        st.write('Doesn\'t seem Right? Try Entering Correct Details!')
        print('Prediction : ',self.model.predict(a))

class UI():
    def __init__(self):
        self.Cat = None
        self.Num = None
    
    @st.cache()
    def getData(self):
        with open('bike_name.csv') as f:
            content = f.readlines()
        content = [x.strip() for x in content] 
        bikes = sorted(list(set(content[1:])))

        with open('brand.csv') as f:
            content = f.readlines()
        content = [x.strip() for x in content] 
        brands = sorted(list(set(content[1:])))

        with open('city.csv') as f:
            content = f.readlines()
        content = [x.strip() for x in content] 
        cities = sorted(list(set(content[1:])))

        print('Data Extracted Succesfully')

        return bikes,brands,cities
        
    def build(self):
        self.bikes,self.brands,self.cities = self.getData()
        mVal = st.sidebar.selectbox('Select Option!',['Price Prediction','About'])

        st.sidebar.write('You can browse code on my [Github!](https://github.com/MautKaFarishta/Used_Bike_Price_Prediction)')

        mail = '📩 omkhilariindia@gmail.com'
        st.sidebar.markdown('For any feedback or queries contact me here!')
        st.sidebar.write(mail)
        
        if mVal == 'Price Prediction':
            st.title('Used Bike Price Predction')
            self.getBikeInfo()
        if mVal == 'About':
            st.title('About Used Bike Price Predction')
            self.showAbout()

    def showAbout(self):
        st.markdown('The dataset used for project is Used _\'Bikes Prices in India Dataset of ~32000 used Bike data scraped from www.droom.in\'_')
        link = 'https://www.kaggle.com/saisaathvik/used-bikes-prices-in-india'
        st.markdown(link, unsafe_allow_html=True)

        st.markdown('**Requirements** of \'Used Bike Price Prediction\' are specified below:')
        st.write('➡ Pandas')
        st.write('➡ Numpy')
        st.write('➡ Sklearn')
        st.write('➡ Joblib')
        st.write('➡ Streamlit')

        st.markdown('In **\'Used Bike Price Prediction\'** I have tried to leaverage some of the popular Data Science and Machine Learning techniques as part of skill enhancement.')
        st.markdown('The flow of the _ETL Pipeline_ is Explained below.')

        flow = Image.open('FlowDiagram.png')

        st.image(flow,caption='Flow Chart of Pipeline')


    def getBikeInfo(self):
        brand = st.selectbox('Select Brand',self.brands)
        brand_bikes = [x for x in self.bikes if x.startswith(brand)]
        bike = st.selectbox('Select Bike',brand_bikes)

        power = st.number_input('Power in cc?',100)

        city = st.selectbox('Select City',self.cities)

        kms = st.number_input('How many Kms Driven?',100)
        age = st.number_input('Age in Years?',1)

        owner = st.selectbox('Previous Owners if any',[1,2,3,4])

        testCat = [bike,city,brand]
        testNum = [kms,owner,age,power]

        if st.button('Predict Price of Bike.'):
            test = testPipeline(testCat,testNum)
            test.seeFuture()

if __name__ == '__main__':

    # testCat = ['Royal Enfield Classic 500cc','Delhi','Royal Enfield'] #bike_name,city,company
    # testNum = [11000,1,4,500] #kms_driven,owner,age,power

    main = UI()
    main.build()

