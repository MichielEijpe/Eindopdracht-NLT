import requests as rq
import shutil as sh
import os
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk

# Dataset opsplitsen
from sklearn.model_selection import train_test_split

# R2 Score
from sklearn.metrics import r2_score

# Algoritmes
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

st.set_page_config(
   page_title="Eindopdracht",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)
st.set_option('deprecation.showPyplotGlobalUse', False)


with open("dataset.zip", 'wb') as file:
  file.write(rq.get("https://drive.google.com/uc?export=download&id=19Jdeh7zcbIq6SLQ3sLsGoSXpgccWBrtN").content)
sh.unpack_archive("dataset.zip", "./", "zip")

os.remove("./dataset.zip")

generation = pd.read_csv('./Plant_1_Generation_Data.csv')
weather = pd.read_csv('./Plant_1_Weather_Sensor_Data.csv')
generation['DATE_TIME'] = pd.to_datetime(generation['DATE_TIME'],format = '%d-%m-%Y %H:%M')
weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
data = pd.merge(generation.drop(columns = ['PLANT_ID']), weather.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')

# data["DATE"] = pd.to_datetime(data["DATE_TIME"]).dt.date
data["TIME"] = pd.to_datetime(data["DATE_TIME"]).dt.time
data['DAY'] = pd.to_datetime(data['DATE_TIME']).dt.day
data['MONTH'] = pd.to_datetime(data['DATE_TIME']).dt.month
data['WEEK'] = pd.to_datetime(data['DATE_TIME']).dt.week
data['HOURS'] = pd.to_datetime(data['TIME'],format='%H:%M:%S').dt.hour
data['MINUTES'] = pd.to_datetime(data['TIME'],format='%H:%M:%S').dt.minute
#data['TOTAL MINUTES PASS'] = data['MINUTES'] + data['HOURS']*60
# data["DATE_STRING"] = data["DATE"].astype(str)
# data["HOURS"] = data["HOURS"].astype(str)
#data["TIME"] = data["TIME"].astype(str)
st.title("NLT Eindopdracht - Opbrengste zonnepaneel voorspellen")
st.caption("Michiel, Thijs, Beer, Gijs")
st.header("Datasets")

st.subheader("Opbrengsten + Weergegevens")
st.dataframe(data)

st.header("Visualisatie")
tab1, tab2= st.tabs(["Scatter Matrix", "Correlatie"])
with tab1:
    pd.plotting.scatter_matrix(data.drop(columns = ['DAY', 'MONTH', 'WEEK', 'HOURS', 'MINUTES']), figsize=(15,15))
    fig = plt.show()
    st.pyplot(fig)
with tab2:
    plt.figure(figsize=(15,10))
    correlation_matrix = data.drop(columns = ['DAY', 'MONTH', 'WEEK', 'HOURS', 'MINUTES']).corr()
    sns.heatmap(data=correlation_matrix, annot=True)
    fig = plt.show()
    st.pyplot(fig)
# X = data[['DAILY_YIELD','TOTAL_YIELD','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION','DC_POWER']]
X = data[['HOURS', 'AMBIENT_TEMPERATURE']]
Y = data['AC_POWER']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2)

tab1, tab2, tab3 = st.tabs(["Linear Regression", "Correlatie", "Decision Tree Regressor"])
with tab1:
  LR = LinearRegression()
  LR.fit(X_train,Y_train)
  Y_predLR = LR.predict(X_test)
  LR_train, LR_test, LR_score = LR.score(X_train , Y_train), LR.score(X_test , Y_test), round(r2_score(Y_predLR,Y_test) * 100, 2)

  st.write("R2 Score: ", LR_score,"%")
  st.write(f"Training Score: ", LR_train)
  st.write(f"Test Score: ", LR_test)

with tab2:
  RFR = RandomForestRegressor()
  RFR.fit(X_train,Y_train)
  Y_predRFR = RFR.predict(X_test)
  RFR_train, RFR_test, RFR_score = RFR.score(X_train , Y_train), RFR.score(X_test , Y_test), round(r2_score(Y_predRFR,Y_test) * 100, 2)

  st.write("R2 Score: ", RFR_score,"%")
  st.write(f"Training Score: ", RFR_train)
  st.write(f"Test Score: ", RFR_test)

with tab3:
  DTR = DecisionTreeRegressor()
  DTR.fit(X_train,Y_train)
  Y_predDTR = DTR.predict(X_test)
  DTR_train, DTR_test, DTR_score = DTR.score(X_train , Y_train), DTR.score(X_test , Y_test), round(r2_score(Y_predDTR,Y_test) * 100, 2)

  st.write("R2 Score: ", DTR_score,"%")
  st.write(f"Training Score: ", DTR_train)
  st.write(f"Test Score: ", DTR_test)