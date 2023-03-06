import requests as rq
import shutil as sh
import os
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


with open("dataset.zip", 'wb') as file:
  file.write(rq.get("https://drive.google.com/uc?export=download&id=19Jdeh7zcbIq6SLQ3sLsGoSXpgccWBrtN").content)
sh.unpack_archive("dataset.zip", "./", "zip")
os.remove("./dataset.zip")
for dir_name in ["generation", "weather"]:
  if not os.path.exists(dir_name):
      os.makedirs(dir_name)
      print("Directory created successfully.")
if not os.path.isfile("/content/generation/Plant_1_Generation_Data.csv"):
  sh.move("/content/Plant_1_Generation_Data.csv", "generation")
if not os.path.isfile("/content/generation/Plant_2_Generation_Data.csv"):
  sh.move("/content/Plant_2_Generation_Data.csv", "generation")
if not os.path.isfile("/content/weather/Plant_1_Weather_Sensor_Data.csv"):
  sh.move("/content/Plant_1_Weather_Sensor_Data.csv", "weather")
if not os.path.isfile("/content/weather/Plant_2_Weather_Sensor_Data.csv"):
  sh.move("/content/Plant_2_Weather_Sensor_Data.csv", "weather")


generation = pd.read_csv('/content/generation/Plant_1_Generation_Data.csv')
weather = pd.read_csv('/content/weather/Plant_1_Weather_Sensor_Data.csv')
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
data.head(5)



# sns.displot(data=data, x="AMBIENT_TEMPERATURE", kde=True, bins = 100, height = 5, aspect = 3.5);
# sns.displot(data=data, x="MODULE_TEMPERATURE", kde=True, bins = 100, height = 5, aspect = 3.5);

pd.plotting.scatter_matrix(data, figsize=(15,15))
plt.show()

corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')


# X = data[['DAILY_YIELD','TOTAL_YIELD','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE','IRRADIATION','DC_POWER']]
X = data[['HOURS', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']]
Y = data['AC_POWER']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2)
LR = LinearRegression()
LR.fit(X_train,Y_train)

Y_predLR = LR.predict(X_test)
r_score = LR.score(X_test, Y_test)
print("R-squared score: ", r_score)

Y_predLR = LR.predict([[11, 28, 40]])
print(Y_predLR)

