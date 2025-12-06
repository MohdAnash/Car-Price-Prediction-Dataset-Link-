# Car-Price-Prediction-Dataset-Link-
ChatGPT said:  The Car Price Prediction dataset contains ~8,128 used-car records with features like name, year, selling_price, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats. Cars range from 1983–2020. Useful for ML regression tasks to predict car prices based on vehicle attributes.

#Importing the dependence :

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

#Data Collection and Processing
# loading the data from csv file to pandas dataframe
car_dataset = pd.read_csv('/content/car data.csv')

# inspecting the first 5 rows of the dataframe
car_dataset.head()

# checking the number of rows and columns
car_dataset.shape

# getting some information about the dataset
car_dataset.info()

# checking the number of missing values
car_dataset.isnull().sum()

# checking the distribution of categorical data
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())

# encoding "Fuel_Type" Column
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

# encoding "Seller_Type" Column
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

# encoding "Transmission" Column
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

car_dataset.head()

X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y = car_dataset['Selling_Price']

print(X)

print(Y)

#Spliting Traning and Test Data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)

#Model Traning
  1- Linear Regression

  # loading the linear regression model
lin_reg_model = LinearRegression()

lin_reg_model.fit(X_train,Y_train)

# Model Evalution

# prediction on Training data
training_data_prediction = lin_reg_model.predict(X_train)

# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

#Visualize the actual price and Predicated price

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()
<Figure size 640x480 with 1 Axes><img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/30eeb385-ebf6-4b2b-b40d-d206d7931d32" />

# prediction on Training data
test_data_prediction = lin_reg_model.predict(X_test)

# R squared Error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()

<Figure size 640x480 with 1 Axes><img width="573" height="455" alt="image" src="https://github.com/user-attachments/assets/50ed1b86-5571-4fe4-88cf-858dfec9e877" />

# loading the linear regression model
lass_reg_model = Lasso()

lass_reg_model.fit(X_train,Y_train)

#Modern Evaluation
# prediction on Training data
training_data_prediction = lass_reg_model.predict(X_train)

# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

#Visulazie the actual price and Predicate price

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()

<Figure size 640x480 with 1 Axes><img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/c6c27057-7e67-4617-a0ac-e06b8801838a" />

# prediction on Training data
test_data_prediction = lass_reg_model.predict(X_test)

# R squared Error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()

<Figure size 640x480 with 1 Axes><img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/dc0b1911-181d-4e2f-bf03-c4ee1e9697c0" />



