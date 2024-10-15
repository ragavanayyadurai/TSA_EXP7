# Ex.No: 07 AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = '/content/Summer_olympic_Medals.csv'  # Ensure the path is correct
data = pd.read_csv(file_path)

# Inspect the first few rows and columns
print("First few rows of the dataset:")
print(data.head())
print("\nColumn names in the dataset:")
print(data.columns)

# Assuming column names are "Gold", "Silver", "Bronze":
data['Total_Medals'] = data['Gold'] + data['Silver'] + data['Bronze']

# Convert the 'Year' column to datetime format, note the column name is 'Year', not 'year'
data['Year'] = pd.to_datetime(data['Year'], format='%Y')

# Set 'Year' as the index, note the column name is 'Year', not 'year'
data.set_index('Year', inplace=True)

# Group by year and sum the total medals
annual_medals = data.resample('Y')['Total_Medals'].sum()

# Check for stationarity using Augmented Dickey-Fuller test
result = adfuller(annual_medals.dropna())
print(f'\nADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
if result[1] < 0.05:
    print("The data is stationary.")
else:
    print("The data is non-stationary.")

# Split data into training and testing sets
train_size = int(len(annual_medals) * 0.8)
train, test = annual_medals[:train_size], annual_medals[train_size:]

# Plot ACF and PACF
fig, ax = plt.subplots(2, figsize=(10, 6))
plot_acf(train.dropna(), ax=ax[0], title='Autocorrelation Function (ACF)')
plot_pacf(train.dropna(), ax=ax[1], title='Partial Autocorrelation Function (PACF)')
plt.show()

# Fit the AutoRegressive model
ar_model = AutoReg(train.dropna(), lags=1).fit()

# Make predictions
ar_pred = ar_model.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Plot the test data vs predictions
plt.figure(figsize=(10, 4))
plt.plot(test, label='Test Data')
plt.plot(ar_pred, label='AR Model Prediction', color='red')
plt.title('AR Model Prediction vs Test Data')
plt.xlabel('Time')
plt.ylabel('Total Medals')
plt.legend()
plt.show()

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(test, ar_pred)  # Assuming you want to calculate MSE
print(f'Mean Squared Error: {mse}')

```
### OUTPUT:

Augmented Dickey-Fuller test

![image](https://github.com/user-attachments/assets/f56ba012-67b1-4421-ba0a-bd8e3b632669)

PACF - ACF

![image](https://github.com/user-attachments/assets/a2c333f0-7a3c-41c8-8107-8e1bf7516d0b)

Mean Squared Error

![image](https://github.com/user-attachments/assets/406f010c-e10f-4c68-8b27-3c8b1065fa97)

PREDICTION
![image](https://github.com/user-attachments/assets/2e7295bd-4ed5-4e5e-8100-13db498e53c5)

### RESULT:
Thus we have successfully implemented the auto regression function using python.
