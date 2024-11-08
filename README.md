# EX.NO.09        A project on Time series analysis on Electric Production Data using ARIMA model 
### Date: 
### Name:Koduru Sanath Kumar Reddy 
### Register no: 212221240024

### AIM:
To Create a project on Time series analysis on Electric Production Data using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # This line is added to import the necessary plotting functions

data = pd.read_csv("Electric_Production.csv")
data.head()
data.plot(figsize=(10,5))
plt.title("Electric Production Over Time")
plt.show()

def adf_test(series):
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    return result[1] < 0.05  # True if stationary #Fixed: Indentation corrected to align with function body
is_stationary = adf_test(data["IPG2211A2N"])

if not is_stationary:
    data_diff = data["IPG2211A2N"].diff().dropna()
    plt.plot(data_diff)
    plt.title("Differenced Electric Production")
    plt.show()
else:
    data_diff = data["Electric_Production"]

plot_acf(data_diff, lags=20) # Assuming 'plot_acf' is imported from statsmodels.graphics.tsaplots
plt.show()

plot_pacf(data_diff, lags=20) # Assuming 'plot_pacf' is imported from statsmodels.graphics.tsaplots
plt.show()

p, d, q = 1, 1, 1  # example values; adjust based on plots

model = ARIMA(data["IPG2211A2N"], order=(p, d, q))
fitted_model = model.fit()
print(fitted_model.summary())

forecast_steps = 12  # Number of months to forecast
forecast = fitted_model.forecast(steps=forecast_steps)

last_date = pd.to_datetime(data['DATE'].iloc[-1])  # Convert to datetime if necessary
forecast_index = pd.date_range(last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='MS') 

plt.plot(data["IPG2211A2N"], label="Historical Data") #Fixed: Plot data["IPG2211A2N"]
plt.plot(forecast_index, forecast, label="Forecast", color='orange')
plt.legend()
plt.title("Electric Production Forecast")
plt.show()
~~~
### OUTPUT:
### Electric Production Over Time

<img width="688" alt="image" src="https://github.com/user-attachments/assets/34e77938-12aa-4f74-848e-1b68f8476465">


### Differenced Electric Production
<img width="462" alt="image" src="https://github.com/user-attachments/assets/883014e9-9c81-40e9-8bd7-aa433b483ec6">


### AutoCorrelation

<img width="462" alt="image" src="https://github.com/user-attachments/assets/189532e8-9dd8-4314-8667-66937de7e9ee">

### Partial Autocorrelation
<img width="462" alt="image" src="https://github.com/user-attachments/assets/4a66781e-92db-4a72-a331-603f9268cd3e">

### Model Results
<img width="608" alt="image" src="https://github.com/user-attachments/assets/069355a3-1f2e-472e-8e4c-e3e1fb1da8fd">



### Electric Production Forecast


<img width="480" alt="image" src="https://github.com/user-attachments/assets/9380dffd-5bbb-487a-929f-b3dd1efacf69">




### RESULT:
Thus the program run successfully based on the ARIMA model using python.
