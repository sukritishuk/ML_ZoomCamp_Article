## MLZoomCamp Article

# **Amazon Stock Price Prediction using Time Series Forecasting**

## Introduction:

Time has always been an important factor in statistical analysis because of its quantifiable and ever-changing nature and its impact on our daily lives. Today, it has become all the more important in both financial & non-financial contexts with the emergence of real-time Data and real-time Analytics. 


In machine learning, Time Series Analsis & Forecasting is among the most applied in Data Science in real-world scenarios like financial analysis, demand forecasting, production planning etc. Through this article, I wanted to discuss key concepts, techniques and methods used to perform basic **Time Series Analysis & Forecasting in Python**. Time series analysis in Python uses timestamps, time deltas, and time periods for plotting time series data. My goal was to explore these areas and create a prediction service (model) to make accurate forecasts for Amazon stock data. 

Steps for making Amazon Stock Forecasting -

* Loading & Studying the Amazon Stock Data 
* Data Cleaning and Formatting - Changing data types, Missing Values Detection, Feature Engineering
* Exploratory Data Analysis (EDA) of Amazon Time Series - Correlation, Moving Averages, Percentage Change in Stocks, Resampling 
* Slicing Amazon Stock Data for Forecasting 
* Testing for Stationarity & Making a time series Stationary
* Seasonal Decomposition of Time Series
* Selection of Non-seasonal and Seasonal Orders - 
  a) Manual selection of orders
  b) Automated selection of orders - using Pmdarima library (auto_arima)
* Splitting the dataset into Train and Test subsets
* Model Selection for Stock Predictions - ARIMA, Seasonal ARIMA (SARIMA) Model, Auto ARIMA, Prophet
* Time Series Forecasting - 
  a) Generating In-Sample (One-Step Ahead) Predictions for Amazon stock data
  b) Generating Dynamic Forecasts 
  c) Making Out-of-Sample Forecasts for Amazon stocks 



### Sources of Data -

I retrieved Amazon Stock data from Kaggle Public Dataset - [FAANG- Complete Stock Data](https://www.kaggle.com/aayushmishra1512/faang-complete-stock-data). The ata was available as a csv file (*Amazon.csv*). The data contained all the relevant data such as Opening price, Closing price, Volumes of stock traded about the Amazon stock from 1997 to 2020.


### Data Preparation - 

Firstly, I retrieved the time series data for Amazon company stock from Kaggle. I cleaned and pre-processed it, using Python in-built functions and libraries such as Pandas, Numpy. Then, Matplotlib and Seaborn libraries were used to perform EDA analysis on Amazon time series data, make visualizations and draw insights from the different features. 

![image](https://user-images.githubusercontent.com/50409210/151208664-a286b590-89b6-4cdf-b42a-fb7760d2ccce.png)

There are multiple feature variables in this dataset like below, apart from one time-based feature Date. 
* Open and Close - starting and final price at which the stock is traded on a particular day.
* High and Low - maximum and minimum trading price of the stock for the day. 
* Close - last price at which the stock trades before markets closed for the day.
* Adjusted Close (Adj Close) - Closing price for the stock on a day adjusted for corporate actions like stock-splits or dividends.
* Volume - total number of shares traded (bought or sold) for Amazon during a day.

An important thing to note is that the market remains closed on weekends and any public holidays therefore, we may notice some date values to be missing in the dataset. To perform time Series Forecasting I used only the Date and Adjusted Close. The data was then split it into train and test subsets to verify our predictions from models.
I used **Autoregressive Moving Average (ARMA)**, **Autoregressive Integrated Moving Average (ARIMA)** and **Facebook Prophet** models to make predictions. 


Before we move on to Time Series Analysis and Forecasting techniques let me explain some key concepts about time series analysis and time-related data in general.

Key Concepts about Time-specific Data -
* **Time series** is a set of data points that occur over a span or succesive periods of time like Years, Months, Weeks, Days, Horus, Minutes, and Seconds.
* Time series differs from **Cross-Sectional Data** as the latter considers only a single point in time while the former looks at a series or repeated samples of data over a time period. 
* **TimeStamp** is a Python equivalent for Date and Time.
* **Trends** involve a gradual increase or decrease in time series data like a rise in death rates due to outbreak of a deadly disease.
* **Seasonality** occurs in time series when a trend is periodically repeating itself e.g., increase in sale of home decorations around Christmas season. 
* **Stationarity Time Series** means that the distribution of the data in here, doesn't change with time i.e., the series must have zero trend, constant variance & constant autocorrelation. Absence, of any of these 3 conditions makes it **Non-Stationary**. 
* **Rolling or Moving Aggregations** on data involves performing aggregations like average or sum of data, using a moving window over the entire dataset.
* **Resampling** of time series data is common to ease anaysis and improve visualizations. It involves converting data to a higher or lower frequency like daily data to monthly or yearly.


Key Concepts about Time Series Analysis -
* **Time Series Analysis** involves studying what changes are made in the economic asset or concerned variable during a period of time (e.g. price of gold, birth rates, rainfall levels etc.). It also studies the influence of time over other variables in data.
* Characteristics in time series data like **seasonality, structural breaks, trends and cyclicity**, often differ from other types of data therefore, require unique set of tools and techniques for analysis.
* A time series includes three systematic components: **Level, Trend, and Seasonality**, as well as one non-systematic component termed **Noise**.
  a) Level - average value in the series
  b) Trend - increasing or falling value in the series
  c) Seasonality - short-term recurring movements in series
  d) Noise - random variance in series
* Time Series Analysis & modelling of time-specific data can be done only on Stationary time series.  
* Time series analysis is done only on **continuous variables** and includes Trend Analysis, Forecasting for Cyclical fluctuations or Seasonal patterns. **Trend Analysis** aims to analyze the movements in historical data using visualizations like line plots, bar charts while, **Forecasting** tries to predict the future values with available present & past data.

 



### Testing for Stationarity & Making a Time Series Stationary - 

Time Series analysis only works with stationary data therefore, we need to determine if our Amazon stock data is stationary or not. A dataset is stationary if its statistical properties like mean, variance, and autocorrelation do not change over time. We need to make the time-series stationary before fitting any model.

Testing for Stationarity can be done using one or more of the following methods:-
* Visualizing Rolling Statistics 
* Performing Augmented Dicky-Fuller (ADF) Test


 Making a Time Series Stationary - If the time series is non-stationary the below methods can be used to make them stationary:-
 * De-trending the time series
 * Differencing the time series
 


### Selection of Non-seasonal and Seasonal Orders -

There are three important parameters in ARIMA:

* p (past values used for forecasting the next value)
* q (past forecast errors used to predict the future values)
* d (order of differencing)
 



### Model Selection for Stock Predictions -

Broadly, Time-Series models like **Autoregressive (AR), Integrated (I), Moving Average(MA)** are combined to form **Autoregressive Moving Average (ARMA), and Autoregressive Integrated Moving Average (ARIMA)** models. Some deep learning-based models include **Long-short term memory(LSTM)**.


Autoregressive Integrated Moving Average (ARIMA) - 
* Very popular statistical method for time series forecasting and capable of predicting short-term share market movements.
* Considers historical values to predict future values. 


Prophet - 
* Designed and pioneered by Facebook, a third-party time series forecasting library.
* Requires almost no data preprocessing, very simple to implement. 
* Library is capable of handling stationarity within the data and seasonality related components in data.
* The input for Prophet is a dataframe with two columns: date and target variable column (columns named as - **ds** and **y**).
* [Prophet](https://github.com/facebook/prophet) enables the use of simple parameters to fine-tune the model like specifying holidays, daily seasonality etc. 






