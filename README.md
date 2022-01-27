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
* Testing for Stationarity 
* Making a Time Series Stationary
* Seasonal Decomposition of a Time Series
* Selection of Non-seasonal and Seasonal Orders - 
  a) Manual selection of orders
  b) Automated selection of orders - using Pmdarima library (auto_arima)
* Splitting the dataset into Train and Test subsets
* Model Selection for Stock Predictions - ARIMA, Seasonal ARIMA (SARIMA) Model, Auto ARIMA, Prophet
* Time Series Forecasting - 

    * Generating In-Sample (One-Step Ahead) Predictions for Amazon stock data
    * Generating Dynamic Forecasts 
    * Making Out-of-Sample Forecasts for Amazon stocks 



### Sources of Data -

I retrieved Amazon Stock data from Kaggle Public Dataset - [FAANG- Complete Stock Data](https://www.kaggle.com/aayushmishra1512/faang-complete-stock-data). The data was available as a csv file (*Amazon.csv*). The data contained all the relevant data such as Opening price, Closing price, Volumes of stock traded about the Amazon stock from 1997 to 2020.


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
* **Stationarity** of time series means that the distribution of the data in here, doesn't change with time i.e., the series must have zero trend, constant variance & constant autocorrelation. Absence, of any of these 3 conditions makes it **Non-Stationary**. 
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

 



### Testing for Stationarity - 

Time Series analysis only works with stationary data therefore, we need to determine if our Amazon stock data is stationary or not. A dataset is stationary if its statistical properties like mean, variance, and autocorrelation do not change over time. We need to make the time-series stationary before fitting any model.

Testing for Stationarity can be done using one or more of the following methods:-
* **Rolling Statistics** - This mathod is a visualization technique where we plot the mean and standard deviation of our time series to determine stationarity. A series becomes stationary if both the mean and standard deviation are flat lines (constant mean and constant variance).  We needed to check this for our Amazon time series as well. From the plots below, we found that there was an increasing mean and standard deviation, indicating that our series was not stationary.

![image](https://user-images.githubusercontent.com/50409210/151359702-8330f044-5174-4d6a-8fa2-bc2722c12f5d.png)

However, to be more sure about stationarity or not for our Amazon time series we used the ADF test method as well as explained below.

* **Augmented Dicky-Fuller (ADF) Test** - The ADF Test is one of the most common tests for stationarity and is based on the concept of unit root. Null Hypothesis (H0) for this test states that the time series is non-stationary due to trend. If the null hypothesis is not rejected, the series is said to be non-stationary. The result object of the ADF test is a tuple with following key elements:

           * Zeroth element - **Test statistic** (the more negative this number is, the more likely that data is stationary) 
           * Next element - **p-value** (if p-value < 0.05, we reject the H0 and assume our time series to be stationary)
           * Last element - **a Python Dictionary with critical values** of the test statistic equating to different p-values

By performing ADF test for Amazon stock data we ran a test for statistical significance to determine whether it was staionary or not, with different levels of confidence and got the following outputs.

![image](https://user-images.githubusercontent.com/50409210/151357493-7c6aa93b-99ab-4495-ba45-5d8d0be35a6a.png)

As we can see from the test results that our Test statistic was a positive value and our p-value > 0.05.  Additionally, the test statistics exceeded the critical values hence, the data was nonlinear and we did not reject the Null Hypothesis (H0). Thus, by using both the rolling statistic plots and ADF test results we determined that our **Amazon stock price data was Non-Stationary**. Next, we had to work towards making our time series stationary before applying any Machine Learning modelling techniques.


 ### Making a Time Series Stationary  - 
 
 As we discussed earlier, to proceed with any time series analysis using models, we needed to stationarize our Amazon stock time series. 
 
 If the time series is non-stationary the below methods can be used to make them stationary:-
 
 * **De-trending the time series** - This method removes the underlying trend in the time series by standardizing it. It subtracts the mean and divides the result by the standard deviation of the data sample. This has the effect of transforming the data to have mean of zero, or centered, with a standard deviation of 1. Then the ADF test is performed on the de-trended time series to confirm the results.

The ADF test results on de-trended stock data for Amazon below, shows a p=value < 0.05 and also a negative Test statistic, which means that the Amazon time series has become stationary now.

![image](https://user-images.githubusercontent.com/50409210/151385463-0e6bc76f-8005-4ace-9cd2-e7bbdaf22a79.png)

In addition to this, on plotting the rolling statistics for the de-trended Amazon stock (see below) further shows that the time series has become stationary. This is indicated by the relative smoothness of the rolling mean and rolling standard deviation compared to the original De-trended data.

![image](https://user-images.githubusercontent.com/50409210/151386719-78eaddc9-40dd-40b3-8fb7-564a240a8b5f.png)


 * **Differencing the time series** - Another widely used method to make a time series stationary is Differencing. This method removes the underlying seasonal or cyclical patterns in the time series thereby removing the series' dependence on time also-called temporal dependence.

Following are some key concepts regarding Differencing:-

     * Here, from each value in a time series we subtract the previous value. Differencing can be performed manually or using the Pandas' ***.diff() function***. The resulting missing or NaN value at the start is removed using the ***.dropna() method***. Using Pandas function helps in maintaining the date-time information for the differenced series.
     * When applied for first time the process of differencing is called the **First Order of Differencing**. 
     * Sometimes, the differencing might be applied more than once if the time series still has some temporal dependence, then it is called **Second Order of Differencing** and so on.
     * The value of d, therefore, is the minimum number of differencing needed to make the series stationary. And if the time series is already stationary, then d = 0.
     * We need to be careful not to **over-difference a time series**, because, an over-differenced series may still be stationary but would affect the model parameters.
     * **Optimal order for Differencing** - minimum differencing order required to get a near-stationary series with a defined mean and for which the ACF plot quickly reaches zero.
     
In order to get the right order of differencing for Amazon time series we took the **first order of differencing**, performed the ADF test on the differenced series and then plotted the correlation plots (ACF and PACF) for the differenced series.

![image](https://user-images.githubusercontent.com/50409210/151394410-11eeacb5-58f4-44e4-b6a2-c0d020b8d7cd.png)

As an experiment, we also took a **second order of differencing** for our Amazon time series and repeated the further steps here as well. This was ust to analyze the impact of differencing a series twice.

![image](https://user-images.githubusercontent.com/50409210/151395152-ae6b1ef4-ce3a-46ec-89ea-6dd72c1bf409.png)

Both the 1st Order and 2nd Order of Differencing yielded very small p-values and also very negative test statistics. For the above plots of ACF and PACF, we found that the time series reaches stationarity with two orders of differencing. But on looking at the PACF plot for the 2nd differencing the lag goes into the negative zone very quickly. This indicates that the Amazon stock series might get over-differenced with 2nd order differencing. So, to avoid over-differencing, we restricted the **Order of Differencing** for our dataset to **d = 1**.


 ### Seasonal Decomposition of a Time Series - 
 
A seasonal time series generally has some predictable patterns that repeat regularly (i.e., after any length of time). 

* Every time series is a combination of 3 parts - **Trend, Seasonality and Residual**. Separating a time series into its 3 components is called **decomposition of time series**. * Automatic decomposition of a time series is available in ***statsmodel library***, using the ***seasonal_decompose() function***. It requires one to specify whether the model is additive or multiplicative. The result object contains arrays to access four pieces of data from the decomposition.

* **Additive vs multiplicative seasonality** - These are the two methods to analyze seasonality of a Time Series:- 

  a) ***Additive*** - Seasonal pattern just adds or has a linear behaviour i.e., where changes over time are consistently made by the same amount.
  
  ![image](https://user-images.githubusercontent.com/50409210/151403195-6bf025e3-bb0e-4e02-bf0f-c024da996b57.png)
  
  b) ***Multiplicative*** - Trend and seasonal components are multiplied and then added to the error component and behaviour is non-linear (exponential or quadratic). The amplitude of the seasonal oscillations get larger as the data trends up or get smaller as it trends down. To deal with this we take the log transform of the data before modelling it.

![image](https://user-images.githubusercontent.com/50409210/151403945-c5a0ed42-a2bc-4f11-aac9-ae6efe1078e8.png)

We decomposed our Amazon time series using statsmodel's seasonal_decompose() function and setting the ***period parameter*** to 12 (number of data points in each repeated cycle). We also specified the ***model paramter*** as ***multiplicative***. It returned a decompose-results object. We then used the ***plot() method*** of this object to plot the components of our decomposed Amazon time series. 

![image](https://user-images.githubusercontent.com/50409210/151399016-7fb9d988-14b4-4789-82f7-fa575512f10d.png)

We could see from the decomposition plot that our Amazon stock data has both trend and seasonality. The pattern of trend is positive and increasing in nature however, the seasonality pattern does not appear to be very clear from this decomposition plot.


 
 


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






