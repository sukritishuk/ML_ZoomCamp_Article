## ML_ZoomCamp Article

# Time Series Modelling & Stock Forecasting in Python 

Time has always been a key factor in statistical analysis because of its dynamic, quantifiable nature and its impact on our daily lives. Time series algorithms are used extensively for analyzing and forecasting time-based data like budgeting, demand forecasting, production planning etc. Forecasting is among the most applied concept in data science and today, it has become all the more important with the emergence of real-time data and real-time analytics. 

I was new to time-series analysis in Python and wanted to challenge myself to make time series forecast using **Time Series Modelling** in Python. Trying to predict the stock market appeared an exciting eventhough complex area to explore.  Through this article, I wanted to discuss key concepts, techniques and methods used to perform basic **Time Series Analysis & Forecasting** in Python.  My goal here, was to use different time series algorithms and create a time-series based prediction service to make forecasts regarding Amazon stocks.

Before moving ahead, let me list down the key topics covered by me in this article.

### Topics Covered -

* [Key Concepts about Time-specific Data & Time Series Analysis](https://github.com/sukritishuk/ML_ZoomCamp_Article/blob/main/README.md#key-concepts-about-time-specific-data--time-series-analysis--)
* [Sources of Data](https://github.com/sukritishuk/ML_ZoomCamp_Article/blob/main/README.md#sources-of-data--)
* [Data Preparation](https://github.com/sukritishuk/ML_ZoomCamp_Article/blob/main/README.md#data-preparation--)
* [Exploratory Data Analysis (EDA) of Time Series](https://github.com/sukritishuk/ML_ZoomCamp_Article/blob/main/README.md#exploratory-data-analysis-eda-of-time-series--)
* [Slicing Amazon Stock Dataset for Forecasting](https://github.com/sukritishuk/ML_ZoomCamp_Article/blob/main/README.md#slicing-amazon-stock-dataset-for-forecasting--)
* [Testing for Stationarity](https://github.com/sukritishuk/ML_ZoomCamp_Article/blob/main/README.md#testing-for-stationarity--)   
* [Making a Time Series Stationary](https://github.com/sukritishuk/ML_ZoomCamp_Article/blob/main/README.md#making-a-time-series-stationary---)
* [Seasonal Decomposition of a Time Series](https://github.com/sukritishuk/ML_ZoomCamp_Article/blob/main/README.md#seasonal-decomposition-of-a-time-series--)
* [Selection of Non-seasonal and Seasonal Orders](https://github.com/sukritishuk/ML_ZoomCamp_Article/blob/main/README.md#selection-of-non-seasonal-and-seasonal-orders--)
* [Comparing Models and Interpreting Results](https://github.com/sukritishuk/ML_ZoomCamp_Article/blob/main/README.md#comparing-models-and-interpreting-results--)
* [Splitting the Dataset for Time Series Analysis](https://github.com/sukritishuk/ML_ZoomCamp_Article/blob/main/README.md#splitting-the-dataset-for-time-series-analysis--)
* [Model Selection for Stock Predictions](https://github.com/sukritishuk/ML_ZoomCamp_Article/blob/main/README.md#model-selection-for-stock-predictions--)
* [Making Predictions on Testing Set (unseen data) using most Optimal Model](https://github.com/sukritishuk/ML_ZoomCamp_Article/blob/main/README.md#making-predictions-on-testing-set-unseen-data-using-most-optimal-model--)
* [Time Series Forecasting](https://github.com/sukritishuk/ML_ZoomCamp_Article/blob/main/README.md#time-series-forecasting--) 


Before we move on to Time Series Analysis of Amazon Stock, let me first explain some key concepts about time-related data in general and time series analysis.

## Key Concepts about Time-specific Data & Time Series Analysis -

* **TimeStamp** refers to a particular moment in time (e.g., January 4th, 2022 at 7:00am). **Time Intervals and Periods** refer to a length of time between a particular beginning and end point. **Time deltas or durations** refer to an exact length of time, between two dates or times.
* **Time series** is a set of data points that occur over a span or succesive periods of time like Years, Months, Weeks, Days, Hours, Minutes, and Seconds. Time series differs from **Cross-Sectional Data** as the latter considers only a single point in time while the former looks at a series or repeated samples of data over a time period. 
* **Trends** involve a gradual increase or decrease in time series data like a rise in death rates due to outbreak of a deadly disease.
* **Seasonality** occurs in time series when a trend is periodically repeating itself e.g., increase in sale of home decorations around Christmas season. 
* **Stationarity** of time series means that the distribution of the data in here, doesn't change with time i.e., the series must have zero trend, constant variance & constant autocorrelation. Absence, of any of these 3 conditions makes it **Non-Stationary**. 
* **Rolling or Moving Aggregations** on data involves performing aggregations like average or sum of data, using a moving window over the entire dataset.
* **Resampling** of time series data is common to ease anaysis and improve visualizations. It involves converting data to a higher or lower frequency like daily data to monthly or yearly.
* **Time Series Analysis** involves studying what changes are made in the economic asset or concerned variable during a period of time (e.g. price of gold, birth rates, rainfall levels etc.). It also studies the influence of time over other variables in data.
* Characteristics in time series data like **seasonality, structural breaks, trends and cyclicity**, often differ from other types of data therefore, require unique set of tools and techniques for analysis.
* A time series includes three systematic components: **Level, Trend, and Seasonality**, as well as one non-systematic component termed **Noise**.
* Time Series Analysis & modelling of time-specific data can be done **only on Stationary time series**.  
* Time series analysis is done only on **continuous variables** and includes Trend Analysis, Forecasting for Cyclical fluctuations or Seasonal patterns. 
* **Trend Analysis** aims to analyze the movements in historical data using visualizations like line plots, bar charts while, **Forecasting** tries to predict the future values with available present & past data. 
* **Time Series Modelling** is different from other types of modelling techniques. **Univariate time series models** are forecasting models using only one variable (the target variable) and its temporal variation to forecast the future. **Multivariate time series models** are models using external variables also into forecasting alongwith the target variable. While the former models ares based only on relationships between past and present the latter use these alonwith the relation between external factors (exogenous variables).
* **One-step Forecasting** techniques involve predicting the only the next step for a time series. These can only predict one step ahead and cannot predict multiple steps at once. **Multi-step Forecasting** techniques can predict multiple steps into the future. They are more accurate and stable in forecasting.

Now, that we have briefly talked about the basics of time-related concepts lets start with understanding our dataset for this article and project.


## Sources of Data -

For Stock Forecasting I selected the stocks of [Amazon](https://en.wikipedia.org/wiki/Amazon_(company)), an American multinational technology company which focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence. I retrieved Amazon Stock data from Kaggle Public Dataset - [FAANG- Complete Stock Data](https://www.kaggle.com/aayushmishra1512/faang-complete-stock-data). 

The data for Amazon was available as a csv file (*Amazon.csv*). The data contained all the relevant columns such as Opening price, Closing price, Volumes of stock traded from 1997 to 2020. Once the dataset was finalized, I cleaned and formatted it using Python libraries such as Pandas, Numpy. Matplotlib and Seaborn were used to perform EDA analysis on Amazon time series, make visualizations and draw insights from different features. After this, time series analysis and Forecasting techniques were used to train different algorithms and make predictions.


## Data Preparation - 

### Loading & Studying the Data:

I retrieved the time series data for Amazon company stock from Kaggle as a csv file and then loaded it into a DataFrame using different Pandas' functions. It is a simple univariate time series. There were multiple numerical feature columns in the dataset (as listed below) along with one time-based feature column, *Date*. 

* *Open* and *Close* - Starting and final price at which the stock is traded on a particular day.
* *High* and *Low* - Maximum and minimum trading price of the stock for the day.
* *Close* - Last price at which the stock trades before markets closed for the day.
* *Adjusted Close (Adj Close)* - Closing price for the stock on a day adjusted for corporate actions like stock-splits or dividends.
* *Volume* - Total number of shares traded (bought or sold) for Amazon during a day.

![image](https://user-images.githubusercontent.com/50409210/151208664-a286b590-89b6-4cdf-b42a-fb7760d2ccce.png)

The dataset contained values from *May 1997 to August 2020* with **5852 rows** in total. Each row represented one trading day during this period. As the *Date* column was in **object** datatype format, before any time series analysis, we had to convert all time-specific columns into correct **Datetime** format. Python's [Pandas library](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html) contains extensive tools for working with dates, times, and time-indexed data.


### Data Cleaning and Formatting:

Python's built-in [datetime module](https://docs.python.org/3/library/datetime.html) helps in working with dates and times. The most fundamental of date/time objects are the **Timestamp** and **DatetimeIndex** objects. 

* First, I used the ***pd.to_datetime() function***, to parse the *Date* column and convert it from object to **datetime64** format. This encodes dates as 64-bit integers, allowing an arrays of dates to be represented very compactly. 

![image](https://user-images.githubusercontent.com/50409210/151963829-972f5aef-35a9-4f3b-aa36-dd2127d17873.png)

* I also used the [DatetimeIndex object](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html) to convert the *Date* column into DatetimeIndex first.
* Then. various attributes of this object were used (like year, month_name etc.) to extract different elements from each date entry. For instance, I used the ***.year() and .month_name()*** attributes to strip only the year and month names respectively from every Date row. These elements were then used as part of **Feature Engineering** to create new columns for *Year*, *Month Name*, *Day Name* etc. which would be used in EDA analysis.

![image](https://user-images.githubusercontent.com/50409210/151671634-deddae70-1374-4fc2-915e-25e6093f6d6c.png)

As the Amazon stock dataset had no missing values in any columns no imputations were needed for missing data.


## Exploratory Data Analysis (EDA) of Time Series -

Now, that I had the dataset cleaned and formatted I carried out some of the following Exploratory Data Analysis to better understand the time series for Amazon stocks.

As shown below, Amazon stocks had a very steep rise in Closing (adjusted) Price after 2016. However, its Volumes have declined from the levels in 2000 or 2008, when they were at peak.

  ![image](https://user-images.githubusercontent.com/50409210/151713881-263a10c8-8939-4e48-aec2-832cf1d612c3.png)
 
Both the 50-day and 200-day Moving Average plots for Amazon show a fairly smooth but rising trend in daily (Adjusted) Closing prices. This indicates that there has been an overall increase in the Amazon stock prices over the years.

  ![image](https://user-images.githubusercontent.com/50409210/151674901-950f0f48-fbb4-4896-8d28-4db07a7f28dc.png)
  
My next plot, visualized the average daily *Returns* for Amazon stock. The daily movements have reduced over time as we can see small percentage change (increase or decrease) in the stock value from 2016 onwards compared to that in the 2000s.
  
  ![image](https://user-images.githubusercontent.com/50409210/151675037-0ffa48a1-21ce-4aec-9cd7-9c4d464e8841.png)
  
Visualizing movements in Amazon stock Volumes from a Weekday and Month perpective was my next visulaization. As shown below, *Volumes* appear to be the largest on *Wednesdays* and *Thursdays* while almost Nil on weekends (Saturday, Sunday). This was due to no trading on weekends. Also, the month of *January* appears to have the highest Volumes for Amazon stocks followed by *July, October* and *November*. The month of *August* on the other hand, has the lowest levels of Volume. 

  ![image](https://user-images.githubusercontent.com/50409210/151675266-0d9c80ca-6921-4996-bb40-9f18f598ed5f.png)

Next, I wanted to explore the correlation among different price indicators, difference between them and Volume for the Amazon stock dataset. For example. I wanted to understand the relation between Volume and Lowest Price when deducted from Daily Close. This was done by adding new columns for Price differences like *Close-High, High-Open* etc. and creating a Heatmap to visualize correlations among variables.

As we can see from the plot below, the **Close minus High** (shown as Close-High in plot below) variable showed the maximum positive correlation with Volume. We can say that if the Amazon Closing price stays away from High value, it may lead to more transactions or trading volumes that day for the stock. Also, the **Open minus Close** (shown as Open-Close in plot below) variable had the most negative correlation with Volume. This could indicate that the lesser the difference between Closing price of Amazon stock on a day and its Opening price the next day, higher would be its trading Volume.

![image](https://user-images.githubusercontent.com/50409210/151675419-4490f38d-652b-47e7-b8be-9d6dab70c664.png)

Both these were a very preliminary analysis, because technical analysis for a stock depends on many other factors. 
 
  

## Slicing Amazon Stock Dataset for Forecasting - 

For ease of time series analysis and forecasting, I wanted to restrict using the Amazon stock dataset to include only recent time periods. Therefore, I went down reducing it from 1997 to include stock price data from the **2015 onwards** only. In addition to this, I used only the **Adjusted Close** and **Date** columns from the dataset to further analyze the time series, for training different algorithms and making predictions. 

This reduced my target variable (Adusted Close) and time-based column from 5852 rows to **1415 rows only**. Hereafter, only this data subset would be split into training and testing sets before training different time series algorithms like ARIMA or Prophet and making forecasts.


## Testing for Stationarity - 

Time Series analysis only works with stationary data therefore, I needed to determine if the Amazon time series was stationary or not. A dataset is stationary if its statistical properties like mean, variance, and autocorrelation do not change over time. If Amazon time series had some trend or was non-stationary it had to be made stationary before fitting any time series model.

Testing for Stationarity can be done using one or more of the following methods:-
* Visualizing Rolling Statistics
* Performing Augmented Dicky-Fuller (ADF) Test


### Rolling Statistics:

One of the most popular rolling statistics is the Moving Average. This takes a moving window of time, and calculates the average or the mean of that time period as the current value. Visualizing the rolling statistics for Amazon time series would help me determine if it is stationary or not. A series becomes stationary when both the mean and standard deviation are flat lines (constant mean and constant variance).  

I needed to check this for Amazon time series as well so calculated and visualized the mean and standard deviation of Adjusted Close price for a 12-day window. Moving Standard Deviation is another statistical measurement of market volatility which I tried analyzing. As visible from the plots below, the rolling mean for Amazon Adjusted Close fitted the original data values quite closely.

  ![image](https://user-images.githubusercontent.com/50409210/151713977-fef5d7b1-ab0f-4b65-a2d4-6f592e99a3dd.png)

The rolling standard deviation plot appears to be quite smooth in the beginning (2015-17 end) but becomes little volatile after 2018. In 2019 and end of 2020 there are some large movements. 

From the plots above, I found that there was an increasing mean and standard deviation, indicating that this series was not stationary. However, to become more sure about stationarity I performed the ADF test as well.


### Augmented Dicky-Fuller (ADF) Test:

The [Augmented Dicky-Fuller or ADF Test](https://machinelearningmastery.com/time-series-data-stationary-python/), is a common statistical hypothesis test for stationarity and is based on the concept of unit root. *Null Hypothesis (H0)* for this test states that the time series is non-stationary due to trend i.e., a unit root is present in the time series. The *Alternative Hypothesis (Ha)* states that the data is stationary.  In this case, we cannot reject the null hypothesis we will have to assume that the data is non-stationary.

In Python, the [ADF test](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html) can be performed using the ***adfuller() function*** from the **statsmodels library**. The result object of the ADF test is a tuple with following key elements:

  * Zeroth element - **Test statistic** (the more negative this number is, the more likely that data is stationary) 
  * Next element - **p-value** (if p-value < 0.05, we reject the H0 and assume our time series to be stationary)
  * Last element - **a Python Dictionary with critical values** of the test statistic equating to different p-values

I performed an ADF test for Amazon time series afor different statistical significance to determine whether it was staionary or not. I got the following results form the test:-

![image](https://user-images.githubusercontent.com/50409210/151357493-7c6aa93b-99ab-4495-ba45-5d8d0be35a6a.png)

* The test results above show that the ***Test statistic was positive*** and ***p-value > 0.05***.  
* Additionally, the test statistics exceeded the critical values hence, the data was nonlinear. I did not reject the null hypothesis, and concluded that Amazon time series was non-stationary. 

Thus, by using both methods, plotting the rolling statistics and analyzing ADF test results I could determine that **Amazon time series was Non-Stationary**. 

Next, I had to work towards making the time series stationary before applying any Time series modelling.



 ## Making a Time Series Stationary  - 
 
 As the Amazon dataset is a non-stationary time series I had to now stationarize it before applying any time series modelling techniques. 
 
 The following methods can be used to make a time series stationary:-
 * De-trending the time series
 * Differencing the time series
 
### De-trending the time series:

This method removes the underlying trend in the time series by standardizing it. It subtracts the mean and divides the result by the standard deviation of the data sample. This has the effect of transforming the data to have mean of zero, or making it centered, with a standard deviation of 1. Once this is done, an ADF test is performed on the de-trended time series to confirm the results.

The ADF test results on de-trended stock data for Amazon below, shows a p=value < 0.05 (the p-value is very small, indicating that the alternative hypothesis  or stationarity is true) and there is also a negative Test statistic, which means that the Amazon time series has become stationary now.

![image](https://user-images.githubusercontent.com/50409210/151385463-0e6bc76f-8005-4ace-9cd2-e7bbdaf22a79.png)

On plotting the rolling statistics for the de-trended Amazon stock as shown below I found that the Amazon time series has become stationary. This was indicated by the relative smoothness of the rolling mean and rolling standard deviation plots compared to the original plot for de-trended Amazon stocks.

![image](https://user-images.githubusercontent.com/50409210/151700801-3c67db0b-a6d2-489e-a97d-bdb34c41e778.png)

### Differencing the time series:

Another widely used method to make a time series stationary is Differencing. This method removes the underlying seasonal or cyclical patterns in the data thereby removing the temporal dependence or series' dependence on time. 
    
Following are some key concepts regarding Differencing:-

* In [Differencing](https://machinelearningmastery.com/difference-time-series-dataset-python/), from each value in a time series we subtract the previous value. Differencing can be performed manually or using the Pandas' ***.diff() function***. The resulting missing or NaN value at the start is removed using the ***.dropna() method***. Using Pandas function helps in maintaining the date-time information for the differenced series.
* When applied for the first time the process of differencing is called the **First Order of Differencing**. 
* Sometimes, the differencing might be applied more than once if the time series still has some temporal dependence, then it is called **Second Order of Differencing** and so on.
* The value of **d**, therefore, is the minimum number of differencing needed to make the series stationary. And if the time series is already stationary, then d = 0.
* We need to be careful not to **over-difference a time series**, because, an over-differenced series may still be stationary but would affect the model parameters.
* **Optimal order for Differencing** - minimum differencing order required to get a near-stationary series with a defined mean and for which the ACF plot quickly reaches zero.
     

In order to get the right order of differencing for Amazon time series I took the **first order of differencing**, performed the ADF test on the differenced series and then plotted the correlation plots (ACF and PACF) for the differenced series. I got the following result:-

![image](https://user-images.githubusercontent.com/50409210/151394410-11eeacb5-58f4-44e4-b6a2-c0d020b8d7cd.png)

Allthough I got satisfactory ADF test results i.e., a negative test statistic and a p_value < 0.05, as an experiment, I also took a **second order of differencing** by repeating the same steps again. This was done just to analyze the impact of second order of differencing on Amazon time series. I got the following results:-

![image](https://user-images.githubusercontent.com/50409210/151395152-ae6b1ef4-ce3a-46ec-89ea-6dd72c1bf409.png)

Both the 1st Order and 2nd Order of Differencing yielded very small p-values and also very negative test statistics. But by looking at the ACF & PACF plots for the 2nd order of differencing I found that the lag goes into the negative zone very quickly. This indicates that the Amazon stock series might be getting over-differenced with 2nd order differencing. To avoid this over-differencing, I restricted the **Order of Differencing** for this dataset to the first order only i.e., to **d = 1**.

By looking at the EDA for Amazon time series I could not be very sure about seasonality in the data. This can only be found if the Amazon time series is decomposed. So, next we will look into the seasonal elements in our dataset.


## Seasonal Decomposition of a Time Series - 
 
A seasonal time series generally has some predictable patterns that repeat regularly (i.e., after any length of time). 

* Every time series is a combination of 3 parts - **Trend, Seasonality and Noise or Residual**. Separating a time series into its 3 components is called **decomposition of time series**. 

  * Trend - Increasing or falling value in a time series.
  * Seasonality - Short-term recurring movements in a time series.
  * Noise - Random variance in a time series explainable neither by trend nor seasonality.

* In Python, [Automatic decomposition](https://analyticsindiamag.com/why-decompose-a-time-series-and-how/) of a time series can be performed using the ***seasonal_decompose() function*** in ***statsmodel library***. It requires one to specify whether the model is additive or multiplicative. The result object contains arrays to access four pieces of data from the decomposition. This can then be used to generate a plot that will split the time series into trend, seasonality, and noise.

* **Additive vs multiplicative seasonality** - These are the two methods to analyze seasonality of a Time Series:- 

   a) ***Additive*** - Seasonal pattern just adds or has a linear behaviour i.e., where changes over time are consistently made by the same amount.
  
  ![image](https://user-images.githubusercontent.com/50409210/151403195-6bf025e3-bb0e-4e02-bf0f-c024da996b57.png)
  
   b) ***Multiplicative*** - Trend and seasonal components are multiplied and then added to the error component and behaviour is non-linear (exponential or quadratic). The amplitude of the seasonal oscillations get larger as the data trends up or get smaller as it trends down. To deal with this we take the log transform of the data before modelling it.

  ![image](https://user-images.githubusercontent.com/50409210/151403945-c5a0ed42-a2bc-4f11-aac9-ae6efe1078e8.png)

I decomposed the Amazon time series using statsmodel's seasonal_decompose() function and setting the ***period parameter*** to 12 (number of data points in each repeated cycle). I also specified the ***model paramter*** as ***multiplicative***. It returned a decompose-results object. Then I used the ***plot() method*** of this object to plot the components of the decomposed Amazon time series. It resulted in the following plot:-

![image](https://user-images.githubusercontent.com/50409210/151399016-7fb9d988-14b4-4789-82f7-fa575512f10d.png)

We can see from the decomposition plot above that the Amazon stock time series has both trend and seasonality. The pattern of trend is positive and increasing in nature however, the seasonality pattern does not appear to be very clear from this decomposition plot as well. But as we know that there is some seasonality in the Amazon time series I would be using the seasonal orders also while training time series algorithms to our dataset. 


Next, I needed to select optimal orders for the dataset both seasonal and non-seasonal before applying any time series algorithms for making forecasts.



## Selection of Non-seasonal and Seasonal Orders -

### Model orders:

When fitting and working with AR, MA, ARMA or SARIMA models, it is very important to understand the model order. We need to pick the most optimal model order before fitting our time series to a model in order to make better predictions. 

* When we set either p or q to zero, we get a simpler AR or MA model.
* There are three important orders in ARIMA:

  * **Autoregressive order (p)** - Past values used for forecasting the next value or number of lag observations included in model (lag order).
  * **Order of differencing (d)** - If d = 0, we simply have an ARMA model.
  * **Moving Average order (q)** - Size of the moving average window.
  
* SARIMA or seasonal ARIMA model is a combination of non-seasonal orders and seasonal orders. Both these parts have orders for the autoregressive, difference and moving average parts.
  * **Non-seasonal orders** - autoregressive order (p), order of differencing (d) and moving average order (q)
  * **Seasonal orders** - autoregressive order (P), order of differencing (D), moving average order (Q) and a new order (S), which is the length of the seasonal cycle


### Choosing Model Orders: 

#### Using ACF and PACF plots - 

[Autocorrelation and Partial Autocorrelation plots](https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/) are used in time series analysis and forecasting to graphically summarize the strength of a relationship among observations. A plot of the autocorrelation of a time series by lag is called the AutoCorrelation Function or ACF. In Python, it can be plotted using the ***plot_acf() function*** from the ***statsmodels library***. Partial Autocorrelation Function or PACF on the other hand measures the correlation of the observation, with observations at intervening time steps. It can be plotted using the ***plot_pacf() function*** in Python.

One of the main ways to identify the correct model order is by using the Autocorrelation Function (ACF) and the Partial Autocorrelation Function (PACF). By comparing the ACF and PACF for a time series one can deduce the model order. The time series must be made stationary before making these plots.


#### Using AIC and BIC values - 

The Akaike Information Criterion (AIC), is a metric which tells us how good a model is. The Bayesian Information Criterion (BIC), is very similar to the AIC. AIC and BIC are two ways of scoring a model based on its log-likelihood and complexity. Models which fit the data better have lower AICs and BICs, while BIC penalizes overly complex models. Mostly, both AIC and BIC will choose the same model. 

These can be found on the right side of the summary of the fitted-models-results object. In Python these can be computed using the ***.aic attribute*** and the ***.bic attribute***.


### Manual Selection of Orders:

For Amazon time series, I tried to manually select orders using for-loops and different ranges. Then, I fitted a time series algorithm with different order combinations, extracted the AIC and BIC values for each combination of orders and selected the order combination yielding the lowest AIC value. I found that the BIC score also turned out to be the lowest for the same order combinations. 

I used this same process to select the following order types:- 

* **Finding only the optimal Non-Seasonal Orders (p,d,q)** - This resulted in giving me the lowest AIC and BIC values for Amazon stock data by setting the order values to **p=2, d=1 and q=2**. 
* **Finding only the optimal Seasonal Orders (P,D,Q)** - Here, I already pre-defined the length of seasonal cycle (S) as 7, as the seasonlity of our Amazon time series appeared to be daily, I also pre-defined the non-seasonal orders to be p=2, d=1, q=2 (results from our last order selection) then used the for-loop to select the best seasonal order combinations. From the results, lowest AIC and BIC values were yielded by setting the seasonal order values to **P=0, D=1 and Q=2**.
* **Finding both Non-seasonal (p,d, q) and Seasonal (P,D,Q) Orders** - Here, I had already pre-defined the length of seasonal cycle or S = 7, d = 1 and D = 1. It resulted in yielding the folowing order combinations, with the first row showing the lowest AIC & BIC values:- 

![image](https://user-images.githubusercontent.com/50409210/151419135-8f5b94c8-3fa2-4c20-9a8f-1d938fbe943d.png)

  * Non-Seasonal Orders - p = 0, d = 1 and q = 1
  * Seasonal Orders - P = 0, D = 1 and Q = 1


### Automated Selection of Orders:

The [pmdarima package](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html), automatically discovers the most optimal order for an ARIMA model. The ***auto_arima function*** from this package loops over model orders to find the best one. Many of these have default values and the only required argument to the function is the data. Optionally, we can also set the order of non-seasonal differencing; initial estimates of the non-seasonal orders; and the maximum values of non-seasonal orders to test.

I used this method to choose the most optimal model order for Amazon time series.

* I set the ***period for seasonal differencing*** as Daily or m=7.
* I specified the ***seasonal parameter***=True as Amazon time series appears to be seasonal in nature.
* The auto_arima function performs differencing tests (e.g., Kwiatkowski–Phillips–Schmidt–Shin, Augmented Dickey-Fuller, or Phillips–Perron) to determine the order of differencing, d. Here, I specified the ****test argument***='adf' to specify the use of Augmented Dickey-Fuller test.
* I specified only the ***max_p*** and ***max_q*** parameters as 2.
* After determining the optimal order of seasonal differencing (D), auto_arima also seeks to identify the optimal P and Q hyper-parameters if the seasonal option (i.e., seasonal=True) is enabled.

The object returned by the function is the results object of the best model found by the search.

![image](https://user-images.githubusercontent.com/50409210/151422371-39e053bb-e78b-47bd-be95-6e04df35011e.png)

As shown above, the Auto ARIMA model assigned the following order values to Amazon time series data:-
   * Non-Seasonal Orders - 2, 1 and 2 to p, d, and q, respectively.
   * Seasonal Orders - 0, 0, 2 and to P, D, Q respectively.


Below is a snapshot of most optimal order values yielded by different manual and automatic selection methods for Amazon time series:- 

![image](https://user-images.githubusercontent.com/50409210/152012097-4a8c3b5d-bf56-48d6-8f14-5b86443e0797.png)



## Comparing Models and Interpreting Results - 

### Interpreting Model Summary:

Interpreting results from a time series algorithm can be a trying experience. An important component of statsmodel library is that we can inspect the results from the fitted model using a ***.summary() method***. 

![image](https://user-images.githubusercontent.com/50409210/152013535-e99ae03a-ea7f-4298-b72c-53763cf2f337.png)

Let us walkthrough the components of a summary result as shown in the figure above:-
* **General Information** - The top section includes useful information such as the order of the model that we fit, the number of observations or data points, the name of the time series. Statsmodels uses the same module for all of the autoregressive models, therefore, the header displays **SARIMAX Results** even for an AR model.
* **Statistical Significance** - The next section of the summary shows the fitted model parameters, like the ar.L1 and ar.L2 rows for e.g., if fitting an ARMA(2,1) model having AR-lag-1 and lag-2 coefficients. Similarly, the MA coefficients are in the last rows. The first column shows the model coefficients whilst the second column shows the standard error in these coefficients. This is the uncertainty on the fitted coefficient values. We want each term to have a **p-value < 0.05**, so we can reject the null hypothesis with values that are statistically significant.
* **Assumption Review** - 
  * *Ljung Box (Q)* test estimates that the errors are white noise. Since its probability Prob(Q) is above 0.05, we can’t reject the null that the errors are white noise.
  * *Heteroscedasticity (H)* tests that the error residuals have the same variance. Our summary statistics shows  a p-value of 0.00, which means we reject the null hypothesis and our residuals show variance. 
  * *Jarque-Bera (JB)* tests for the normality of errors. We see a test statistic with a probability of 0, which means we reject the null hypothesis, and the data is not normally distributed. 
  * We also see that the distribution has a slight positive skewness and a large kurtosis.
* **Fit Analysis** - Values in this section of the summary like the Log-Likelihood, AIC, BIC, and HQIC help compare one model with another. AIC penalizes a model for adding parameters and BIC alongwith AIC penalize complex models. Lower the values for these indicators better is the fit of the model on the distribution of data.



### Interpreting Plot Diagnostics:

For an ideal model the residuals should be uncorrelated white Gaussian noise centered to zero. We can use the results object's ***.plot_diagnostics method*** to generate four common plots for evaluating this. This [4-plot](https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMAResults.plot_diagnostics.html) is a convenient graphical technique for model validation and consists of the following set of plots:-

![image](https://user-images.githubusercontent.com/50409210/151553656-6dd826ed-e8df-4afd-975e-f305f34eafe8.png)

* **Residuals Plot** - This plot shows the one-step-ahead standardized residuals. For good fitted model there should be no obvious structure in the residuals. 
* **Histogram plus KDE Estimate Plot** - This plot shows us the distribution of the residuals and tests for (normal) distribution. The red line shows a smoothed version of this histogram and the yellow line, shows a normal distribution. For a good model both the red and yellow lines should be as close as possible.
* **Normal Normal Q-Q Plot** - This one also also compares the distribution of the model residuals to normal distribution. For residuals to be normally distributed all the points should lie along the red line, except some values at either end.
* **Correlogram** - This is an ACF plot of the residuals wherein 95% of the correlations for lag greater than zero should not be significant. If there is significant correlation in the residuals, it means there is some information in the data not captured by our model.

  ![image](https://user-images.githubusercontent.com/50409210/151554111-41043315-8bb7-4670-8f44-97e3d613511a.png)

Model results from each of the models selected and used for Amazon time series were interpreted using summary results and plot diagnostics as discussed above.


## Splitting the Dataset for Time Series Analysis - 

Splitting the dataset is an important exerise before selecting and fitting machine learning algorithms on any dataset. For time series analysis, this split in dataset would be slightly different from other cases where we try to split datasets randomly into train-test subsets. Here, as we would be using past values to make future predictions, we would be splitting the data in relation to time i.e., training our algorithms on data coming earlier in time and testing data that comes later. 

The plot below, depicts the split Amazon time series into training and testing subsets in the ratio of 80:20. The plot shows how data earlier in time (in green) would be used for training the algorithms while latest data (in blue) would be used for making forecasts.

![image](https://user-images.githubusercontent.com/50409210/151556040-343ff83f-f9d1-455f-9156-e0c88452bc5d.png)


## Model Selection for Stock Predictions -

Now that we have seen the main specificities of time series data, it is time to look into the types of models that can be used for predicting time series. This task is generally referred to as forecasting.

There are different kind of time series analysis techniques with the most common and classic being the *ARIMA models* like the following -

* Autoregression (AR)
* Moving Average (MA)
* Autoregressive Moving Average (ARMA)
* Autoregressive Integrated Moving Average (ARIMA)
* Seasonal Autoregressive Integrated Moving-Average (SARIMA)
* Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)

The set of models are smaller in nature and can be combined or used as a standalone model in itself. **Autoregressive (AR), Integrated (I), Moving Average(MA)** are combined to form **Autoregressive Moving Average (ARMA), and Autoregressive Integrated Moving Average (ARIMA)** models. **Seasonal Autoregressive Integrated Moving-Average (SARIMA)** model combines the ARIMA model with the ability to perform the same autoregression, differencing, and moving average modeling at the seasonal level. This model is useful if seasonality is present in the time series as it works with seasonal parameter on top of the regular or non-seasonal parameters. 

The **Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)** is an extension of the SARIMA model that also includes the modeling of exogenous variables. These are parallel input sequences having observations at the same time steps as the original series.

Some deep learning-based techniques include **Long-short Term Memory(LSTM)**, which are specific type of Recurrent Neural Networks (RNNs) and quite complex in nature.


For analyzing Amazon stock time series I used the following Time Series Modelling techniques:-

### Autoregressive Integrated Moving Average (ARIMA):
  
  * It is used to predict the future values of a time series using its past values and forecast errors.
  * Very popular statistical method for time series forecasting and capable of predicting short-term share market movements.
  * We can also implement an ARIMA model using the SARIMAX model class from statsmodels.
  * ARIMA model has three model orders - p the autoregressive order; d the order of differencing; and q the moving average order
  
Like we discussed before, manual selection of model orders on Amazon time series resulted in the most optimal non-seasonal orders to be **p=2, d=1 and q=2**. These were now used to fit an **ARIMA(2,1,2) model** on our training subset. The folowing summary results were generated from this model as output:-
  
![image](https://user-images.githubusercontent.com/50409210/151714219-823c3192-af13-4a04-a20b-81e6c47628e7.png)

As we can see in the results above, the JB p-value or Prob(JB) is zero, which means we should reject the null hypothesis that the residuals are normally distributed.

![image](https://user-images.githubusercontent.com/50409210/151714232-67a5c4d4-13de-4dff-a441-c839bcc6fee3.png)

The residual diagnostic plots generated showed some obvious patterns in the residuals plot towards the right end of the plot. The KDE curve was not very similar to the normal distribution plot. Most of the data points did not lie on the red line as shown by the histogram and Q-Q plots. The last plot, correlogram, which is just an ACF plot of the residuals 95% of the correlations for lag greater than zero should not be significant and they appear to be not significant here as well.

Both these output interpretations above, suggested that the ARiMA(2,1,2) model did not fit the Amazon stock time series too well yet.


### Seasonal Autoregressive Integrated Moving-Average (SARIMA):

The decomposition plot for Amazon data suggested that there was some seasonality in the time series. This prompted me to use the SARIMA modelling technique next. The notation for SARIMA model involves specifying the order for the AR(p), I(d), and MA(q) models as parameters to an ARIMA function and AR(P), I(D), MA(Q) and m parameters at the seasonal level, e.g. SARIMA(p, d, q)(P, D, Q)m where “m” is the number of time steps in each season (the seasonal period). 

Although, manual selection of orders yielded SARIMA(0,1,1)(0,1,1,7) for Amazon time series instead of using these parameters, I fittied the training subset with a SARIMA model, using auto_arima function of the pmdarima package.  As we can see below, the most optimal orders selected from here were **p=2, d=1, q=2, P=0, D=0, Q=2, S or m=7**. Thus, the SARIMA model used was **SARIMA(2,1,2)(0,0,2,7)**.

![image](https://user-images.githubusercontent.com/50409210/151714253-257db7f0-40e0-4684-a30a-901639b12ed1.png)

The summary result from the model above show, that the model does not meet the condition of no correlation (independence in the residuals) because the p-value of the Ljung-Box test Prob(Q) is greater than 0.05. Thus, we cannot reject the null hypothesis of independence. Also, we cannot say that the residual distribution is having constant variance (homoscedastic) because the p-value of the Heteroskedasticity test Prob(H) is smaller than 0.05.

As we see below, there appears to be very little difference in the residual plot from SARIMA as compared to that from ARIMA yet the SARIMA model appears to perform better in this time series than the ARIMA model.

![image](https://user-images.githubusercontent.com/50409210/151714282-ce214044-23c7-4bc9-8948-df92674e6a4a.png)

We can get a better picture of the model's performance when we predict on the testing subset of the dataset using these algorithms. This is dealt with further, but before that I will explain using another algorithm Prophet.


### Prophet by Facebook:

I used an open-source algorithm developed by Facebook’s Core Data Science team named [Prophet](https://github.com/facebook/prophet). It is a third-party time series forecasting library which requires almost little data preprocessing and is very simple to implement. 

  * Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality. 
  * It considers the effects of holiday quite well and is capable of handling seasonality-related components in data by the use of simple parameters to fine-tune the model like     specifying holidays, daily seasonality etc
  * The input for Prophet is a dataframe with two columns, a Date and a target variable column named as - **ds** and **y** respectively.

  ![image](https://user-images.githubusercontent.com/50409210/151583807-582fe6b6-159f-4836-97d9-e7a6cd58d94c.png)
  
   While working with the Prophet algorithm for Amazon time series I undertook the following steps:-

   *  Firstly, I created a **Prophet instance** by specifying the Daily seasonality effects for the time series using the ***daily_seasonality=True parameter***. 
   *  **Making In-sample Forecast for Amazon Stock** - After this I trained the model on the training subset and then made in-sample predictions for the duration of testing subset i.e., 283 time periods into the future by specifying the dates for this period. This yielded a DataFrame of forecasts with key parameters as columns ("ds” indicating the Date, “yhat” indicating the predicted time series data, “yhat_lower” and “yhat_upper” indicating the probable lower and upper limit of forecasts).
  
  ![image](https://user-images.githubusercontent.com/50409210/152033936-493a5576-50b3-4ba8-8d98-9722a057a0e9.png)
  
  * **Making Out-of-Sample Forecast for Amazon Stock** - Then, I used Prophet to make predictions about the next 182 time periods i.e., approximately the next 6 months into the future. This was done by fitting a fresh Prophet model to the entire Amazon data (combined training and testing subset) and setting the ***periods paramter*** to 182. Then using the ***.predict() function*** to make predictions.
  
  ![image](https://user-images.githubusercontent.com/50409210/152023826-b6e8af6e-ceaf-464a-988d-46a82218ee28.png)
  
  *  Prophet also allowed me to plot different components from the forecasts using the ***plot_components parameter*** thereby showing the trend, yearly weekly and daily plots.
  *  **Adding Changepoints** is another useful feature in Prophet, allowing one to put more emphasis and find reasons for changes in trajectory or trends in data. Changepoints      are the datetime points where the time series have abrupt changes in the trajectory. By default, Prophet adds 25 changepoints to the initial 80% of the dataset.

     ![image](https://user-images.githubusercontent.com/50409210/152024007-cca80ab0-8d2a-46e4-a706-d60b26c059f6.png)


## Making Predictions on Testing Set (unseen data) using most Optimal Model - 

Although, training both **ARIMA(2,1,2)** and **SARIMA(2,1,2)(0,0,2,7)** on the training subset of Amazon series yielded very similar summary results and diagnostic plots, the SARIMA model seemed to be a better fit for this dataset because it has seasonal orders hence, takes into account some seasonality in our data.

Therefore, I used the **SARIMA(2,1,2)(0,0,2,7)**, to fit to the training set and make predictions on the testing set i.e., our unseen time series for Amazon. 

![image](https://user-images.githubusercontent.com/50409210/152025111-0fed364a-bb2c-4b92-8016-409a95245cfb.png)

From the above plot, we find that this model SARIMA(2,1,2)(0,0,2,7) performs not extremely well in making prediction about Amazon time series testing subset. From the plot above we can see that it over-predicts (blue line) the stock prices for the testing set while in actual they remain little low (yellow line). 

Next, I would use this model to make future or out-of-sample forecasts on the stock. But before that let me evaluate the performance of our forecasting algorithm.

I used **Mean Absolute Percentage Error (MAPE)** to test predictions from the SARIMA(2,1,2)(0,0,2,7) model on the testing subset against actual values. It yielded a **MAPE of 12.9%** indicating thet optimal SARIMA model was only **87.1% accurate** in predicting the test set observations. This suggests that this model is not exceptional and could have performed better if MAPE value was much lower.

But for a starter like me and looking at the complexity in stock price movements this classical ARIMA-based model yielded a fairly decent output for Amazon time series.


## Time Series Forecasting -

Lastly, I tried my hands on doing some forcasting for Amazon time series. Time series forecasting is the use of a model to predict future values based on previously observed values. 

Here, I performed 3 kinds of forecasting:-
* In-sample forecasting
* Dynamic Forecasting
* Out-of-Sample Forecasting


### Generating In-Sample (One-Step Ahead) Predictions:
  
This forecasting technique allows to evaluate how good our model is at predicting just one value ahead. 

  * We can use the SARIMAX fitted results object's ***get_prediction() method*** to generate in-sample predictions. 
  * We can set the ***start parameter*** as a negative integer stating how many steps back to begin the forecast. For the Amazon time series, I set the start parameter to -30 as I wanted to make predictions for the last 30 time-periods of the Amazon data. It returned a forecast object. 
  * The central value of the forecast is extracted and stored in the ***predicted_mean attribute*** of the forecast object and can be used alongwith the lower and upper confidence limits to plot forecasts. 
  * Here, the mean prediction is marked with a red line while the uncertainty range is shaded. The uncertainty is due to the random shock terms that we can't predict.

![image](https://user-images.githubusercontent.com/50409210/151714373-5c16c06c-ac16-4c44-880a-4c628355770f.png)

In the above plot I tried to make one-step ahead or in-sample predictions by training most optimal SARIMA(2,1,2)(0,0,2,7) model on training subset. I then made predictions for the last 30 days after the training subset, i.e. into the testing subset. This forecasted data is shown as a red line in the above plot. The actual or observed values were made lighter by the use of ***alpha parameter***. It appeared from the plot that our forecasted values aligned quite well with the observed ones as the plots appeared to completeley overlap.

### Generating Dynamic Forecasts: 

We can make predictions further than just one step ahead by using dynamic prediction technique. It first predicts one step ahead, and then use this predicted value to forecast the next value after that. 
  * Here, we set the  ***dynamic parameter=True*** in addition to the steps for making one-step ahead predictions. 
  * Making dynamic predictions, implies that the model makes predictions with no corrections for a future period of time, unlike making one-step-ahead predictions. Once the predictions are made for the period, we can compare predictions with actual values.

![image](https://user-images.githubusercontent.com/50409210/151714390-efd27b09-6b4f-46d6-b659-bc303f241c74.png)

For Amazon dataset, here too I set the start parameter to -30 as I wanted to make dynamic predictions for the last 30 time-periods of the Amazon data. Here also the dynamic forecasts are shown as a red line. The actual or observed values have been made lighter by the use of alpha parameter. It appears that our dynamic forecast do not align very well with the observed ones but are reasonably fine.


### Making Out-of-Sample Forecasts:

After testing our predictions in-sample, we can use our model to predict the future. Lastly, I tried making out-of-sample forecasts on Amazon time series.

* To make future forecasts we use the ***get_forecast method*** of the results object. 
* We choose the number of steps after the end of the training data to forecast up to by specifying the ***steps parameter***. For Amazon time series I wanted to make forecast for approximately 6 months into the future therefore used **steps=182** to make forecasts.

This yielded the following range of predicted values as confidence intervals:-

![image](https://user-images.githubusercontent.com/50409210/151714446-e17166a6-2396-4a92-a2d1-0119448ce2a3.png)

The plot in red below shows the Forecasted Adjusted Closing price for Amazon stock in the next 6 months or 182 time-periods into the future. It appears that the price would be fairly high in future also as shown by the shaded area of confidence intervals or predicted price ranges.

![image](https://user-images.githubusercontent.com/50409210/151714422-62599fa3-a7b7-4c22-82b4-25136bfaf75d.png)



#### References and Sources -

* https://jakevdp.github.io/PythonDataScienceHandbook/03.11-working-with-time-series.html
* https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
* https://analyzingalpha.com/interpret-arima-results
* https://neptune.ai/blog/select-model-for-time-series-prediction-task

