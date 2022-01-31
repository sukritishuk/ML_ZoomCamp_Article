## ML_ZoomCamp Article

# **Amazon Stock Price Prediction using Time Series Forecasting**

## Introduction:

Time has always been a key factor in statistical analysis because of its quantifiable and ever-changing nature and its impact on our daily lives. Today, it has become all the more important in both financial & non-financial contexts with the emergence of real-time Data and real-time Analytics. 


In machine learning, Time Series Analsis & Forecasting is among the most applied in Data Science in real-world scenarios like financial analysis, demand forecasting, production planning etc. Through this article, I wanted to discuss key concepts, techniques and methods used to perform basic **Time Series Analysis & Forecasting in Python**. Time series analysis in Python uses timestamps, time deltas, and time periods for plotting time series data. My goal was to explore these areas and create a prediction service (model) to make accurate forecasts for Amazon stock data. 

Steps for making Amazon Stock Forecasting -

* Loading & Studying the Data 
* Data Cleaning and Formatting - Changing data types, Missing Values Detection, Feature Engineering
* Exploratory Data Analysis (EDA) of Time Series - Correlation, Moving Averages, Percentage Change in Stocks, Resampling 
* Slicing Amazon Stock Data for Forecasting 
* Testing for Stationarity 
* Making a Time Series Stationary
* Seasonal Decomposition of a Time Series
* Selection of Non-seasonal and Seasonal Orders -
 
  * Manual selection of orders
  * Automated selection of orders - using Pmdarima library (auto_arima)
* Comparing Models and Interpreting Results
* Splitting the Dataset for Time Series Analysis
* Model Selection for Stock Predictions - ARIMA, Seasonal ARIMA (SARIMA) Model, Auto ARIMA, Prophet
* Making Predictions from Most Optimal Model on Unseen Data (testing subset)
* Time Series Forecasting - 

    * Generating In-Sample (One-Step Ahead) Predictions for Amazon stock data
    * Generating Dynamic Forecasts 
    * Making Out-of-Sample Forecasts for Amazon stocks 



### Sources of Data -

I retrieved Amazon Stock data from Kaggle Public Dataset - [FAANG- Complete Stock Data](https://www.kaggle.com/aayushmishra1512/faang-complete-stock-data). The data was available as a csv file (*Amazon.csv*). The data contained all the relevant data such as Opening price, Closing price, Volumes of stock traded about the Amazon stock from 1997 to 2020.


### Data Preparation - 

**Loading & Studying the Data:**

Firstly, I retrieved the time series data for Amazon company stock from Kaggle. There are multiple numerical feature columns in this dataset like below, apart from one time-based feature Date. 
* Open and Close - starting and final price at which the stock is traded on a particular day.
* High and Low - maximum and minimum trading price of the stock for the day. 
* Close - last price at which the stock trades before markets closed for the day.
* Adjusted Close (Adj Close) - Closing price for the stock on a day adjusted for corporate actions like stock-splits or dividends.
* Volume - total number of shares traded (bought or sold) for Amazon during a day.

![image](https://user-images.githubusercontent.com/50409210/151208664-a286b590-89b6-4cdf-b42a-fb7760d2ccce.png)

The dataset pertains to the period between May 1997 and August 2020. Before any time series analysis, all the time-specific variables must be in the Datetime format. Thus, the following columns would have to be formatted first. Pandas library contains extensive tools for working with dates, times, and time-indexed data.


**Data Cleaning and Formatting:**

I cleaned and pre-processed it, using Python in-built functions and libraries such as Pandas, Numpy. Then, Matplotlib and Seaborn libraries were used to perform EDA analysis on Amazon time series data, make visualizations and draw insights from the different features.

Python's built-in [datetime modeule](https://docs.python.org/3/library/datetime.html) helps in working with dates and times. Here, I used this modoule to convert the Date column to DatetimeIndex data type. The most fundamental of date/time objects are the *Timestamp** and *DatetimeIndex* objects. I used the ***pd.to_datetime() function***, to parse Date columns. This converted the format from object to *datetime64* format, encoding dates as 64-bit integers, and thus allowing an arrays of dates to be represented very compactly. 

![image](https://user-images.githubusercontent.com/50409210/151670939-1b9a3c28-ef20-4100-9251-32a672580067.png)

As the Amazon stock dataset had no missing values in any columns no imputations were needed. We used different instance attributes in the datetime module like ***.year() or .month_name()*** to strip or parse the date column features and create separate columns from them like for Year, Month Name, Day Name etc. This would be used in further EDA analysis on the dataset.

![image](https://user-images.githubusercontent.com/50409210/151671634-deddae70-1374-4fc2-915e-25e6093f6d6c.png)


### Exploratory Data Analysis (EDA) of Time Series -

Now, that we had our dataset cleaned and formatted we carried out some basic Exploratory data analysis on our Amazon stock dataset.

* Amazon's Closing (adusted) Price show very steep rise during the period after 2016 however, its Volumes have declined from 2000 or 2008 levels when they were at peak.
  
  ![image](https://user-images.githubusercontent.com/50409210/151713881-263a10c8-8939-4e48-aec2-832cf1d612c3.png)
 
* Both the 50-day and 200-day Moving Average for Amazon shows a fairly smoothed but rising in Daily adjusted closing price.

  ![image](https://user-images.githubusercontent.com/50409210/151674901-950f0f48-fbb4-4896-8d28-4db07a7f28dc.png)
  
* The movements in Amazon stock have reduced over time as we can see small percentage change (increase or decrease) in stock value 2016 onwards compared to that in the 2000s.
  
  ![image](https://user-images.githubusercontent.com/50409210/151675037-0ffa48a1-21ce-4aec-9cd7-9c4d464e8841.png)
  
* Amazon stock volumes appear to be the largest on Wednesday and Thursday while almost Nil on weekends (Saturday, Sunday). This is due to no trading on weekends. Also the January appears to be the highest month for trades in Amazon stocks followed by July, october and November compared to August which has the lowest. 

  ![image](https://user-images.githubusercontent.com/50409210/151675266-0d9c80ca-6921-4996-bb40-9f18f598ed5f.png)
  
 * The Close-High feature shows the maximum positive correlation with the Volume feature. We can say that if the Amazon Closing price stays away from High value, it may lead to more transactions or trading volumes that day for the stock. However, many other factors would also be influencing the trading volumes for a stock per day.
 
  ![image](https://user-images.githubusercontent.com/50409210/151675419-4490f38d-652b-47e7-b8be-9d6dab70c664.png)


Before we move on to Time Series Analysis and Forecasting techniques let me explain some key concepts about time series analysis and time-related data in general.

Key Concepts about Time-specific Data -

* **TimeStamp** references a particular moment in time (e.g., January 4th, 2022 at 7:00am). **Time Intervals and Periods** reference a length of time between a particular beginning and end point. **Time deltas or durations** refer to an exact length of time, between two dates or times.
* **Time series** is a set of data points that occur over a span or succesive periods of time like Years, Months, Weeks, Days, Hours, Minutes, and Seconds. Time series differs from **Cross-Sectional Data** as the latter considers only a single point in time while the former looks at a series or repeated samples of data over a time period. 
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

 
### Slicing Amazon Stock Data for Forecasting - 

* For ease of Stock Forecasting, henceforth we would only be using Amazon Stock price data from the Year **2015 onwards**. 
* Also, we would be using only the **Adjusted Close** and **Date** columns of our Amazon time series for training different Stock Forecasting techniques or algorithms and making predictions. 


### Testing for Stationarity - 

Time Series analysis only works with stationary data therefore, we need to determine if our Amazon stock data is stationary or not. A dataset is stationary if its statistical properties like mean, variance, and autocorrelation do not change over time. We need to make the time-series stationary before fitting any model.

Testing for Stationarity can be done using one or more of the following methods:-
* **Rolling Statistics** - One of the more popular rolling statistics is the moving average. This takes a moving window of time, and calculates the average or the mean of that time period as the current value. Visualizing the rolling statistics for our time series will help us determine stationarity. A series becomes stationary if both the mean and standard deviation are flat lines (constant mean and constant variance).  

We needed to check this for our Amazon time series as well so calculated and visualized the mean and standard deviation of Adjusted Close price for a 12-day window. Moving Standard Deviation is a statistical measurement of market volatility. The rolling standard deviation plot appears to be quite smooth in the beginning (2015-17 end) but becomes little volatile after 2018. In 2019 and end of 2020 there are some large movements. The rolling mean fits the original data for Amazon Adjusted Close quite closely.

  ![image](https://user-images.githubusercontent.com/50409210/151713977-fef5d7b1-ab0f-4b65-a2d4-6f592e99a3dd.png)

From the plots above, we found that there was an increasing mean and standard deviation, indicating that our series was not stationary. However, to be more sure about stationarity or not for our Amazon time series we used the ADF test method as well as explained below.


* **Augmented Dicky-Fuller (ADF) Test** - The ADF Test is one of the most common tests for stationarity and is based on the concept of unit root. Null Hypothesis (H0) for this test states that the time series is non-stationary due to trend. If the null hypothesis is not rejected, the series is said to be non-stationary. The result object of the ADF test is a tuple with following key elements:

     * Zeroth element - **Test statistic** (the more negative this number is, the more likely that data is stationary) 
     * Next element - **p-value** (if p-value < 0.05, we reject the H0 and assume our time series to be stationary)
     * Last element - **a Python Dictionary with critical values** of the test statistic equating to different p-values

We performed an ADF test for Amazon stock data and ran a test for different statistical significance to determine whether it was staionary or not. We got the following test results:-

![image](https://user-images.githubusercontent.com/50409210/151357493-7c6aa93b-99ab-4495-ba45-5d8d0be35a6a.png)

We can see from the results above that our Test statistic was a positive value and our p-value > 0.05.  Additionally, the test statistics exceeded the critical values hence, the data was nonlinear. We did not reject our null hypothesis, and realized that Amazon time series was non-stationary. 

Thus, by using both methods, plotting the rolling statistics and analyzing ADF test results we determined that our **Amazon time series was Non-Stationary**. Next, we had to work towards making our time series stationary before applying any Machine Learning modelling techniques.


 ### Making a Time Series Stationary  - 
 
 As we discussed earlier, to proceed with any time series analysis using models, we needed to stationarize our Amazon stock time series. 
 
 If the time series is non-stationary the below methods can be used to make them stationary:-
 
 * **De-trending the time series** - This method removes the underlying trend in the time series by standardizing it. It subtracts the mean and divides the result by the standard deviation of the data sample. This has the effect of transforming the data to have mean of zero, or centered, with a standard deviation of 1. Then the ADF test is performed on the de-trended time series to confirm the results.

The ADF test results on de-trended stock data for Amazon below, shows a p=value < 0.05 and also a negative Test statistic, which means that the Amazon time series has become stationary now.

![image](https://user-images.githubusercontent.com/50409210/151385463-0e6bc76f-8005-4ace-9cd2-e7bbdaf22a79.png)

On plotting the rolling statistics for the de-trended Amazon stock (see below) further shows that the time series has become stationary. This is indicated by the relative smoothness of the rolling mean and rolling standard deviation compared to the original De-trended Amazon data.

![image](https://user-images.githubusercontent.com/50409210/151700801-3c67db0b-a6d2-489e-a97d-bdb34c41e778.png)

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

By looking at the EDA for Amazon time series we could not be very sure about seasonality in the data. This can only be found if we decompose our time series. So, next we look into seasonal elements in our dataset.

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

As we know that there is some seasonality in our Amazon time series we would use seasonal orders also while fitting different algorithms to our dataset. But before that we first need to select optimal orders for our dataset both seasonal and non-seasonal.


### Selection of Non-seasonal and Seasonal Orders -

Model orders - 

When fitting and working with AR, MA, ARMA or SARIMA models, it is very important to understand the model order. We need to pick the most optimal model order before fitting our time series to a model inorder to make better predictions. 

* When we set either p or q to zero, we get a simpler AR or MA model.
* There are three important orders in ARIMA:

  * **Autoregressive order (p)** - Past values used for forecasting the next value or number of lag observations included in model (lag order).
  * **Order of differencing (d)** - If d = 0, we simply have an ARMA model.
  * **Moving Average order (q)** - Size of the moving average window.
  
* SARIMA or seasonal ARIMA model is a combination of non-seasonal orders and seasonal orders. Both these parts have orders for the autoregressive, difference and moving average parts.
  * **Non-seasonal orders** - autoregressive order (p), order of differencing (d) and moving average order (q)
  * **Seasonal orders** - autoregressive order (P), order of differencing (D), moving average order (Q) and a new order (S), which is the length of the seasonal cycle.

Choosing Model Orders - 

* **Using ACF and PACF plots** - One of the main ways to identify the correct model order is by using the Autocorrelation Function (ACF) and the Partial Autocorrelation Function (PACF). By comparing the ACF and PACF for a time series we can deduce the model order. The time series must be made stationary before making these plots.
* **Using AIC and BIC values** - The Akaike Information Criterion (AIC), is a metric which tells us how good a model is. The Bayesian Information Criterion (BIC), is very similar to the AIC. Models which fit the data better have lower AICs and BICs, while BIC penalizes overly complex models. Mostly, both AIC and BIC will choose the same model. These can be found on the right side of the summary of the fitted-models-results object or by using the ***.aic attribute*** and the ***.bic attribute***.


a) Manual Selection of Orders - 

For our Amazon time series, we manually tried to select orders by writing for loops. We tried to fit models with multiple order combinations, extracted the AIC and BIC values for each combination of orders and then selected the order yielding the lowest AIC value. We found that the BIC score for the same order also turned out to be the lowest in almost every case. 

We used this same process to select the following order types:- 

* **Finding only the optimal Non-Seasonal Orders (p,d,q)** - This resulted in giving us the lowest AIC and BIC values for our Amazon stock data by setting the order values to **p=2, d=1 and q=2**. 
* **Finding only the optimal Seasonal Orders (P,D,Q)** - We already pre-defined our length of seasonal cycle (S) as 7, as the seasonlity of our Amazon time series appeared to be daily, We also pre-defined our non-seasonal orders to be p=2, d=1, q=2 (results from our last order selection). From the results, lowest AIC and BIC values were yielded by setting the seasonal order values to **P=0, D=1 and Q=2**.
* **Finding both Non-seasonal (p,d, q) and Seasonal (P,D,Q) Orders** - Here, we had already pre-defined our length of seasonal cycle or S = 7, d = 1 and D = 1. It resulted in yielding the folowing:- 

![image](https://user-images.githubusercontent.com/50409210/151419135-8f5b94c8-3fa2-4c20-9a8f-1d938fbe943d.png)

  * Non-Seasonal Orders - p = 0, d = 1 and q = 1
  * Seasonal Orders - P = 0, D = 1 and Q = 1


b) Automated Selection of Orders - 

The [pmdarima package](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html), automatically discovers the optimal order for an ARIMA model. The ***auto_arima function*** from this package loops over model orders to find the best one. We used this method also to choose our most optimal model orders. he auto-arima function has a lot of parameters that we may want to set. Many of these have default values and the only required argument to the function is the data. Optionally we can also set the order of non-seasonal differencing; initial estimates of the non-seasonal orders; and the maximum values of non-seasonal orders to test.

* We set the ***period for seasonal differencing*** as Daily or m=7 for our Amazon time series.
* We specified the ***seasonal parameter***=True as our time series appeared to be seasonal in nature.
* The function performs differencing tests (e.g., Kwiatkowski–Phillips–Schmidt–Shin, Augmented Dickey-Fuller, or Phillips–Perron) to determine the order of differencing, d. Here, we specified the ****test argument***='adf' to specify the use of Augmented Dickey-Fuller test.
* We can also specify fitting models within start p, max p, start q, max q ranges. Here, we specified only the ***max_p*** and ***max_q*** parameters as 2 for our model.
* After determining the optimal order of seasonal differencing, D, auto_arima also seeks to identify the optimal P and Q hyper-parameters if the seasonal option (i.e., seasonal=True) is enabled.
* The object returned by the function is the results object of the best model found by the search.

![image](https://user-images.githubusercontent.com/50409210/151422371-39e053bb-e78b-47bd-be95-6e04df35011e.png)

As shown above, the Auto ARIMA model assigned the following order values to our Amazon time series data:-
   * Non-Seasonal Orders - 2, 1 and 2 to p, d, and q, respectively.
   * Seasonal Orders - 0, 0, 2 and to P, D, Q respectively.


Thus, by using both manual selection and automatic selection of orders we could find the range of values suitable for our Amazon time series. Below is a snapshot of values yielded for respective orders by different selection methods. 

![image](https://user-images.githubusercontent.com/50409210/151420930-061cb8d6-bff2-4d6b-9053-bef526f0b870.png)



### Comparing Models and Interpreting Results - 

Interpreting Model Summary:

Interpreting results from a machine learning algorithm can be a trying experience. An important component of statsmodel library is that we can inspect the results from the fitted model using a ***.summary() method***. 

![image](https://user-images.githubusercontent.com/50409210/151540717-d2910469-d224-4e7a-b0e9-a89cac66d1d3.png)

Let us walkthrough the summary result components:-
* **General Information** - The top section includes useful information such as the order of the model that we fit, the number of observations or data points, the name of the time series. Statsmodels uses the same module for all of the autoregressive models, therefore, the header displays **SARIMAX Results** even for an AR model.
* **Statistical Significance** - The next section of the summary shows the fitted model parameters, like the ar.L1 and ar.L2 rows for e.g., if fitting an ARMA(2,1) model having AR-lag-1 and lag-2 coefficients. Similarly, the MA coefficients are in the last rows. The first column shows the model coefficients whilst the second column shows the standard error in these coefficients. This is the uncertainty on the fitted coefficient values. We want each term to have a **p-value < 0.05**, so we can reject the null hypothesis with values that are statistically significant.
* **Assumption Review** - 
  * *Ljung Box (Q)* test estimates that the errors are white noise. Since its probability Prob(Q) is above 0.05, we can’t reject the null that the errors are white noise.
  * *Heteroscedasticity (H)* tests that the error residuals have the same variance. Our summary statistics shows  a p-value of 0.00, which means we reject the null hypothesis and our residuals show variance. 
  * *Jarque-Bera (JB)* tests for the normality of errors. We see a test statistic with a probability of 0, which means we reject the null hypothesis, and the data is not normally distributed. 
  * We also see that the distribution has a slight positive skewness and a large kurtosis.
* **Fit Analysis** - Values in this section of the summary like the Log-Likelihood, AIC, BIC, and HQIC help compare one model with another. AIC penalizes a model for adding parameters and BIC alongwith AIC penalize complex models. Lower the values for these indicators better is the fit of the model on the distribution of data.



Interpreting Plot Diagnostics:

For an ideal model the residuals should be uncorrelated white Gaussian noise centered to zero. We can use the results object's ***.plot_diagnostics method*** to generate four common plots for evaluating this. This 4-plot is a convenient graphical technique for model validation and consists of the following set of plots:-

![image](https://user-images.githubusercontent.com/50409210/151553656-6dd826ed-e8df-4afd-975e-f305f34eafe8.png)

* **Residuals Plot** - This plot shows the one-step-ahead standardized residuals. For good fitted model there should be no obvious structure in the residuals. 
* **Histogram plus KDE Estimate Plot** - This plot shows us the distribution of the residuals and tests for (normal) distribution. The orange line shows a smoothed version of this histogram and the green line, shows a normal distribution. For a good model both the green and orange lines should be as close as possible.
* **Normal Normal Q-Q Plot** - This one also also compares the distribution of the model residuals to normal distribution. For residuals to be normally distributed all the points should lie along the red line, except some values at either end.
* **Correlogram** - This is an ACF plot of the residuals wherein 95% of the correlations for lag greater than zero should not be significant. If there is significant correlation in the residuals, it means there is some information in the data not captured by our model.

![image](https://user-images.githubusercontent.com/50409210/151554111-41043315-8bb7-4670-8f44-97e3d613511a.png)

Model results from each of the models selected and used for Amazon time series were interpreted using summary results and plot diagnostics methods as discussed above.


### Splitting the Dataset for Time Series Analysis - 

Splitting the dataset is an important exerise before selecting and fitting machine learning algorithms on any dataset. For time series analysis, the split in dataset would be slightly different from other cases where we try to split datasets randomly into train-test subsets. Here, as we would be using past values to make future predictions hence, we would be splitting the data in relation to time i.e., training our algorithms on data coming earlier in time series and testing on data that comes later. 

![image](https://user-images.githubusercontent.com/50409210/151556040-343ff83f-f9d1-455f-9156-e0c88452bc5d.png)

As we can see from the plot above, we have split our Amazon time series also into training and testing subsets in the ratio of 80:20. The plot depicts how data earlier in time would be used for training the algorithms while latest data would be used for making forecasts.


### Model Selection for Stock Predictions -

There are different kind of time series analysis techniques with the most common ones like the following -
* Autoregression (AR)
* Moving Average (MA)
* Autoregressive Moving Average (ARMA)
* Autoregressive Integrated Moving Average (ARIMA)
* Seasonal Autoregressive Integrated Moving-Average (SARIMA)

**Autoregressive (AR), Integrated (I), Moving Average(MA)** are combined to form **Autoregressive Moving Average (ARMA), and Autoregressive Integrated Moving Average (ARIMA)** models. SARIMA combines the ARIMA model with the ability to perform the same autoregression, differencing, and moving average modeling at the seasonal level. A SARIMA model can be used to develop AR, MA, ARMA and ARIMA models. 

The **Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)** is an extension of the SARIMA model that also includes the modeling of exogenous variables. These are parallel input sequences having observations at the same time steps as the original series.

Some deep learning-based techniques include **Long-short term memory(LSTM)**.


For analyzing Amazon stock time series I used the following Time Series Analysis techniques:-

**Autoregressive Integrated Moving Average (ARIMA)** - 
  
  * It is used to predict the future values of a time series using its past values and forecast errors.
  * Very popular statistical method for time series forecasting and capable of predicting short-term share market movements.
  * We can also implement an ARIMA model using the SARIMAX model class from statsmodels.
   * ARIMA model has three model orders - p the autoregressive order; d the order of differencing; and q the moving average order
  
Manual selection of model orders on Amazon time series yielded lowest AIC and BIC values by setting the non-seasonal order values to **p=2, d=1 and q=2**. These were used to fit an ARIMA(2,1,2) model on our training subset. The folowing summary results and residual diagnostic plots were outputed from this model.
  
![image](https://user-images.githubusercontent.com/50409210/151714219-823c3192-af13-4a04-a20b-81e6c47628e7.png)


The JB p-value or Prob(JB) is zero, which means we should reject the null hypothesis that the residuals are normally distributed.

![image](https://user-images.githubusercontent.com/50409210/151714232-67a5c4d4-13de-4dff-a441-c839bcc6fee3.png)

There are some obvious patterns in the residuals plot towards the right end of the plot. The KDE curve is not very similar to the normal distribution plot. Most of the data points do not lie on the red line as shown by the histogram and Q-Q plots. In the last plot is the correlogram, which is just an ACF plot of the residuals 95% of the correlations for lag greater than zero should not be significant and they appear to be not significant here as well.

Both these output interpretations above, suggest that our ARiMA(2,1,2) model does not fit our Amazon stock data too well yet.


**Seasonal Autoregressive Integrated Moving-Average (SARIMA)** -

The decomposition plot for our Amazon data suggested that there was some seasonality in the time series. This prompted us to use the SARIMA techniques next. The notation for SARIMA model involves specifying the order for the AR(p), I(d), and MA(q) models as parameters to an ARIMA function and AR(P), I(D), MA(Q) and m parameters at the seasonal level, e.g. SARIMA(p, d, q)(P, D, Q)m where “m” is the number of time steps in each season (the seasonal period). 

Although, manual selection of orders yielded SARIMA(0,1,1)(0,1,1,7) for Amazon time series instead of using these parameters to fit our model we fittied the SARIMA model, using auto_arima function of the pmdarima package.  As we can see below, the most optimal orders selected from here were **p=2, d=1, q=2, P=0, D=0, Q=2, S or m=7**.

![image](https://user-images.githubusercontent.com/50409210/151714263-549e124f-20e6-4cff-8395-f2eaf0e984d4.png)

![image](https://user-images.githubusercontent.com/50409210/151714253-257db7f0-40e0-4684-a30a-901639b12ed1.png)

The summary result above shows, that the model does not meet the condition of no correlation (independence in the residuals) because the p-value of the Ljung-Box test Prob(Q) is greater than 0.05, so we cannot reject the null hypothesis of independence. Also, we cannot say that the residual distribution is having constant variance (homoscedastic) because the p-value of the Heteroskedasticity test Prob(H) is smaller than 0.05.

![image](https://user-images.githubusercontent.com/50409210/151714282-ce214044-23c7-4bc9-8948-df92674e6a4a.png)

There appears to be very little difference in the residual plot from SARIMA as compared to that from ARIMA.



### Making Predictions from Most Optimal Model on Unseen Data (testing subset) - 

Although, training both **ARIMA(2,1,2)** and **SARIMA(2,1,2)(0,0,2,7)** on the training subset of Amazon series yielded very similar summary results and diagnostic plots, the SARIMA model seems to be a better fit for our dataset because it has seasonal orders hence, takes into account some seasonality in our data.

Next, we used the **SARIMA(2,1,2)(0,0,2,7)**, to fit to the training set and make predictions on the testing set i.e., our unseen time series for Amazon. 

![image](https://user-images.githubusercontent.com/50409210/151714737-2a297e0a-12aa-497c-adb1-6224e2d6f2b3.png)

From the above plot, we find that this model SARIMA(2,1,2)(0,0,2,7) performs fairly well in making prediction about Amazon time series testing subset. Now we would put this model into practice to make future or out-of-sample forecasts as well.


**Prophet** - 

Lastly, I used the open-source [Prophet](https://github.com/facebook/prophet) algorithm developed by Facebook’s Core Data Science team. It is a third-party time series forecasting library which requires almost little data preprocessing and is very simple to implement. 

  * Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality. 
  * It considers the effects of holiday quite well and is capable of handling seasonality-related components in data by the use of simple parameters to fine-tune the model like     specifying holidays, daily seasonality etc
  * The input for Prophet is a dataframe with two columns, a Date and a target variable column named as - **ds** and **y** respectively.

  ![image](https://user-images.githubusercontent.com/50409210/151583807-582fe6b6-159f-4836-97d9-e7a6cd58d94c.png)
  
   While working with the Prophet algorithm for Amazon time series we undertook the following series of teps:-

   *  Firstly, we created a **Prophet instance** by specifying the Daily seasonality effects for our time series using the ***daily_seasonality=True parameter***. 
   *  After this we trained the model and made predictions for 1 year time in future by specifying the ***period parameter=365***. This yielded a DataFrame of forecasts with key       parameters as columns ("ds” indicating the Date, “yhat” indicating the predicted time series data, “yhat_lower” and “yhat_upper” indicating the probable lower and upper         limit of forecasts).
  
     ![image](https://user-images.githubusercontent.com/50409210/151591087-606e2804-441b-48b2-96af-9768c4f5eeb7.png)
  
  *  Prophet also allowed us to plot different components from our forecasts using the ***plot_components parameter*** thereby showing the trend, yearly weekly and daily plots.
  *  Adding Changepoints is another useful feature in Prophet, as they allow one to put more emphasis and find reasons for changes in trajectory or trends in data. Changepoints      are the datetime points where the time series have abrupt changes in the trajectory. By default, Prophet adds 25 changepoints to the initial 80% of the dataset.

     ![image](https://user-images.githubusercontent.com/50409210/151587907-a0c69115-441c-4f6b-a310-4f190404199f.png)



### Time Series Forecasting -

Time series forecasting is the use of a model to predict future values based on previously observed values.

**Generating In-Sample (One-Step Ahead) Predictions** - 
  
This forecasting technique allows us to evaluate how good our model is at predicting just one value ahead. 

  * We can use the SARIMAX fitted results object's ***get_prediction() method*** to generate in-sample predictions. 
  * We can set the ***start parameter*** as a negative integer stating how many steps back to begin the forecast. For the Amazon time series, we set the start parameter to -30 as we wanted to make predictions for the last 30 time-periods of the Amazon data. It returns a forecast object. 
  * The central value of the forecast is extracted and stored in the ****predicted_mean attribute*** of the forecast object and can be used alongwith the lower and upper confidence limits to plot forecasts. 
  * Here, the mean prediction is marked with a red line while the uncertainty range is shaded. The uncertainty is due to the random shock terms that we can't predict.

![image](https://user-images.githubusercontent.com/50409210/151714373-5c16c06c-ac16-4c44-880a-4c628355770f.png)


**Generating Dynamic Forecasts** - 

We can make predictions further than just one step ahead by using dynamic prediction technique. It first predicts one step ahead, and then use this predicted value to forecast the next value after that. 
  * Here, we set the  ***dynamic parameter=True*** in addition to the steps for making one-step ahead predictions. 
  * Making dynamic predictions, implies that the model makes predictions with no corrections for a future period of time, unlike making one-step-ahead predictions. Once the predictions are made for the period, we can compare predictions with actual values.

![image](https://user-images.githubusercontent.com/50409210/151714390-efd27b09-6b4f-46d6-b659-bc303f241c74.png)

For our dataset, here too we set the start parameter to -30 as we wanted to make dynamic predictions for the last 30 time-periods of the Amazon data.


**Making Out-of-Sample Forecasts** - 

Finally, after testing our predictions in-sample, we can use our model to predict the future. To make future forecasts we use the ***get_forecast method*** of the results object. 
* We choose the number of steps after the end of the training data to forecast up to by specifying the ***steps parameter***. For making forecasts using Amazon data for the next we used steps=182 in order to forecast for approximately 6 months into the future.

![image](https://user-images.githubusercontent.com/50409210/151714446-e17166a6-2396-4a92-a2d1-0119448ce2a3.png)

![image](https://user-images.githubusercontent.com/50409210/151714422-62599fa3-a7b7-4c22-82b4-25136bfaf75d.png)




#### References and Sources -

* https://jakevdp.github.io/PythonDataScienceHandbook/03.11-working-with-time-series.html
* https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
* https://analyzingalpha.com/interpret-arima-results

