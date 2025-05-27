# Stock Market Analysis for Tech Stocks

In this project, we'll analyse data from the stock market for some technology stocks.

 Again, we'll use Pandas to extract and analyse the information, visualise it, and look at different ways to analyse the risk of a stock, based on its performance history.

Here are the questions we'll try to answer:

- What was the change in a stock's price over time?
- What was the daily return average of a stock?
- What was the moving average of various stocks?
- What was the correlation between daily returns of different stocks?
- How much value do we put at risk by investing in a particular stock?
- How can we attempt to predict future stock behaviour?

```python
#Python Data Analysis imports
import pandas as pd
from pandas import Series,DataFrame
import numpy as np

#Visualisation imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

#To grab stock data
from pandas_datareader import DataReader
from datetime import datetime
```

We're going to analyse some tech stocks, and it seems like a good idea to look at their performance over the last year. We can create a list with the stock names, for future looping.

```python
#We're going to analyse stock info for Apple, Google, Microsoft, and Amazon
tech_list = ['AAPL','GOOGL','MSFT','AMZN']
```

```python
#Setting the end date to today
end = datetime.now()

#Start date set to 1 year back
start = datetime(end.year-1,end.month,end.day)
```

```python
file_paths = ['AAPL.csv', 'GOOGL.csv', 'MSFT.csv','AMZN.csv']  # Replace with actual file paths

# Create a dictionary to store the DataFrames
stock_data = {}

# Loop through each file and read it into a DataFrame
for stock, file_path in zip(tech_list, file_paths):
    stock_data[stock] = pd.read_csv(file_path)

# Now you can access each DataFrame using the stock symbol as the key in the stock_data dictionary
# For example, to access the DataFrame for AAPL:
AAPL = stock_data['AAPL']
GOOGL = stock_data['GOOGL']
MSFT = stock_data['MSFT']
AMZN = stock_data['AMZN']
```

Thanks to the globals method, Apple's stock data will be stored in the AAPL global variable dataframe. Let's see if that worked.

```python
AAPL.head()
```

```python
#Basic stats for Apple's Stock
AAPL.describe()
```

And that easily, we can make out what the stock's minimum, maximum, and average price was for the last year.

```python
#Some basic info about the dataframe
AAPL.info()
```

No missing info in the dataframe above, so we can go about our business.

## What's the change in stock's price over time?

```python
#Plotting the stock's adjusted closing price using pandas
AAPL['Adj Close'].plot(legend=True,figsize=(9,3))
```

Similarily, we can plot change in a stock's volume being traded, over time.

```python
#Plotting the total volume being traded over time
AAPL['Volume'].plot(legend=True,figsize=(9,3))
```

### What was the moving average of various stocks?

Let's check out the moving average for stocks over a 10, 20 and 50 day period of time. We'll add that information to the stock's dataframe.

```python
ma_day = [10,20,50]

for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))
    
    AAPL[column_name] = AAPL['Adj Close'].rolling(window=ma,center=False).mean()
```

```python
AAPL.tail()
```

```python
AAPL[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(subplots=False,figsize=(9,3))
```

Moving averages for more days have a smoother plot, as they're less reliable on daily fluctuations. So even though, Apple's stock has a slight dip near the start of September, it's generally been on an upward trend since mid-July.

### What was the daily return average of a stock?

```python
#The daily return column can be created by using the percentage change over the adjusted closing price
AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
```

```python
AAPL['Daily Return'].tail()
```

```python
#Plotting the daily return
AAPL['Daily Return'].plot(figsize=(9,3),legend=True,linestyle='--',marker='o')
```

```python
sns.histplot(x=AAPL['Daily Return'].dropna(),bins=100,color='red')
```

Positive daily returns seem to be slightly more frequent than negative returns for Apple.

### What was the correlation between daily returns of different stocks?

```python
#Reading just the 'Adj Close' column this time
#close_df = DataReader(tech_list,'file_paths',start,end)['Adj Close']
```

```python
adj_close_data = {}

# Loop through each file and read just the 'Adj Close' column into a DataFrame
for stock, file_path in zip(tech_list, file_paths):
    df = pd.read_csv(file_path, usecols=['Date', 'Adj Close'])  # Read only 'Date' and 'Adj Close' columns
    adj_close_data[stock] = df.set_index('Date')  # Set 'Date' column as index

# Now you can access each DataFrame containing only the 'Adj Close' data 
# using the stock symbol as the key in the adj_close_data dictionary
# For example, to access the 'Adj Close' data for AAPL:
aapl_adj_close = adj_close_data['AAPL']
googl_adj_close = adj_close_data['GOOGL']
msft_adj_close = adj_close_data['MSFT']
amzn_adj_close = adj_close_data['AMZN']
```

```python
close_df = pd.concat([aapl_adj_close, googl_adj_close, msft_adj_close, amzn_adj_close], axis=1)

# Rename the columns to indicate the corresponding stock symbols
close_df.columns = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']

# Print the combined DataFrame
print(close_df)
```

```python
close_df.tail()
```

Everything works as expected.

Just as we did earlier, we can use Pandas' pct_change method to get the daily returns of our stocks.

```python
rets_df = close_df.pct_change()
```

```python
rets_df.tail()
```

Let's try creating a scatterplot to visualise any correlations between different stocks. First we'll visualise a scatterplot for the relationship between the daily return of a stock to itself.

```python
# Assuming 'GOOGL' is a column in the DataFrame rets_df
sns.jointplot(x='GOOGL', y='GOOGL', data=rets_df, kind='scatter', color='green')
```

As expected, the relationship is perfectly linear because we're trying to correlate something with itself. Now, let's check out the relationship between Google and Apple's daily returns.

```python
# Assuming 'GOOGL' and 'AAPL' are columns in the DataFrame rets_df
sns.jointplot(x='GOOGL', y='AAPL', data=rets_df, kind='scatter')
```

There seems to be a minor correlation between the two stocks, looking at the figure above. The Pearson R Correlation Coefficient value of 0.45 echoes that sentiment.

But what about other combinations of stocks?

```python
sns.pairplot(rets_df.dropna())
```

Quick and dirty overarching visualisation of the scatterplots and histograms of daily returns of our stocks. To see the actual numbers for the correlation coefficients, we can use seaborn's corrplot method.

```python
sns.heatmap(rets_df.dropna(),annot=True)
```

Google and Microsoft seem to have the highest correlation. But another interesting thing to note is that all tech companies that we explored are positively correlated.

### How much value do we put at risk by investing in a particular stock?

A basic way to quantify risk is to compare the expected return (which can be the mean of the stock's daily returns) with the standard deviation of the daily returns.

```python
rets = rets_df.dropna()
```

```python
plt.figure(figsize=(8,5))

plt.scatter(rets.mean(),rets.std(),s=25)

plt.xlabel('Expected Return')
plt.ylabel('Risk')


#For adding annotatios in the scatterplot
for label,x,y in zip(rets.columns,rets.mean(),rets.std()):
    plt.annotate(
    label,
    xy=(x,y),xytext=(-120,20),
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    arrowprops = dict(arrowstyle='->',connectionstyle = 'arc3,rad=-0.5'))
```

We'd want a stock to have a high expected return and a low risk; Google and Microsoft seem to be the safe options for that. Meanwhile, Yahoo and Amazon stocks have higher expected returns, but also have a higher risk.

### Value at Risk

We can treat Value at risk as the amount of money we could expect to lose for a given confidence interval. We'll use the 'Bootstrap' method and the 'Monte Carlo Method' to extract this value.

#### Bootstrap Method

Using this method, we calculate the empirical quantiles from a histogram of daily returns. The quantiles help us define our confidence interval.

```python
AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()

# Plot histogram of daily returns
sns.histplot(x=AAPL['Daily Return'].dropna(), bins=100, color='purple')
```

To recap, our histogram for Apple's stock looked like the above. And our daily returns dataframe looked like:

```python
rets.head()
```

```python
#Using Pandas built in qualtile method
rets['AAPL'].quantile(0.05)
```

The 0.05 empirical quantile of daily returns is at -0.019. This means that with 95% confidence, the worst daily loss will not exceed 2.57% (of the investment).

### How can we attempt to predict future stock behaviour?

#### Monte Carlo Method

Check out this [link](https://www.investopedia.com/articles/07/montecarlo.asp) for more info on the Monte Carlo method. In short: in this method, we run simulations to predict the future many times, and aggregate the results in the end for some quantifiable value.

```python
days = 365

#delta t
dt = 1/365

mu = rets.mean()['GOOGL']

sigma = rets.std()['GOOGL']
```

```python
#Function takes in stock price, number of days to run, mean and standard deviation values
def stock_monte_carlo(start_price,days,mu,sigma):
    
    price = np.zeros(days)
    price[0] = start_price
    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in range(1,days):
        
        #Shock and drift formulas taken from the Monte Carlo formula
        shock[x] = np.random.normal(loc=mu*dt,scale=sigma*np.sqrt(dt))
        
        drift[x] = mu * dt
        
        #New price = Old price + Old price*(shock+drift)
        price[x] = price[x-1] + (price[x-1] * (drift[x]+shock[x]))
        
    return price
```

We're going to run the simulation of Google stocks. Let's check out the opening value of the stock.

```python
GOOGL.head()
```

Let's do a simulation of 100 runs, and plot them.

```python
start_price = 622.049 #Taken from above

for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))

plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for Google')
```

```python
runs = 10000

simulations = np.zeros(runs)

for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]
```

```python
q = np.percentile(simulations,1)

plt.hist(simulations,bins=200)

plt.figtext(0.6,0.8,s="Start price: $%.2f" %start_price)

plt.figtext(0.6,0.7,"Mean final price: $%.2f" % simulations.mean())

plt.figtext(0.6,0.6,"VaR(0.99): $%.2f" % (start_price -q,))

plt.figtext(0.15,0.6, "q(0.99): $%.2f" % q)

plt.axvline(x=q, linewidth=4, color='r')

plt.title(u"Final price distribution for Google Stock after %s days" %days, weight='bold')
```

We can infer from this that, Google's stock is pretty stable. The starting price that we had was USD622.05, and the average final price over 10,000 runs was USD623.36.

The red line indicates the value of stock at risk at the desired confidence interval. For every stock, we'd be risking USD18.38, 99% of the time.

```python
import statsmodels.api as sm

# Assuming you have separate DataFrames for AAPL, MSFT, AMZN, and GOOGL
# Define the features (predictors) and the target variable for each stock
X_aapl = AAPL[['Open', 'High', 'Low', 'Volume']]  # Features for AAPL
y_aapl = AAPL['Close']  # Target variable for AAPL

X_msft = MSFT[['Open', 'High', 'Low', 'Volume']]  # Features for MSFT
y_msft = MSFT['Close']  # Target variable for MSFT

X_amzn = AMZN[['Open', 'High', 'Low', 'Volume']]  # Features for AMZN
y_amzn = AMZN['Close']  # Target variable for AMZN

X_googl = GOOGL[['Open', 'High', 'Low', 'Volume']]  # Features for GOOGL
y_googl = GOOGL['Close']  # Target variable for GOOGL

# Fit multiple regression models for each stock
model_aapl = sm.OLS(y_aapl, sm.add_constant(X_aapl)).fit()
model_msft = sm.OLS(y_msft, sm.add_constant(X_msft)).fit()
model_amzn = sm.OLS(y_amzn, sm.add_constant(X_amzn)).fit()
model_googl = sm.OLS(y_googl, sm.add_constant(X_googl)).fit()

# Print summary of the regression models
print("AAPL Regression Summary:")
print(model_aapl.summary())
print("\nMSFT Regression Summary:")
print(model_msft.summary())
print("\nAMZN Regression Summary:")
print(model_amzn.summary())
print("\nGOOGL Regression Summary:")
print(model_googl.summary())
```

```python
import matplotlib.pyplot as plt

# Create subplots for each stock
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Observed vs. Predicted Close Price')

# Plot for AAPL
axes[0, 0].scatter(y_aapl, model_aapl.predict(sm.add_constant(X_aapl)))
axes[0, 0].set_title('AAPL')
axes[0, 0].set_xlabel('Observed Close Price')
axes[0, 0].set_ylabel('Predicted Close Price')
axes[0, 0].grid(True)

# Plot for MSFT
axes[0, 1].scatter(y_msft, model_msft.predict(sm.add_constant(X_msft)))
axes[0, 1].set_title('MSFT')
axes[0, 1].set_xlabel('Observed Close Price')
axes[0, 1].set_ylabel('Predicted Close Price')
axes[0, 1].grid(True)

# Plot for AMZN
axes[1, 0].scatter(y_amzn, model_amzn.predict(sm.add_constant(X_amzn)))
axes[1, 0].set_title('AMZN')
axes[1, 0].set_xlabel('Observed Close Price')
axes[1, 0].set_ylabel('Predicted Close Price')
axes[1, 0].grid(True)

# Plot for GOOGL
axes[1, 1].scatter(y_googl, model_googl.predict(sm.add_constant(X_googl)))
axes[1, 1].set_title('GOOGL')
axes[1, 1].set_xlabel('Observed Close Price')
axes[1, 1].set_ylabel('Predicted Close Price')
axes[1, 1].grid(True)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show the plot
plt.show()
```

Trend: The plots generally show a positive linear trend, indicating that the regression models capture relationships well.
    
Scatter Distribution: Points around the diagonal line suggest good fit but some variance in predicted close prices.

Accuracy: Predictions vary across stocks, with some showing closer alignment between observed and predicted close prices.

Outliers: Occasional deviations between observed and predicted close prices may occur due to exceptional market conditions or unaccounted factors.

```python
import matplotlib.pyplot as plt

# Plot observed vs. predicted values for all stocks
plt.figure(figsize=(9, 4))
plt.scatter(y_aapl, model_aapl.predict(sm.add_constant(X_aapl)), color='blue', label='AAPL')
plt.scatter(y_msft, model_msft.predict(sm.add_constant(X_msft)), color='red', label='MSFT')
plt.scatter(y_amzn, model_amzn.predict(sm.add_constant(X_amzn)), color='green', label='AMZN')
plt.scatter(y_googl, model_googl.predict(sm.add_constant(X_googl)), color='purple', label='GOOGL')

# Set labels and title
plt.xlabel('Observed Close Price')
plt.ylabel('Predicted Close Price')
plt.title('Observed vs. Predicted Close Price for Multiple Stocks')

# Add legend
plt.legend()

# Show grid
plt.grid(True)

# Show plot
plt.show()
```

We use plt.scatter() to plot the observed (actual) close prices (y) against the predicted close prices (model.predict(sm.add_constant(X))) for each stock.

