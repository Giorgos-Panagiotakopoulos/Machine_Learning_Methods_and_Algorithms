import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from sklearn.linear_model import ElasticNet, Lasso, Ridge

from sklearn.model_selection import KFold, train_test_split

import os

os.chdir("C:\\Users\\Giorgos\\Desktop\\ergasies_metaptxiakwn\\ergasia_magklogianni")

rossmann_df = pd.read_csv("train.csv")

store_df = pd.read_csv("store.csv")

test_df = pd.read_csv("test.csv")

# data information

rossmann_df.head()

rossmann_df.tail()

rossmann_df.info()

store_df.info()

test_df.info()

##############################################################################

######################      Data Management         ##########################

##############################################################################

# If Open is missing (in test_df) then Open=1 if DayOfWeek != 7 (Sunday)

test_df["Open"][test_df["Open"].isna()] = (test_df["DayOfWeek"] != 7).astype(int)

rossmann_df['Date'] = rossmann_df['Date'].apply(lambda x: (str(x)[:7]))

test_df['Date'] = test_df['Date'].apply(lambda x: (str(x)[:7]))

# group by date and get average sales, and precent change

average_sales = rossmann_df.groupby('Date')["Sales"].mean()

pct_change_sales = average_sales.pct_change()

# Create dummy varibales for DayOfWeek

day_dummies_rossmann = pd.get_dummies(rossmann_df['DayOfWeek'], prefix='Day')

day_dummies_rossmann.drop(['Day_7'], axis=1, inplace=True)

day_dummies_test = pd.get_dummies(test_df['DayOfWeek'], prefix='Day')

day_dummies_test.drop(['Day_7'], axis=1, inplace=True)

rossmann_df = rossmann_df.join(day_dummies_rossmann)

test_df = test_df.join(day_dummies_test)

# remove all rows(store,date) that were closed

rossmann_df = rossmann_df[rossmann_df["Open"] != 0]

# remove all rows(store,date) that were closed

test_df = test_df[test_df["Open"] != 0]

##############################################################################


##############################################################################

#################      Exploratory Data Analysis         #####################

##############################################################################


# Figure 1. Bar Chart for “Day of the Week” by “Open”

# Open

# Seaborn countplot

fig, (axis1) = plt.subplots(1, 1, figsize=(15, 4))

sns.countplot(x='Open', hue='DayOfWeek', data=rossmann_df, palette="husl", ax=axis1)

#

# fill NaN values in test_df with Open=1 if DayOfWeek != 7

test_df["Open"][test_df["Open"] != test_df["Open"]] = (test_df["DayOfWeek"] != 7).astype(int)
plt.show()
#

# Drop Open column

# rossmann_df.drop("Open", axis=1, inplace=True)

# test_df.drop("Open", axis=1, inplace=True)


# Figure 2. Time Plot for Average Sales and Percentage Change of the Sales over Time (year-month)

# Date

# Create Year and Month columns

rossmann_df['Year'] = rossmann_df['Date'].apply(lambda x: int(str(x)[:4]))

rossmann_df['Month'] = rossmann_df['Date'].apply(lambda x: int(str(x)[5:7]))

#

test_df['Year'] = test_df['Date'].apply(lambda x: int(str(x)[:4]))

test_df['Month'] = test_df['Date'].apply(lambda x: int(str(x)[5:7]))

#

# Assign Date column to Date(Year-Month) instead of (Year-Month-Day)

# this column will be useful in analysis and visualization

rossmann_df['Date'] = rossmann_df['Date'].apply(lambda x: (str(x)[:7]))

test_df['Date'] = test_df['Date'].apply(lambda x: (str(x)[:7]))

#

# group by date and get average sales, and precent change

average_sales = rossmann_df.groupby('Date')["Sales"].mean()

pct_change_sales = rossmann_df.groupby('Date')["Sales"].sum().pct_change()

#

fig, (axis1, axis2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8))

# Figure 2. Time Plot for Average Sales and Percentage Change of the Sales over Time (year-month)

# plot average sales over time(year-month)

ax1 = average_sales.plot(legend=True, ax=axis1, marker='o', title="Average Sales")

ax1.set_xticks(range(len(average_sales)))

ax1.set_xticklabels(average_sales.index.tolist(), rotation=90)

#

# plot precent change for sales over time(year-month)

ax2 = pct_change_sales.plot(legend=True, ax=axis2, marker='o', rot=90, colormap="summer", title="Sales Percent Change")

plt.show()

# Figure 3. Bar Charts, Number of Sales and Customers per year

# Plot average sales & customers for every year

fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))

#

sns.barplot(x='Year', y='Sales', data=rossmann_df, ax=axis1)

sns.barplot(x='Year', y='Customers', data=rossmann_df, ax=axis2)
plt.show()
#

# Drop Date column


# Figure 4. Time Plot and Box-plot for the Average Number of Customers over Time (year-month).

# Customers

fig, (axis1, axis2) = plt.subplots(2, 1, figsize=(15, 8))

# Plot max, min values, & 2nd, 3rd quartile


sns.boxplot(x="Customers", data=rossmann_df, ax=axis1)
# group by date and get average customers, and precent change

average_customers = rossmann_df.groupby('Date')["Customers"].mean()

pct_change_customers = rossmann_df.groupby('Date')["Customers"].sum().pct_change()

# Plot average customers over the time

# it should be correlated with the average sales over time

ax = average_customers.plot(legend=True, marker='o', ax=axis2)

ax.set_xticks(range(len(average_customers)))

xlabels = ax.set_xticklabels(average_customers.index.tolist(), rotation=90)

plt.show()

# Figure 5. Bar Charts, Average Sales and Customers per Day of the week.

# In both cases where the store is closed and opened
pd.set_option('display.max_columns', None)
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))

sns.barplot(x='DayOfWeek', y='Sales', data=rossmann_df, order=[1, 2, 3, 4, 5, 6, 7], ax=axis1)

sns.barplot(x='DayOfWeek', y='Customers', data=rossmann_df, order=[1, 2, 3, 4, 5, 6, 7], ax=axis2)
plt.show()

# Figure 6. Bar Charts for Average Sales and Customers per Promo.

# Promo

# Plot average sales & customers with/without promo

fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))

sns.barplot(x='Promo', y='Sales', data=rossmann_df, ax=axis1)

sns.barplot(x='Promo', y='Customers', data=rossmann_df, ax=axis2)
plt.show()
# Figure 7. Box-plot and Histogram for the Average Sales.

# Sales

fig, (axis1, axis2) = plt.subplots(2, 1, figsize=(15, 8))

# Plot max, min values, & 2nd, 3rd quartile



sns.boxplot(x="Sales", data=rossmann_df, ax=axis1)

# Plot sales values

# Notice that values with 0 is mostly because the store was closed

rossmann_df["Sales"].plot(kind='hist', bins=70, xlim=(0, 15000), ax=axis2)
plt.show()
# Using store_df

# Merge store_df with average store sales & customers

average_sales_customers = rossmann_df.groupby('Store')[["Sales", "Customers"]].mean()

average_sales_customers = average_sales_customers.reset_index(drop=False)

#

sales_customers_df = pd.DataFrame({'Store': average_sales_customers.index,

                                   'Sales': average_sales_customers["Sales"],

                                   'Customers': average_sales_customers["Customers"]},

                                  columns=['Store', 'Sales', 'Customers'])

sales_customers_df = sales_customers_df.reset_index(drop=True)

#

store_df = pd.merge(sales_customers_df, store_df, on='Store')

#

store_df.head()
plt.show()

# Figure 8. Bar Charts for Store Type.

# StoreType

# Plot StoreType, & StoreType Vs average sales and customers

# Bar Chart

sns.countplot(x='StoreType', data=store_df, order=['a', 'b', 'c', 'd'])

plt.show()
# Figure 9. Bar Charts for Average Sales and Customers per Score Type.

fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))

sns.barplot(x='StoreType', y='Sales', data=store_df, order=['a', 'b', 'c', 'd'], ax=axis1)

sns.barplot(x='StoreType', y='Customers', data=store_df, order=['a', 'b', 'c', 'd'], ax=axis2)
plt.show()
# Figure 10. Bar Chart for Assortment.

# Assortment

# Plot Assortment, & Assortment Vs average sales and customers

sns.countplot(x='Assortment', data=store_df, order=['a', 'b', 'c'])
plt.show()
# Figure 11. Bar Charts for Average Sales and Customers per Assortment.

fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))

sns.barplot(x='Assortment', y='Sales', data=store_df, order=['a', 'b', 'c'], ax=axis1)

sns.barplot(x='Assortment', y='Customers', data=store_df, order=['a', 'b', 'c'], ax=axis2)
plt.show()
# Figure 12. Bar Chart for Promo2.

# Promo2

# Plot Promo2, & Promo2 Vs average sales and customers

sns.countplot(x='Promo2', data=store_df)
plt.show()
# Figure 13. Bar Charts for Average Sales and Customers per Promo2.

fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))

sns.barplot(x='Promo2', y='Sales', data=store_df, ax=axis1)

sns.barplot(x='Promo2', y='Customers', data=store_df, ax=axis2)
plt.show()
# Figure 14. Scatterplot between Sales and CompetitionDistance.

# CompetitionDistance

# fill NaN values

store_df["CompetitionDistance"].fillna(store_df["CompetitionDistance"].median())

# Plot CompetitionDistance Vs Sales

store_df.plot(kind='scatter', x='CompetitionDistance', y='Sales', figsize=(15, 4))

store_df.plot(kind='kde', x='CompetitionDistance', y='Sales', figsize=(15, 4))
plt.show()
# Figure 15. Correlation Coefficients between the First Five Stores

# Correlation

# Visualize the Correlation between stores

store_piv = pd.pivot_table(rossmann_df, values='Sales', index='Date', columns=['Store'], aggfunc='sum')

store_pct_chage = store_piv.pct_change().dropna()

store_piv.head()

# Plot correlation between range of stores

start_store = 1

end_store = 5

fig, (axis1) = plt.subplots(1, 1, figsize=(15, 5))

# using summation of sales values for each store

sns.heatmap(store_piv[list(range(start_store, end_store + 1))].corr(), annot=True, linewidths=2)

# using percent change for each store

plt.show()

##############################################################################


##############################################################################

######################      Data Management         ##########################

##############################################################################

rossmann_df.drop(['DayOfWeek'], axis=1, inplace=True)

test_df.drop(['DayOfWeek'], axis=1, inplace=True)

# drop unnecessary columns, these columns won't be useful in prediction

rossmann_df.drop(["Open", "Date"], axis=1, inplace=True)

# remove all rows(store,date) that were closed

test_df = test_df[test_df["Open"] != 0]

# drop unnecessary columns, these columns won't be useful in prediction

test_df.drop(['Open', 'Date'], axis=1, inplace=True)

##############################################################################


##############################################################################

#################                Modelling               #####################

##############################################################################

lm = LinearRegression()

lasso = Lasso(alpha=0.05, max_iter=100000, random_state=1)

ENet = ElasticNet(alpha=0.05, l1_ratio=0.8,

                  random_state=3)

ridge = Ridge(alpha=0.0005)

GBR = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,

                                max_depth=2, random_state=0, loss='ls')

RFG = RandomForestRegressor(max_depth=9, random_state=5, n_estimators=100)

###############################################################################

# clean data

rossmann_df.drop(rossmann_df[~rossmann_df['StateHoliday'].isin([0, 1])].index, inplace=True)

# Loop through each store,

# train the model using the data of current store, and predict it's sales values.

rossmann_dic = dict(list(rossmann_df.groupby('Store')))


def rmse_model(model, mdata):
    mse = []

    for i in rossmann_dic:
        # current store

        store = mdata[i]

        # define training and testing sets

        X = store.drop(["Sales", "Store"], axis=1)

        y = store["Sales"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_pred = pd.Series(y_pred)

        y_test = y_test.reset_index(drop=True)

        msei = ((y_pred - y_test) ** 2).mean()

        mse.append(msei)

    mse = pd.Series(mse)

    RMSE = np.sqrt(mse.mean())

    print(model)

    print('RMSE: %.2f ' % RMSE)

    return RMSE


rmse_model(lm, rossmann_dic)

rmse_model(lasso, rossmann_dic)

rmse_model(ENet, rossmann_dic)

rmse_model(ridge, rossmann_dic)

rmse_model(GBR, rossmann_dic)

rmse_model(RFG, rossmann_dic)
