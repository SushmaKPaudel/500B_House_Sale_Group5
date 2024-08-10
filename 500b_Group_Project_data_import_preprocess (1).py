#!/usr/bin/env python
# coding: utf-8

# In[212]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib as mp1 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[213]:


# this is just example


# 1. Data Importing and Pre-processing
# 
# 1.1 Import dataset and describe characteristics such as dimensions, data types, file types, and import methods used

# In[214]:


#Importing the data as dataframe.We are going to use pd.read_csv to import the file in dataframe.


# In[215]:


df =pd.read_csv('/Users/niyatkahsay/Desktop/ADS500B/week_six/Dataset 2 (House Sales)/house_sales.csv')


# In[216]:


## Looking few rows os data at glance 
df.head()


# In[217]:


print(df.shape) ## Shape attribute gives us dimension of dataframe so lets's use .shape to identify number of rows and columns of dataframe.


# In[218]:


print(df.dtypes) ## dttypes attribute tells us data types of each column from the the dataframe.


# In[219]:


## Let's use Python's built in libraray to identyfy the extenstion of file using os.path.splitext().

file_extension = os.path.splitext("house_sales.csv")
print(file_extension)

Lets see the dataype of each columns in dataframe
# In[220]:


numeric_columns = df.select_dtypes(include=['number']).columns
nonnumeric_columns =df.select_dtypes(exclude=['number']).columns

print (numeric_columns)
print (nonnumeric_columns)


# 1.2.Clean, wrangle, and handle missing data.

# lets's count null values for each columns

# In[221]:


null_counts = df.isnull().sum()

print(null_counts)

4 fields bedrooms, bathrooms, sqft_living and sqft_lot contain null values. Now , lets replace the null values with the mean value of each fields as we already know the 4 fields that are missing values are numeric types.
# In[222]:


df.fillna(df.mean(numeric_only=True).round(1), inplace=True)

Now, lets verify there are no more null values in those 4 fields.
# In[223]:


null_counts2 = df.isnull().sum()
print(null_counts2)

1.3. Transform data appropriately using techniques such as aggregation, normalization, and feature construction
# In[224]:


#Normalize columns all numeric columns
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Verify normalization
df.head()


# In[225]:


1.4. Reduce redundant data and perform need-based discretization


# In[ ]:


# Check duplicates based on the 'id' column.
duplicates = df.duplicated(subset=['id'], keep=False)

# identyfying dataframe duplicate_rows from the df to get the duplicate rows
duplicate_rows = df[duplicates]

print(duplicate_rows)

Remove the duplicate records for each primarykey/id from dataframe df1
# In[ ]:


df = df.drop_duplicates(subset=['id'], keep='last') ## we decided to keep last from each dupe id

# Now , lets verify there are no more duplicates in dataframe.
# In[ ]:


# Check duplicates based on the 'id' column.
duplicates = df.duplicated(subset=['id'], keep=False)

# identyfying dataframe duplicate_rows from the df to get the duplicate rows
duplicate_rows = df[duplicates]

print(duplicate_rows)


# In[ ]:


#2.1. Identify Categorical, Ordinal, and Numerical Variables


# In[ ]:


categorical_vars = ['waterfront', 'view', 'zipcode']
ordinal_vars = ['grade']
numerical_vars = list(set(df.columns) - set(categorical_vars) - set(ordinal_vars) - {'id', 'date'})

print("Categorical variables:", categorical_vars)
print("Ordinal variables:", ordinal_vars)
print("Numerical variables:", numerical_vars)


# In[ ]:


# 2.2. Measures of Centrality and Distribution with Visualizations


# In[ ]:


#lets find numerical summary first (statistical distribution)
numerical_summary = df[numerical_vars].describe()
numerical_summary


# In[ ]:


#visualizations 
#first histogram for numerical values 


# In[ ]:


df[numerical_vars].hist(bins=20, figsize=(20, 15))
plt.show()


# In[ ]:


#2nd boxplots


# In[ ]:


plt.figure(figsize=(20, 15))
for i, var in enumerate(numerical_vars):
    plt.subplot(4, 4, i+1)
    sns.boxplot(x=df[var])
plt.tight_layout()
plt.show()


# In[ ]:


#2.3. Correlation Analysis


# In[ ]:


correlation_matrix = df[numerical_vars].corr()
correlation_matrix
#we can explain this correlation table


# In[ ]:


#visualize the correlation with Heatmap, to check for multicollinearity


# In[228]:


plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[ ]:


# having our dependent variable 'Price', 
# we choose independent variables that have correlation with price, corr > o.5 and corr <-0.5


# In[246]:


# Extracting correlations with the target variable 'price'
correlation_with_price = correlation_matrix['price'].sort_values(ascending=False)

# Display correlations with price
print(correlation_with_price)


# In[247]:


# Select variables with strong correlation to price
strong_correlations = correlation_with_price[abs(correlation_with_price) > 0.5]
print(strong_correlations)


# In[ ]:


#after getting highly correlated variables lets check for multicollinearity among the variables


# In[227]:


for col1 in numerical_vars:
    for col2 in numerical_vars:
        if col1 != col2:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=df[col1], y=df[col2])
            plt.title(f'{col1} vs {col2}')
            plt.show()


# In[ ]:


#Calculating R-squared 


# In[231]:


y = df['price']

variables = ['sqft_living', 'sqft_above', 'sqft_living15', 'bathrooms', 'sqft_basement', 
             'bedrooms', 'lat', 'floors', 'yr_renovated', 'sqft_lot', 
             'sqft_lot15', 'yr_built', 'condition', 'long']


# In[243]:


for var in variables:
    X = df[[var]]
    X = sm.add_constant(X)  # Adds a constant (intercept) term to the model
    model = sm.OLS(y, X).fit()
    r_squared = model.rsquared
    r_squared_values[var] = r_squared

# Create a DataFrame from the r_squared_values dictionary
r_squared_df = pd.DataFrame(list(r_squared_values.items()), columns=['Variable', 'R-squared'])

# Round the R-squared values to two decimal places
r_squared_df['R-squared'] = r_squared_df['R-squared'].round(2)

# Print the DataFrame
print(r_squared_df)


# In[ ]:


#now we check for multicollinearity between our selected variables


# In[248]:


X = df [['sqft_living', 'sqft_above', 'sqft_living15', 'bathrooms']].dropna()
X = sm.add_constant(X)

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)


# In[ ]:




