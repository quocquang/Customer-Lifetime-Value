# 🛒 Customer LifeTime Values Analysis

---

# :briefcase: Business Case and Requirement


* Calculate Customer Lifetime Value (CLTV) and Build a predictive model for CLTV based on historical purchase data.

 - The model utilizes:
 
    - Statistical Model:
        - Calculating CLTV: Using the CLTV formula to compute the CLTV value for each customer. This formula relies on the following factors:
        - Average Order Value (AOV): Total revenue divided by the number of orders.
        - Purchase Frequency: The number of orders divided by the number of days.
        - Churn Rate: The rate at which customers stop purchasing within a certain period of time.
        - Profit Margin: The percentage of profit earned from each order.
        - Statistical Analysis: Utilize statistical metrics such as mean, median, MAE (Mean Absolute Error), and RMSE (Root Mean Squared Error) to evaluate the effectiveness of the CLTV prediction model..
   - Linear Regression:
     - The code uses a linear regression model to predict CLTV for each customer based on purchase history data.
     - Predictive variable: Purchase Month-Year.
     - Dependent variable: CLTV.
     - Using the method of splitting the dataset into training and testing sets to evaluate the model's effectiveness.

---

# :bookmark_tabs:Example Datasets

## ✔ Ecommerce_Data-1 dataset

- InvoiceNo: Unique identifier for each invoice.
- StockCode: Code for the stock item.
- Description: Description of the stock item.
- Quantity: Number of items purchased.
- UnitPrice: Price per unit of the item.
- CustomerID: Unique identifier for each customer.
- Country: Country where the purchase was made.
- Date: Date of the purchase.
- Hour: Hour of the purchase.
- Total_revenue: Total revenue generated from the purchase.

<details><summary>  Click to expand Campaign-Data Dataset </summary>

<div align="center">

**Table: Ecommerce_Data-1 dataset** 

<div align="center">
First 10 rows

| InvoiceNo | StockCode | Description                      | Quantity | UnitPrice | CustomerID | Country        | Date       | Hour | Total_revenue |
|-----------|-----------|----------------------------------|----------|-----------|------------|----------------|------------|------|---------------|
| 574477    | 22591     | CARDHOLDER GINGHAM CHRISTMAS TREE| 1        | 3.25      | 15453.0    | United Kingdom | 2011-11-04 | 12   | 3.25          |
| 570275    | 23541     | WALL ART CLASSIC PUDDINGS        | 12       | 7.45      | 13098.0    | United Kingdom | 2011-10-10 | 10   | 89.40         |
| 566482    | 22508     | DOORSTOP RETROSPOT HEART         | 12       | 3.75      | 16609.0    | United Kingdom | 2011-09-13 | 9    | 45.00         |
| 541215    | 22662     | LUNCH BAG DOLLY GIRL DESIGN      | 10       | 1.65      | 14329.0    | United Kingdom | 2011-01-14 | 13   | 16.50         |
| 565930    | POST      | POSTAGE                          | 5        | 18.00     | 12685.0    | France         | 2011-09-08 | 10   | 90.00         |

</details>

---



#  IMPORT LIBRARY & CLEAN & TRANSFORM DATA.

  
```python
import pandas as pd
import numpy as np
import os 
os.chdir("C:/Users/phu/OneDrive/Pictures")
```

```python
#Upload dataset
customers = pd.read_csv('/content/drive/MyDrive/Final/De 1/dataset/customers_dataset.csv')
```

```python
# Dropping the 'Unnamed: 0' column
df=df.drop(columns='Unnamed: 0')
```

```python
# Calculate total revenue by multiplying quantity and unit price
df['Total_revenue'] = df['Quantity'] * df['UnitPrice']
```

```python
# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])
```

```python
# Remove rows where Quantity is less than or equal to 0
df = df[df['Quantity'] >0]
```

```python
# Define a function to compute customer moder
def customer_moder(data):
    max_date =data['Date'].max()
    data = data.groupby('CustomerID').agg({
    'Date':lambda x:(max_date-x.min()).days,
    'InvoiceNo':lambda x:len(x),
    'Quantity':lambda x:x.sum(),
    'Total_revenue':lambda x:x.sum()
})
    return data
# Apply the function to compute customer moder
data = customer_moder(df)
```

```python
# Rename columns
data.columns = ['num_days','num_transaction','quanity','total_revenue']
# Remove rows with zero or negative quantity
data=data[data['quanity']>0]
```

```python
data
```
| CustomerID | num_days | num_transaction | quanity | total_revenue |
|------------|----------|-----------------|---------|---------------|
| 12347.0    | 367      | 48              | 623     | 1146.59       |
| 12348.0    | 358      | 7               | 531     | 550.92        |
| 12349.0    | 18       | 16              | 203     | 406.32        |
| 12350.0    | 310      | 2               | 36      | 45.60         |
| 12352.0    | 296      | 21              | 163     | 663.33        |
| ...        | ...      | ...             | ...     | ...           |
| 18280.0    | 277      | 3               | 9       | 52.75         |
| 18281.0    | 180      | 2               | 11      | 33.45         |
| 18282.0    | 126      | 4               | 11      | 67.85         |
| 18283.0    | 337      | 133             | 240     | 325.54        |
| 18287.0    | 201      | 17              | 284     | 364.96        |


---



# Calculate CLTV
Calculate Key Metrics:
- num_days: Time customers stay (in days)
- num_transaction: Number of transactions
- quantity: Total purchase quantity
- total_revenue: Total revenue generated
```python
df['Total_revenue'] = df['Quantity'] * df['UnitPrice']
```
- AOV (Average order value): total_revenue / num_transaction
```python
data['AOV'] = data['total_revenue']/data['num_transaction']
```
- purchase_freq: Average transaction frequency per year
```python
purchase_freq = sum(data['num_transaction'])/len(data)
```
- repeat_rate: Percentage of customers making more than one transaction
```python
repeat_rate = data[data['num_transaction'] >1].shape[0]/data.shape[0]
```  
- churn_rate: 1 - repeat_rate
```python
churn_rate = 1-repeat_rate
```  
- profit_margin: 10% của total_revenue
```python
data['profit_margin'] = data['total_revenue'] * .10
```  
- CLTV: ((AOV * purchase_freq) / churn_rate) * 0.10
```python
data['CLTV'] = ((data['AOV']* purchase_freq)/churn_rate)*.10
```  
---

- Num_days (Number of days of purchase):
   - There is a significant variation in the number of days customers make purchases, ranging from 18 days to 367 days.
   - Some customers make purchases more frequently over the observation period.
- Num_transaction (Number of transactions):
   - Ranges from 2 transactions to 133 transactions, indicating a large diversity in the number of purchases among customers.
   - Some customers make more transactions than others, possibly indicating their loyalty or engagement with the product or service.
- Quantity (Quantity of products purchased):
   - The quantity of products purchased ranges from 2 to 623 items.
   - There is significant variation among customers in terms of the quantity of products purchased, reflecting different preferences or shopping needs.
- Total_revenue (Total revenue):
   - Average sales volume ranges from $45.60 to $1,146.59.
   - There is a large discrepancy among customers in terms of spending, possibly reflecting their purchasing power and buying preferences.
- AOV (Average Order Value):
   - Average order value ranges from $2.45 to $78.70.
   - AOV is an important metric to assess sales performance and the profitability of the business. Fluctuations in AOV may reflect changes in pricing strategies or differences in purchasing behavior among customers.


| CustomerID | num_days | num_transaction | quanity | total_revenue | AOV | profit_margin | CLTV |
|------------|----------|-----------------|---------|---------------|-----|---------------|-------|
| 12977.0    | 156      | 1               | 2       | 0.78          | 0.78| 0.078         | 21.47 |
| 14661.0    | 26       | 1               | 1       | 0.85          | 0.85| 0.085         | 23.39 |
| 16658.0    | 368      | 1               | 1       | 0.85          | 0.85| 0.085         | 23.39 |
| 17962.0    | 142      | 3               | 3       | 2.99          | 0.99| 0.299         | 27.43 |
| 17859.0    | 304      | 5               | 5       | 5.21          | 1.04| 0.521         | 28.68 |
| ...        | ...      | ...             | ...     | ...           | ... | ...           | ...   |
| 12931.0    | 239      | 15              | 5590    | 9344.82       | 622.99| 934.48       | 17146.78 |
| 16333.0    | 298      | 11              | 7840    | 7944.40       | 722.22| 794.44       | 19877.93 |
| 17949.0    | 289      | 10              | 4565    | 7671.45       | 767.14| 767.14       | 21114.48 |
| 16000.0    | 2        | 5               | 2630    | 7948.30       | 1589.66| 794.83      | 43752.92 |
| 15749.0    | 332      | 4               | 8898    | 20713.30      | 5178.33| 2071.33     | 142525.36 |

---

- Purchase Frequency:
  - The average purchase frequency of customers is around 19.5 times during the observed period.
  - This indicates that, on average, each customer makes approximately 19.5 transactions during the observed period.
![image](https://github.com/quocquang/Marketing-Strategy-Analysis/assets/87820013/194b2334-efcd-4121-b70e-014f69104fac)
- Repeat Rate:
  - The repeat purchase rate is about 93%, significantly higher than the remaining rate.
  - This suggests that a large proportion of customers return to make purchases after their first purchase, which is a positive sign of customer loyalty.
![image](https://github.com/quocquang/Marketing-Strategy-Analysis/assets/87820013/ca5587e3-60ed-4d6d-a2aa-c185d9a8e4d5)
- Churn Rate:
  - The churn rate (customers leaving) is about 7%, much lower than the rate of customers returning to make purchases.
  - This indicates that a small proportion of customers do not return to make purchases after their first purchase.
![image](https://github.com/quocquang/Marketing-Strategy-Analysis/assets/87820013/61f6d387-15ee-4d2c-b30b-4b9aca6199d9)
- Profit Margin:
  - Profit margin has been calculated based on total sales revenue with a profit margin of 10%.
  - This is the expected profit that the business can generate from sales revenue.
![image](https://github.com/quocquang/Marketing-Strategy-Analysis/assets/87820013/440c4b4a-d67f-4c71-a9b7-3f7146bb19d0)
- Customer Lifetime Value (CLTV):
  - The average and median CLTV are calculated to be approximately $747.74 and $460.54, respectively.
  - This is an estimate of the value that each customer can bring to the business over the entire cooperation period.
![image](https://github.com/quocquang/Marketing-Strategy-Analysis/assets/87820013/0e94aac8-e898-47d1-b9ba-e24d6c90e3c7)

---
### ✔ Use linear regression model to predict CLTV
 
 ```python
# Create a new column 'month-year' containing the month and year information from the 'Date' column
df['month-year'] = df['Date'].apply(lambda x:x.strftime("%b-%y"))
```

 ```python
# Selecting relevant columns for the model
model = df[['CustomerID','InvoiceNo','Total_revenue','Quantity','UnitPrice','month-year']]
```

 ```python
# Creating a pivot table to reshape the data
sales = model.pivot_table(index=['CustomerID'],columns=['month-year'],values='Total_revenue',aggfunc='sum',fill_value=0).reset_index()
```

 ```python
# Calculating CLV by summing up the total revenue across all months for each customer
sales['CLV'] = sales.iloc[:,2:].sum(axis =1)
```

 ```python
# print sales
sales
```
| CustomerID | Apr-11 | Aug-11 | Dec-10 | Dec-11 | Feb-11 | Jan-11 | Jul-11 | Jun-11 | Mar-11 | May-11 | Nov-11 | Oct-11 | Sep-11 | CLV    |
|------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| 12347.0    | 113.60 | 205.56 | 140.00 | 57.00  | 0.00   | 155.95 | 0.00   | 114.00 | 0.00   | 0.00   | 0.00   | 360.48 | 0.00   | 1032.99|
| 12348.0    | 137.00 | 0.00   | 332.16 | 0.00   | 0.00   | 41.76  | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 40.00  | 413.92 |
| 12349.0    | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 406.32 | 0.00   | 0.00   | 406.32 |
| 12350.0    | 0.00   | 0.00   | 0.00   | 0.00   | 45.6   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 45.60  |
| 12352.0    | 0.00   | 0.00   | 0.00   | 0.00   | 42.9   | 0.00   | 0.00   | 0.00   | 330.63 | 0.00   | 84.10  | 0.00   | 205.70 | 663.33 |
| ...        | ...    | ...    | ...    | ...    | ...    | ...    | ...    | ...    | ...    | ...    | ...    | ...    | ...    | ...    |
| 18280.0    | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 52.75  | 0.00   | 0.00   | 0.00   | 0.00   | 52.75  |
| 18281.0    | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 33.45  | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 33.45  |
| 18282.0    | 0.00   | 12.75  | 0.00   | 55.10  | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 67.85  |
| 18283.0    | 33.86  | 0.00   | 0.00   | 28.62  | 42.5   | 36.15  | 10.96  | 40.31  | 0.00   | 22.20  | 85.48  | 9.61   | 15.85  | 291.68 |
| 18287.0    | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 0.00   | 268.08 | 0.00   | 96.88  | 0.00   | 364.96 |

 ```python
# Selecting features and target variable
x = sales[['Dec-11','Nov-11','Oct-11','Sep-11','Aug-11','Jul-11']]
y = sales[['CLV']]
```

 ```python
# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
# Creating and training the linear regression model
linear = LinearRegression()
linear.fit(x_train, y_train)
# Making predictions
y_pred = linear.predict(x_test)
```

 ```python
print(linear.intercept_)
print(linear.coef_)
```
![image](https://github.com/quocquang/Marketing-Strategy-Analysis/assets/87820013/7053d64d-0736-4200-a2c3-347607ee7077)

 ```python
# Extracting coefficients and displaying them
coffiicients = linear.coef_.reshape(6,1)
a = pd.DataFrame(coffiicients,index=x_train.columns)
```

 ```python
print("Intercept:", linear.intercept_)
print("Coefficients:", linear.coef_)

# Evaluate the model
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```
![image](https://github.com/quocquang/Marketing-Strategy-Analysis/assets/87820013/6fdaa450-a887-4a21-92d2-c511a7eb6d03)


---

- Based on the results of the linear regression model using monthly sales revenue as input variables (in this case, from July 2011 to December 2010), we can analyze as follows:
  - Intercept Coefficient: The intercept of the model is approximately 21.48. This can be understood as the expected CLTV value when all input variables are zero, meaning there is no sales revenue in any month.
  - Coefficients (Regression Coefficients): The coefficient of each input variable reflects the degree of influence of sales revenue in each month on CLTV. For example, the coefficient for December 2011 is around 1.13, meaning each unit increase in sales revenue in that month leads to an increase of about 1.13 units in CLTV. Similarly, other coefficients vary in their impact depending on the month.
- Model Evaluation:
  - Mean Absolute Error (MAE): The average magnitude of the absolute error between the predicted CLTV and the actual CLTV on the test set is around 131.41. This indicates that the model can predict CLTV with an average error of about 131.41 units.
  - Mean Squared Error (MSE): The average magnitude of the squared error between the predicted CLTV and the actual CLTV on the test set is approximately 84139.95. A lower MSE is desirable, as it represents the average difference between predictions and actual values.
  - Root Mean Squared Error (RMSE): RMSE is the square root of MSE and is a popular model evaluation metric. In this case, the RMSE is around 290.07 units, indicating the average variation between predictions and actual values is about 290.07 units.

 ---

# 🧾 What can you practice with this case study?
- Python
  - pandas, numpy, LinearRegression.
  - cleaning, check Null values, transforming.
  - import, save csv file. 

