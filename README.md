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

### ✔ Ecommerce_Data-1 dataset

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
<details><summary> 👆🏼 Click to expand Orders Dataset </summary>

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

---



# A. Data Exploration and Cleansing

## 🔤 IMPORT LIBRARY AND DATASET 

<details><summary> Click to expand :arrow_down: </summary>
  
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

## 🔎 EXPLORE, CLEAN & TRANSFORM DATA

### 1️⃣ Ecommerce_Data-1  Dataset

```python

df=df.drop(columns='Unnamed: 0')

```

```python

df['Total_revenue'] = df['Quantity'] * df['UnitPrice']

```

```python

df['Total_revenue'] = df['Quantity'] * df['UnitPrice']

```

```python

df['Date'] = pd.to_datetime(df['Date'])

```

```python

df = df[df['Quantity'] >0]

```

### ✔ Tính toán CLTV
Tính toán các chỉ số chính:
- num_days: Thời gian khách hàng gắn bó (tính bằng ngày)
- num_transaction: Số lượng giao dịch
- quantity: Tổng số lượng mua
- total_revenue: Tổng doanh thu thu được
- AOV (Giá trị đơn hàng trung bình): total_revenue / num_transaction
- purchase_freq: Tần suất giao dịch trung bình mỗi năm
- repeat_rate: Tỷ lệ khách hàng thực hiện hơn một giao dịch
- churn_rate: 1 - repeat_rate
- profit_margin: 10% của total_revenue
- CLTV: ((AOV * purchase_freq) / churn_rate) * 0.10

---

- Num_days (Số ngày mua hàng):
   - Có sự biến động đáng kể trong số ngày mua hàng của các khách hàng, từ 18 ngày đến 367 ngày.
   - Có một phần khách hàng mua hàng thường xuyên hơn trong thời gian quan sát.
- Num_transaction (Số lần giao dịch):
   - Từ 2 lần giao dịch tới 133 lần giao dịch, có sự đa dạng lớn về số lần mua hàng của các khách hàng.
   - Có thể thấy một số khách hàng thực hiện nhiều giao dịch hơn so với số khác, điều này có thể chỉ ra sự trung thành hoặc tích cực của họ đối với sản phẩm hoặc dịch vụ.
- Quanity (Số lượng sản phẩm mua):
   - Số lượng sản phẩm mua hàng dao động từ 2 đến 623 sản phẩm.
   - Có sự biến động lớn giữa các khách hàng trong việc mua sắm số lượng sản phẩm, có thể phản ánh sở thích hoặc nhu cầu mua hàng khác nhau của họ.
- Total_revenue (Tổng doanh số bán hàng):
   - Doanh số bán hàng trung bình từ 45,60 đô la đến 1.146,59 đô la.
   - Có sự chênh lệch lớn giữa các khách hàng về mức độ chi tiêu, có thể phản ánh sức mua và ưu tiên mua hàng của họ.
- AOV (Average Order Value - Giá trị đơn hàng trung bình):
   - Giá trị đơn hàng trung bình dao động từ 2,45 đô la đến 78,70 đô la.
   - AOV là một chỉ số quan trọng để đánh giá hiệu suất bán hàng và khả năng sinh lời của doanh nghiệp, sự biến động trong AOV có thể phản ánh sự biến động trong chiến lược giá cả hoặc sự khác biệt trong hành vi mua hàng của khách hàng.


---

- Purchase Frequency (Tần suất mua hàng):
  - Tần suất mua hàng trung bình của khách hàng là khoảng 19,5 lần trong khoảng thời gian quan sát.
  - Điều này chỉ ra rằng trung bình mỗi khách hàng thực hiện khoảng 19,5 giao dịch trong khoảng thời gian được quan sát.
  - Repeat Rate (Tỷ lệ khách hàng trở lại mua hàng):
- Tỷ lệ khách hàng mua hàng lần thứ hai trở lên là khoảng 93%, cao hơn đáng kể so với tỷ lệ còn lại.
  - Điều này cho thấy một tỷ lệ lớn các khách hàng trở lại mua hàng sau lần mua đầu tiên, đây là một dấu hiệu tích cực về sự trung thành của khách hàng.
- Churn Rate (Tỷ lệ churn - Khách hàng bỏ đi):
  - Tỷ lệ churn (khách hàng bỏ đi) là khoảng 7%, tỷ lệ này thấp hơn rất nhiều so với tỷ lệ khách hàng trở lại mua hàng.
  - Điều này chỉ ra rằng một tỷ lệ nhỏ các khách hàng không trở lại mua hàng sau lần mua đầu tiên.
- Profit Margin (Mar lợi nhuận):
  - Mar lợi nhuận đã được tính dựa trên tổng doanh số bán hàng với mức lợi nhuận là 10%.
  - Đây là lợi nhuận kỳ vọng mà doanh nghiệp có thể thu được từ doanh số bán hàng.
- Customer Lifetime Value (CLTV - Giá trị khách hàng trọn đời):
  - Giá trị trung bình và trung vị của CLTV được tính toán là khoảng 747,74 đô la và 460,54 đô la, tương ứng.
  - Đây là ước tính về giá trị mà mỗi khách hàng có thể mang lại cho doanh nghiệp trong suốt thời gian hợp tác.
---
### ✔ Sử dụng mô hình hồi quy tuyến tính để dự đoán CLTV 
- Dựa trên kết quả của mô hình hồi quy tuyến tính sử dụng các biến đầu vào là doanh số bán hàng theo từng tháng (trong trường hợp này là các tháng từ Tháng 7 năm 2011 đến Tháng 12 năm 2010), chúng ta có thể phân tích như sau:
  - Hệ số Intercept: Intercept của mô hình là khoảng 21.48. Điều này có thể được hiểu là giá trị CLTV dự kiến khi tất cả các biến đầu vào đều bằng 0, tức là không có doanh số bán hàng trong bất kỳ tháng nào.
  - Hệ số Coefficients (Hệ số hồi quy): Hệ số của mỗi biến đầu vào thể hiện mức độ ảnh hưởng của doanh số bán hàng trong từng tháng đối với CLTV. Ví dụ, hệ số cho tháng Dec-11 là khoảng 1.13, có nghĩa là mỗi đơn vị tăng trong doanh số bán hàng của tháng này sẽ dẫn đến một tăng khoảng 1.13 đơn vị trong CLTV. Các hệ số khác cũng tương tự, với mức độ ảnh hưởng khác nhau tùy thuộc vào từng tháng.
- Đánh giá mô hình:
  - Mean Absolute Error (MAE): Độ lớn trung bình của sai số tuyệt đối giữa giá trị CLTV dự đoán và giá trị thực tế trên tập kiểm tra là khoảng 131.41. Điều này cho biết mô hình có thể dự đoán giá trị CLTV với sai số trung bình khoảng 131.41 đơn vị.
  - Mean Squared Error (MSE): Độ lớn trung bình của bình phương sai số giữa giá trị CLTV dự đoán và giá trị thực tế trên tập kiểm tra là khoảng 84139.95. MSE càng thấp càng tốt, vì nó biểu thị sự khác biệt trung bình giữa các dự đoán và giá trị thực tế.
  - Root Mean Squared Error (RMSE): RMSE là căn bậc hai của MSE và là một chỉ số đánh giá mô hình phổ biến. Trong trường hợp này, RMSE khoảng 290.07 đơn vị, cho thấy sự biến thiên trung bình giữa các dự đoán và giá trị thực tế là khoảng 290.07 đơn vị.

# 🧾 What can you practice with this case study?
- Python
  - pandas, numpy,matplotlib,seaborn.
  - cleaning, check Null values, transforming.
  - import, save csv file. 

