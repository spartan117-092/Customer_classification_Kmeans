Mall Customer Segmentation Project
==================================

Description:
------------
This project uses KMeans clustering to group mall customers based on their demographic information.

Input Format:
-------------
- The input data should be a CSV file named 'Mall_Customers.csv'.
- Required columns:
    - Gender: 'Male' or 'Female' (will be converted to 0/1)
    - Age: Integer
    - Annual Income (k$): Integer
- The columns 'CustomerID' and 'Spending Score (1-100)' are removed before clustering.

How to Use:
-----------
1. Load the CSV file.
2. Gender is mapped to numbers: Male = 0, Female = 1.
3. Data is scaled using StandardScaler.
4. KMeans clustering is performed with 2 clusters.
5. To predict the cluster for a customer, use their scaled data and call the model's predict method.

Output:
-------
- The model assigns each customer to a cluster:
    - Cluster 0: High Spending customer
    - Cluster 1: Low Spending customer
- The cluster label is added to the DataFrame as a new column 'ans'.

Error Handling:
---------------
- The code checks for missing columns, missing values, and correct data types.
- If input data is invalid, an error message is printed.

Visualization:
--------------
- The code includes options to plot cluster results and visualize customer groups using PCA.

Notes:
------
- The 'Spending Score (1-100)' column is not used for clustering, so the model predicts spending behavior based on age, gender, and income only.
- This project is for learning purposes.

Example:
--------
Input:
    Gender: Male
    Age: 30
    Annual Income (k$): 65

Output:
    Cluster: 0 (High Spending customer)

Author:
-------
Your Name
