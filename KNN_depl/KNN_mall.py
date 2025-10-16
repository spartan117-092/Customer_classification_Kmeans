import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
import joblib

df = pd.read_csv('Mall_Customers.csv')
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df = df.drop('CustomerID',axis=1)
df = df.drop('Spending Score (1-100)',axis=1)

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df))
print(df_scaled,type(df_scaled))
# l = []
# for i in range(1,10):
#     ml = KMeans(n_clusters=i)
#     ml.fit(df_scaled)
#     l.append(ml.inertia_)

# plt.plot(range(1,10),l)
# plt.show()

ml = KMeans(n_clusters=2)
ans = ml.fit_predict(df_scaled)

pipeline = {
    'scaler': scaler,
    'model': ml
}

joblib.dump(pipeline, 'customer_segmentation.pkl')
print("Model and scaler saved successfully to 'customer_segmentation.pkl'")

# final_ans = ml.predict(df_scaled.iloc[4].values.reshape(1,-1))
# if(final_ans==1):
#     print('Low spending customer')
# elif(final_ans==0):
#     print('High Spending customer')

# df['ans'] = ans
# print(df.iloc[4])



# def validate_and_prepare_input(new_customer_df, expected_columns, scaler):
#     try:
#         # Check all required columns are present
#         if not all([col in new_customer_df.columns for col in expected_columns]):
#             missing_cols = [col for col in expected_columns if col not in new_customer_df.columns]
#             raise ValueError(f"Missing columns in input: {missing_cols}")
        
#         # Reorder columns if necessary
#         if list(new_customer_df.columns) != expected_columns:
#             new_customer_df = new_customer_df[expected_columns]
        
#         # Check for missing values
#         if new_customer_df.isnull().any().any():
#             raise ValueError("Input contains missing values")
        
#         # Convert Gender if necessary
#         if 'Gender' in new_customer_df.columns and not pd.api.types.is_numeric_dtype(new_customer_df['Gender']):
#             new_customer_df['Gender'] = new_customer_df['Gender'].map({'Male': 0, 'Female':1})
#             if new_customer_df['Gender'].isnull().any():
#                 raise ValueError("Gender column contains invalid values")
        
#         # Scale input
#         scaled_input = scaler.transform(new_customer_df)
        
#         return scaled_input
    
#     except Exception as e:
#         print(f"Input validation error: {e}")
#         return None


# # Usage example with prediction
# scaled_customer = validate_and_prepare_input(new_customer, expected_columns, scaler)
# if scaled_customer is not None:
#     try:
#         prediction = ml.predict(scaled_customer)
#         # proceed with prediction results
#     except Exception as e:
#         print(f"Prediction error: {e}")
# else:
#     print("Invalid input data. Prediction aborted.")









# Assuming df_scaled is your scaled data and ans is the cluster labels from KMeans
# score = silhouette_score(df_scaled, ans)
# print(f'Silhouette Score: {score:.3f}')


# ///////////////////////////////////////////////////////////VISUALSATION PART/////////////////////////////////////////////////////////
# # ------------------------------therefore 1- low spending and 0 - high spending-----------------------
# df_ck1  = df[df['ans']==1]
# df_ck0  = df[df['ans']==0]
# print('0 max=',df_ck0.max())
# print('0 min=',df_ck0.min())
# print('1 max=',df_ck1.max())
# print('1 min=',df_ck1.min())
# ----------------------------------------------------------------------------------------------------

# pca = PCA(n_components=2)
# df_pca = pca.fit_transform(df_scaled)
# df_pca = pd.DataFrame(df_pca,columns=['P1','P2'])
# df_pca['ans'] = ans

# df1 = df_pca[df_pca['ans']==1]
# df0 = df_pca[df_pca['ans']==0]
# # df2 = df_pca[df_pca['ans']==2]
# # df3 = df_pca[df_pca['ans']==3]

# # plt.scatter(df3['P1'],df3['P2'],color='purple')
# # plt.scatter(df2['P1'],df2['P2'],color='yellow')
# plt.scatter(df1['P1'],df1['P2'],color='blue')
# plt.scatter(df0['P1'],df0['P2'],color='red')
# plt.show()
# ---------------------plot bgy perplexity----------------------
# plt.figure(figsize=(8, 6))
# plt.scatter(df_pca['P1'], df_pca['P2'], c=df_pca['ans'], cmap='viridis', s=50)
# plt.title('Customer Segmentation using KMeans and PCA')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()
