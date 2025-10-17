# import joblib
# import pandas as pd

# # Load saved pipeline
# pipeline = joblib.load('customer_segmentation.pkl')

# # Prepare new customer data
# while(True):
#     try:
#         gender = int(input('Gender-1 male and 0 female'))
#         age = int(input('age'))
#         anual_income = int(input('anual_income'))

#         new_data = pd.DataFrame({'Gender': [gender], 'Age': [age], 'Annual Income (k$)': [anual_income]})

#         # Predict using loaded model and scaler
#         prediction = pipeline['model'].predict(pipeline['scaler'].transform(new_data))
#         break
#     except Exception as e:
#         print(e)
#         print('Need all the variable in the int form')
# print("Predicted cluster:", prediction)
import streamlit as st
import joblib
import pandas as pd

# Load your saved pipeline (model + scaler)
pipeline = joblib.load('customer_segmentation.pkl')

# Title for the app
st.title('Mall Customer Segmentation Predictor')

# Input fields for user data
gender = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Female' if x==0 else 'Male')
age = st.number_input('Age', min_value=1, max_value=120, value=30)
annual_income = st.number_input('Annual Income (k$)', min_value=0, value=50)

if st.button('Predict'):
    # Prepare input data for prediction
    input_df = pd.DataFrame({'Gender': [gender], 'Age': [age], 'Annual Income (k$)': [annual_income]})
    
    # Predict cluster
    prediction = pipeline['model'].predict(pipeline['scaler'].transform(input_df))
    
    # Display result
    if prediction[0] == 0:
        st.success('High Spending Customer')
    else:
        st.info('Low Spending Customer')

