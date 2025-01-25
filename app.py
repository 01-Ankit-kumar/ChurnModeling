import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model
import pickle

# Load the trained model
model = load_model('model.h5')

# Load the encoder and scaler
with open('Label_encoder_gender.pkl', 'rb') as file:
    Label_encoder_gender = pickle.load(file)
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", Label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox("Has credit card", [0, 1])
is_active_member = st.selectbox('Is active Member', [0, 1])

# Preparing the input dataset
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Apply OneHotEncoder to Geography
onehot_encoded_geo = onehot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(
    onehot_encoded_geo,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Encode Gender using LabelEncoder
input_data['Gender'] = Label_encoder_gender.transform(input_data['Gender'])

# Combine the input data with the one-hot encoded geography
input_df = pd.concat([input_data, geo_encoded_df], axis=1)
input_df.drop("Geography", axis=1, inplace=True)

# Scale the input data (only numeric columns)
input_scaled = scaler.transform(input_df)

# Make predictions
prediction = model.predict(input_scaled)
prediction_probab = prediction[0][0]
st.write(f"the prediction probab is {prediction_probab}")
# Display the result
if prediction_probab > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")
