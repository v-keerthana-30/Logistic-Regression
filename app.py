import streamlit as st
import pandas as pd
import pickle

#Load Model and Scaler
import numpy as np

class DummyModel:
    def predict(self, X):
        # Always return 0 for example
        return np.array([0])
    
    def predict_proba(self, X):
        # Return dummy probability [0.7 for class 0, 0.3 for class 1]
        return np.array([[0.7, 0.3]])
# Dummy scaler class
class DummyScaler:
    def transform(self, X):
        # Return input as-is
        return X
# Load model
try:
    model = pickle.load(open("logistic_model.pkl", "rb"))
except FileNotFoundError:
    st.warning("logistic_model.pkl not found. Using dummy model.")
    model = DummyModel()
# Load scaler
try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    st.warning("scaler.pkl not found. Using dummy scaler.")
    scaler = DummyScaler()

#Streamlit App
st.title("Titanic Survival Prediction (Logistic Regression)")
st.write("Enter passenger details to predict survival probability:")

# Sidebar inputs
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=25)
fare = st.sidebar.number_input("Fare", min_value=0.0, value=30.0)
embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"])

#Preprocess Input
data = {
    "Age": [age],
    "Fare": [fare],
    "Pclass_2": [1 if pclass == 2 else 0],
    "Pclass_3": [1 if pclass == 3 else 0],
    "Sex_male": [1 if sex == "male" else 0],
    "Embarked_Q": [1 if embarked == "Q" else 0],
    "Embarked_S": [1 if embarked == "S" else 0]
}
df = pd.DataFrame(data)

# Scale Age and Fare using the same scaler as training
df[['Age', 'Fare']] = scaler.transform(df[['Age', 'Fare']])

prediction = model.predict(df)[0]
probability = model.predict_proba(df)[0][1]

st.subheader("Prediction Result")
if prediction == 1:
    st.success("The passenger is likely to Survive")
else:
    st.error("The passenger is likely to Not Survive")
st.write(f"Survival Probability: {probability:.2f}")
