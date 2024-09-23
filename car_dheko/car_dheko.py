import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import time

@st.cache_data
def load_data():
    data = pd.read_csv('cleaned_combined_cars.csv')
    data['Registration_Year'].fillna(data['Registration_Year'].median(), inplace=True)  # Impute with median
    return data

@st.cache_resource
def train_model(data):
    features = ['Kilometers_Driven', 'Registration_Year', 'Mileage', 'Engine_Power', 'City']
    X = data[features]
    y = data['Price']

    # Preprocessing for numerical data
    numerical_features = ['Kilometers_Driven', 'Registration_Year', 'Mileage', 'Engine_Power']
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_features = ['City']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', CatBoostRegressor(silent=True))])

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    return model, features
# Show loading spinner
with st.spinner("🚗 Loading bus data..."):
    time.sleep(2)
def main():
    st.title("🚗 Used Car Price Prediction")
    st.image("C:\\Users\\moham\\Downloads\\dd.jpeg", caption="Car Price Prediction", use_column_width=True)

    # Load data
    data = load_data()

    # Train the model
    model, features = train_model(data)

    # Input options in the sidebar
    st.sidebar.subheader("Input Details")
    car_make = st.sidebar.selectbox("Select Car Make", options=data['Car_Model'].unique())
    fuel_type = st.sidebar.selectbox("Select Fuel Type", options=data['Fuel_Type'].unique())
    registration_year = st.sidebar.selectbox("Select Registration Year", options=data['Registration_Year'].unique())
    engine_power = st.sidebar.number_input("Engine Power (in bhp)", min_value=30, max_value=500, value=100)
    kilometers_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=500000, value=10000)
    city = st.sidebar.selectbox("Select City", options=data['City'].unique())

    # Calculate average mileage for the selected car make and city
    mileage = data.loc[
        (data['Car_Model'] == car_make) & 
        (data['City'] == city) & 
        (data['Registration_Year'] == registration_year), 'Mileage'
    ]

    if mileage.empty:
        mileage = data['Mileage'].median()
    else:
        mileage = mileage.mean()

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Kilometers_Driven': [kilometers_driven],
        'Registration_Year': [registration_year],
        'Engine_Power': [engine_power],
        'Mileage': [mileage],
        'City': [city]
    })

    # Add button to trigger prediction
    if st.sidebar.button("Predict Price"):
        predicted_price = model.predict(input_data)[0]
        st.markdown(f"<h1 style='text-align: center; color: black; font-size: 32px;'><b>Predicted Car Price: ₹{predicted_price:,.2f}</b></h1>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
