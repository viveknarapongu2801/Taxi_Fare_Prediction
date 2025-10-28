import streamlit as st
import pandas as pd
import numpy as np


# Load the trained model
# Assuming the model is saved as 'xgb_model.pkl'
# You would need to save the model after training
# joblib.dump(xgb, 'xgb_model.pkl')
try:
    xgb_model = joblib.load('xgb_model.pkl')
except FileNotFoundError:
    st.error("Model file 'xgb_model.pkl' not found. Please train and save the model first.")
    st.stop()


st.title("Taxi Trip Price Prediction")

st.header("Enter Trip Details")

# Input fields for features
trip_distance_km = st.number_input("Trip Distance (km)", min_value=0.1, value=10.0)
passenger_count = st.number_input("Passenger Count", min_value=1, value=1)
base_fare = st.number_input("Base Fare", min_value=0.0, value=3.0)
per_km_rate = st.number_input("Per Km Rate", min_value=0.1, value=0.5)
per_minute_rate = st.number_input("Per Minute Rate", min_value=0.1, value=0.3)
trip_duration_minutes = st.number_input("Trip Duration (minutes)", min_value=1.0, value=30.0)

time_of_day = st.selectbox("Time of Day", ['Morning', 'Afternoon', 'Evening', 'Night'])
day_of_week = st.selectbox("Day of Week", ['Weekday', 'Weekend'])
traffic_conditions = st.selectbox("Traffic Conditions", ['Low', 'Medium', 'High'])
weather = st.selectbox("Weather", ['Clear', 'Rain', 'Snow'])

# Create a dictionary with input values
input_data = {
    'Trip_Distance_km': trip_distance_km,
    'Passenger_Count': passenger_count,
    'Base_Fare': base_fare,
    'Per_Km_Rate': per_km_rate,
    'Per_Minute_Rate': per_minute_rate,
    'Trip_Duration_Minutes': trip_duration_minutes,
    'Time_of_Day_Evening': 1 if time_of_day == 'Evening' else 0,
    'Time_of_Day_Morning': 1 if time_of_day == 'Morning' else 0,
    'Time_of_Day_Night': 1 if time_of_day == 'Night' else 0,
    'Day_of_Week_Weekend': 1 if day_of_week == 'Weekend' else 0,
    'Traffic_Conditions_Low': 1 if traffic_conditions == 'Low' else 0,
    'Traffic_Conditions_Medium': 1 if traffic_conditions == 'Medium' else 0,
    'Traffic_Conditions_High': 1 if traffic_conditions == 'High' else 0, # Added High traffic condition
    'Weather_Rain': 1 if weather == 'Rain' else 0,
    'Weather_Snow': 1 if weather == 'Snow' else 0,
    'Weather_Clear': 1 if weather == 'Clear' else 0 # Added Clear weather condition
}

# Ensure all columns are present in the input data for prediction
# This is important if the model was trained with one-hot encoded columns
# that might not appear in a single user input.
# Get the list of columns from the training data
# Assuming X_train is available or you have a list of expected columns
expected_columns = ['Trip_Distance_km', 'Passenger_Count', 'Base_Fare', 'Per_Km_Rate',
                    'Per_Minute_Rate', 'Trip_Duration_Minutes', 'Time_of_Day_Evening',
                    'Time_of_Day_Morning', 'Time_of_Day_Night', 'Day_of_Week_Weekend',
                    'Traffic_Conditions_Low', 'Traffic_Conditions_Medium', 'Traffic_Conditions_High',
                    'Weather_Rain', 'Weather_Snow', 'Weather_Clear'] # Updated with all possible dummy columns

# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])

# Reindex the DataFrame to ensure all expected columns are present
# Fill missing columns with 0, as they represent absent categories
input_df = input_df.reindex(columns=expected_columns, fill_value=0)


if st.button("Predict Price"):
    # Make prediction
    predicted_price = xgb_model.predict(input_df)

    st.success(f"Predicted Trip Price: ${predicted_price[0]:.2f}")
