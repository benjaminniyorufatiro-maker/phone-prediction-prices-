# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 10:14:20 2025

@author: Admin
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st
import os

# Get current directory
current_dir = os.path.dirname(__file__)

# Load the trained model (use relative path)
model_path = os.path.join(current_dir, 'phone_sales_data.pkl')
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Function to predict phone price
def phone_price_prediction(screen_size, ram, storage, battery_capacity, camera_quality):
    new_phone = pd.DataFrame([{
        'Screen Size (inches)': screen_size,
        'RAM (GB)': ram,
        'Storage (GB)': storage,
        'Battery Capacity (mAh)': battery_capacity,
        'Camera Quality (MP)': camera_quality
    }])
    predicted_price = loaded_model.predict(new_phone)
    return predicted_price[0]

# Main Streamlit app
def main():
    st.title("📱 Phone Price Prediction App")

    # Input fields
    screen_size = st.text_input('Screen Size (inches) (e.g., 6.2)')
    ram = st.text_input('RAM (GB) (e.g., 4)')
    storage = st.text_input('Storage (GB) (e.g., 64)')
    battery_capacity = st.text_input('Battery Capacity (mAh) (e.g., 4000)')
    camera_quality = st.text_input('Camera Quality (MP) (e.g., 48)')

    if st.button('Predict Price'):
        try:
            # Convert inputs
            screen_size = float(screen_size)
            ram = int(ram)
            storage = int(storage)
            battery_capacity = int(battery_capacity)
            camera_quality = int(camera_quality)

            # Prediction
            price = phone_price_prediction(screen_size, ram, storage, battery_capacity, camera_quality)
            st.success(f'The predicted price for the phone is: ${price:.2f}')
        except ValueError:
            st.error("Please enter valid numeric values for all inputs.")

if __name__ == '__main__':
    main()
