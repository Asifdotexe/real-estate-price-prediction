import streamlit as st
import pickle
import json
import numpy as np
import os

# Load saved artifacts
def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __locations
    global __data_columns

    try:
        # Load data columns from JSON file
        columns_file_path = '../real-estate-price-prediction/server/artifacts/columns.json'
        if os.path.exists(columns_file_path):
            with open(columns_file_path, 'r') as f:
                __data_columns = json.load(f)['data_columns']
                __locations = __data_columns[3:]
        else:
            st.error("Error: Columns JSON file not found!")
            return False

        # Load the trained model from a pickled file
        model_file_path = '../real-estate-price-prediction/server/artifacts/hpp-lm.pickle'
        if os.path.exists(model_file_path):
            with open(model_file_path, 'rb') as f:
                __model = pickle.load(f)
        else:
            st.error("Error: Model pickle file not found!")
            return False

        print("Loading saved artifacts...done")
        return True

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return False

# Variables to store location names, data columns, and the trained model
__locations = None
__data_columns = None
__model = None

# Function to get locations
def get_locations():
    return __locations

# Function to predict home price
def predict_home_price(total_sqft, location, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

# Streamlit app
def main():
    st.title('Bangalore Home Price Prediction')

    if not load_saved_artifacts():
        return

    # Select location
    st.subheader('Location')
    locations = get_locations()
    selected_location = st.selectbox('Choose a Location', locations)

    # Input area (square feet)
    st.subheader('Area (Square Feet)')
    total_sqft = st.number_input('Enter total square footage of the home', value=1000, min_value=100, step=50)

    # Input BHK
    st.subheader('BHK')
    bhk = st.selectbox('Number of Bedrooms', ['1', '2', '3', '4', '5'], index=1)

    # Input Bathrooms
    st.subheader('Bath')
    bath = st.selectbox('Number of Bathrooms', ['1', '2', '3', '4', '5'], index=1)

    # Button to predict
    if st.button('Estimate Price'):
        estimated_price = predict_home_price(total_sqft, selected_location, int(bhk), int(bath))
        st.success(f'Estimated Price: ₹{estimated_price}L')

if __name__ == '__main__':
    main()
