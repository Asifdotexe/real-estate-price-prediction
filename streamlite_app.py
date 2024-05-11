import streamlit as st
import pickle
import json
import numpy as np

# Load saved artifacts
def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __locations
    global __data_columns

    # Load data columns from JSON file
    with open('columns.json', 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model

    # Load the trained model from a pickled file
    with open('hpp-lm.pickle', 'rb') as f:
        __model = pickle.load(f)

    print("Loading saved artifacts...done")

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

    load_saved_artifacts()

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
        st.success(f'Estimated Price: ₹{estimated_price}')

if __name__ == '__main__':
    main()