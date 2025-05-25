import pickle
import json
from typing import Any

import streamlit as st
import numpy as np


# Cache the model and metadata loading
@st.cache_resource
def load_model_and_metadata() -> tuple[Any, list[str], list[str]]:
    """Loads the trained machine learning model and associated metadata
    needed for making predictions.

    Why is this function needed?
        To avoid repeatedly loading large files everytime the streamlit app reruns,
        which would slow down the user experience.
        We use Streamlit's `st.cache_resource` to cache this loading step
        and improve performance. Separating this logic also keeps our main code
        cleaner and more modular.
    """
    # reading the file which contains the features the model was trained on
    with open(
            '../real-estate-price-prediction/server/artifacts/columns.json') as f:
        data_columns = json.load(f)['data_columns']

    # reading the file containing the pre-trained model
    with open('../real-estate-price-prediction/server/artifacts/hpp-lm.pickle',
              'rb') as f:
        model = pickle.load(f)

    # the first 3 columns are numerical features (sqft, bathroom, bhk) and the rest are locations
    locations = data_columns[3:]

    return model, data_columns, locations


# Function to predict home price
def predict_home_price(
        model: Any,
        data_columns: list[str],
        total_sqft: float,
        location: str,
        bhk: int,
        bath: int
) -> float:
    """Predicts the price of a house on user input and trained model.

    Args:
        model: The pre-trained model for price prediction
        data_columns: List of feature names the model expects,
        including encodings.
        total_sqft: Total area of the property in square feet.
        location: Location of the property selected by the user.
        bhk: Number of bedrooms (BHK) in the property.
        bath: Number of bathrooms in the property.

    Returns:
        Predicted price of the property, rounded to two decimal prices (in Lakhs).

    """

    # the model expects an input array where the first few indices represent
    # numeric features (total_sqft, bath, bhk),
    # and the rest are one-hot encoded location names
    input_vector = np.zeros(len(data_columns))

    # setting values for total_sqft, bath and bhk.
    # These are fixed columns always expected at the beginning of the feature list.
    input_vector[0] = total_sqft
    input_vector[1] = bath
    input_vector[2] = bhk

    # Location handling
    # Not all user-selected location may be present in the model's training data
    # if it's present, we one-hot encode it by setting the corresponding index to 1
    location = location.lower().strip()
    if location in data_columns:
        location_index = data_columns.index(location)
        input_vector[location_index] = 1
    # NOTE:
    # If the location is unknown,
    # we intentionally skip setting any one-hot encoded location.
    # the model will still make a prediction based on other inputs
    # (not ideal but functional).

    prediction = model.predict([input_vector])[0]
    return round(prediction,2)


### STREAMLIT APP STARTS HERE

def main():
    # header section
    st.set_page_config(page_title='House Price Prediction',
                       page_icon='🏠',
                       layout='centered')
    st.title('🏠 Bangalore Home Price Predictor')
    st.caption("🔍 Predict the estimated price of a home in Bangalore based on its features.")

    # Load model and metadata only once
    model, data_columns, locations = load_model_and_metadata()

    # Use a form to group input and avoid auto-refresh on every interaction
    with st.form('prediction_form'):
        st.markdown('### 📋 Property Details')

        # Two-column layout for compact input view
        location_bhk_column, sqft_bath_column = st.columns(2)

        with location_bhk_column:
            location = st.selectbox(
                '📍 Select Location',
                locations,
                help='Choose the location where the property is located.'
            )

            bhk = st.selectbox(
                '🛏️ Number of Bedrooms (BHK)',
                list(range(1,6)),
                index=1,
                help='Select how many bedrooms the property has.'
            )

        with sqft_bath_column:
            total_sqft = st.number_input(
                '📐 Total Area (in Square Feet)',
                min_value = 100,
                max_value = 10_000,
                value= 1_000,
                step= 50,
                help='Enter the total built-up area of the home.'
            )

            bath = st.selectbox(
                '🛁 Number of Bathrooms',
                list(range(1,6)),
                index=1,
                help='Choose how many bathrooms are available in the property.'
            )

        st.markdown("💡 *Ensure values are realistic for accurate predictions.*")

        submitted = st.form_submit_button("Estimate Price 💰")

    if submitted:
        # Validation: Warn if sqft too small for selected BHK
        if bhk > total_sqft // 300:
            st.warning("⚠️ The square footage seems too small for the number of bedrooms.")

        estimated_price = predict_home_price(model, data_columns, total_sqft,
                                             location, bhk, bath)

        st.success("✅ Prediction Successful!")
        st.metric(label="🏷️ Estimated Home Price",
                  value=f"₹ {estimated_price} Lakh")

if __name__ == '__main__':
    main()