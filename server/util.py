import pickle
import json
import numpy as np

# Variables to store location names, data columns, and the trained model
__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    """
    Get the estimated home price based on input parameters.

    Parameters:
    - location: The locality of the home.
    - sqft: Total square footage of the home.
    - bhk: Number of bedrooms.
    - bath: Number of bathrooms.

    Returns:
    - Estimated home price.
    """
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

def get_locations():
    """
    Get the list of available locations.

    Returns:
    - List of location names.
    """
    return __locations

def get_data_columns():
    return __data_columns

def load_saved_artifacts():
    """
    Load saved artifacts, including location names, data columns, and the trained model.
    """
    print("Loading saved artifacts...start")
    global __locations
    global __data_columns

    # Load data columns from JSON file
    with open('../server/artifacts/columns.json', 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model

    # Load the trained model from a pickled file
    with open('../server/artifacts/hpp-lm.pickle', 'rb') as f:
        __model = pickle.load(f)

    print("Loading saved artifacts...done")

# Test the functions when running this script
if __name__ == '__main__':
    load_saved_artifacts()
    print(get_locations())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2))
    print(get_estimated_price('Ejipura', 1000, 2, 2))
