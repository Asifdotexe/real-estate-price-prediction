import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        location_index = __data_columns.index(location.lower())
    except:
        location_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if location_index >= 0:
        x[location_index] = 1

    return round(__model.predict([x])[0],2)

def get_locations():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __locations, __data_columns

    with open('../server/artifacts/columns.json','r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    
    global __model
    with open('../server/artifacts/hpp-lm.pickle','rb') as f:
        __model = pickle.load(f)
    print("Loading saved artifacts...done")


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_locations())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2))
    print(get_estimated_price('Ejipura', 1000, 2, 2))