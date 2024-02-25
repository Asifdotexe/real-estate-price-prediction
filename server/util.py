import json
import pickle

__locations = None
__data_columns = None
__model = None

def get_locations():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __locations, __data_columns

    with open('../server/artifacts/columns.json','r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open('../server/artifacts/hpp-lm.pickle','rb') as f:
        __model = pickle.load(f)
    print("Loading saved artifacts...done")


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_locations())