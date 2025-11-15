import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_location_names():
    global __locations
    return __locations

def get_estimated_price(location, sqft, bhk, bath):
    load_saved_artifacts()
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    if __data_columns is None:
        with open("./artifact/columns.json", "r") as f:
            __data_columns = json.load(f)["data_columns"]
            __locations = __data_columns[3:]  # first 3 -> sqft,bath,bhk

    if __model is None:
        with open("./artifact/model.pkl", "rb") as f:
            __model = pickle.load(f)

    print("loading saved artifacts...done")
