"""
Contains all the constants.
"""

##==== app.py ====##

# Path to the feature vectors and pickled model
FEATURE_VECTOR = '../real-estate-price-prediction/data/final/columns.json'
PICKLED_MODEL = '../real-estate-price-prediction/data/final/hpp-lm.pickle'

# Fixed sqft parameters
MIN_SQFT = 100
MAX_SQFT = 10_000
DEFAULT_SQFT_VALUE = 1_000

# Fixed bhk and bathroom parameters
MAX_BHK_COUNT = 6
MAX_BATH_COUNT = 6
