import numpy as np

def min_max_normalize(*arrays):
    """
    Normalizes n input arrays using the Min-Max algorithm to a range of [0, 1].
    """
    normalized_arrays = []
    for array in arrays:
        # Check for NaNs in the input arrays.
        if np.isnan(array).any():
            raise ValueError("Input arrays must not contain NaN values")
            
        # Normalize the array.
        min_val = np.nanmin(array)
        max_val = np.nanmax(array)
        normalized_array = (array - min_val) / (max_val - min_val)
        normalized_arrays.append(normalized_array)
    
    return normalized_arrays

# def meanNormalization(column):
#     ''' Normalization function that normalize data in column by mean algotithm '''

#     vector = np.array(column)
#     mean_normalization = (vector - np.mean(vector))/(np.std(vector))
#     return mean_normalization
