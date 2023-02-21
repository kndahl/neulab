import numpy as np

def min_max_normalizer(*vectors):
    """
    Normalizes input arrays using the Min-Max algorithm to a range of [0, 1].
    """
    normalized_vectors = []
    for vector in vectors:
        # Check for NaNs in the input arrays.
        if np.isnan(vector).any():
            raise ValueError("Input arrays must not contain NaN values")
            
        # Normalize the array.
        min_val = np.nanmin(vector)
        max_val = np.nanmax(vector)
        normalized_vector = (vector - min_val) / (max_val - min_val)
        normalized_vectors.append(normalized_vector)
    
    return normalized_vectors

def mean_normalizer(*vectors):
    """
    Normalizes an input array using mean algorithm.
    """
    normalized_vectors = []
    for vector in vectors:
        # Check for NaNs in the input arrays.
        if np.isnan(vector).any():
            raise ValueError("Input arrays must not contain NaN values")
        
        mean = np.mean(vector)
        normalized_arr = (vector - mean) / np.std(vector)
        normalized_vectors.append(normalized_arr)
    
    return normalized_vectors
