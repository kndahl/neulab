import numpy as np
import warnings

def correlation_coefficient(*vectors):
    """Returns list of correlation coefficients between all pairs of input vectors."""
    n_vectors = len(vectors)
    corr_coef_list = []
    
    # Check that the input vectors have the same shape.
    shape = np.array([len(v) for v in vectors])
    if np.any(shape != shape[0]):
        raise ValueError("Input vectors must have the same shape")

    for i in range(n_vectors):
        for j in range(i+1, n_vectors):
            # Center each vector around its mean.
            centered_i = vectors[i] - np.mean(vectors[i])
            centered_j = vectors[j] - np.mean(vectors[j])

            # Calculate the covariance and standard deviation.
            cov = np.sum(centered_i * centered_j) / (len(centered_i) - 1)
            std_i = np.std(centered_i, ddof=1)
            std_j = np.std(centered_j, ddof=1)

            # Calculate the correlation coefficient.
            corr_coef = cov / (std_i * std_j)

            corr_coef_list.append(corr_coef)

    return corr_coef_list


def euclidean_distance(*vectors):
    """Returns Euclidean distance between all pairs of vectors."""
    n = len(vectors)
    distances = []
    
    for i in range(n):
        for j in range(i+1, n):
            point1 = np.array(vectors[i])
            point2 = np.array(vectors[j])
        
            # Check that the input vectors have the same shape.
            if point1.shape != point2.shape:
                raise ValueError("Input vectors must have the same shape")
            
            # Check for NaNs in the input arrays.
            if np.isnan(point1).any() or np.isnan(point2).any():
                warnings.warn("Input arrays contains NaN values")
    
            distances.append(np.sqrt(np.sum((point1 - point2) ** 2)))

    return distances


def manhattan_distance(*vectors):
    """Returns Manhattan distances between all pairs of vectors."""
    n = len(vectors)
    distances = []
    
    for i in range(n):
        for j in range(i+1, n):
            point1 = np.array(vectors[i])
            point2 = np.array(vectors[j])
        
            # Check that the input vectors have the same shape.
            if point1.shape != point2.shape:
                raise ValueError("Input vectors must have the same shape")
            
            # Check for NaNs in the input arrays.
            if np.isnan(point1).any() or np.isnan(point2).any():
                warnings.warn("Input arrays contains NaN values")
        
            # Compute the Manhattan distance.
            distances.append(np.sum(np.abs(point1 - point2)))
        
    return distances


def max_distance(*vectors):
    """Returns Max distance between all pairs of vectors."""
    n = len(vectors)
    distances = []

    for i in range(n):
        for j in range(i+1, n):
            point1 = np.array(vectors[i])
            point2 = np.array(vectors[j])
        
            # Check that the input vectors have the same shape.
            if point1.shape != point2.shape:
                raise ValueError("Input vectors must have the same shape")
            
            # Check for NaNs in the input arrays.
            if np.isnan(point1).any() or np.isnan(point2).any():
                warnings.warn("Input arrays contains NaN values")

            # Compute the istance distance.
            distances.append(np.max(np.abs(point1 - point2)))

    return distances


def std_deviation(vector):
    """Calculates standard deviation."""
    vector = np.array(vector)
    std = np.std(vector, ddof=1)
    return std


def is_symmetric(vector):
    """Detects if vector is symmetric or asymmetric.
    Returns True if vector is symmetric or False if vector is asymmetric."""
    vectype = np.abs(np.median(vector) - np.mean(vector)) <= 3 * np.sqrt((std_deviation(vector=vector) ** 2) / len(vector))
    return vectype
