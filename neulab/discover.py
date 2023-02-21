import numpy as np
import warnings

def correlation_coefficient(*vectors):
    """Returns correlation matrix between vectors."""
    n_vectors = len(vectors)

    # Check that the input vectors have the same shape.
    shape = np.array([len(v) for v in vectors])
    if np.any(shape != shape[0]):
        raise ValueError("Input vectors must have the same shape")

    # Center each vector around its mean.
    centered = [v - np.mean(v) for v in vectors]

    # Calculate the covariance matrix and standard deviation vector.
    cov = np.cov(centered)
    stds = np.sqrt(np.diag(cov))

    # Calculate the correlation matrix.
    corr_coef = cov / np.outer(stds, stds)

    return corr_coef


def euclidean_distance(*vectors):
    """Returns Euclidean distance between all pairs of vectors."""
    n = len(vectors)
    distances = 0
    
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
    
            distance = np.sqrt(np.sum((point1 - point2) ** 2))
            distances += distance

    return distances


def manhattan_distance(*vectors):
    """Returns Manhattan distances between all pairs of vectors."""
    n = len(vectors)
    distances = 0
    
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
            distance = np.sum(np.abs(point1 - point2))
            distances += distance
        
    return distances


def max_distance(*vectors):
    """Returns Max distance between all pairs of vectors."""
    n = len(vectors)
    distances = 0

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
            distance = np.max(np.abs(point1 - point2))
            distances += distance

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
