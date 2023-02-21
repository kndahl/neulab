import numpy as np

def correlation_coefficient(vector1, vector2):
    """Returns correlation coefficient between vector1 and vector2."""
    v1 = np.array(vector1)
    v2 = np.array(vector2)

    # Check that the input vectors have the same shape.
    if v1.shape != v2.shape:
        raise ValueError("Input vectors must have the same shape")

    mean1 = np.mean(v1)
    mean2 = np.mean(v2)
    cov = np.mean((v1 - mean1) * (v2 - mean2))
    std1 = np.std(v1, ddof=0)
    std2 = np.std(v2, ddof=0)

    corr_coef = cov / (std1 * std2)

    return corr_coef


def euclidean_distance(vector1, vector2):
    """Returns Euclidean distance between vector1 and vector2."""
    point1 = np.array(vector1)
    point2 = np.array(vector2)

    # Check that the input vectors have the same shape.
    if point1.shape != point2.shape:
        raise ValueError("Input vectors must have the same shape")
    
    euclidean_distance = np.sqrt(np.sum((point1 - point2) ** 2))

    return euclidean_distance


def manhattan_distance(vector1, vector2):
    """Returns Manhattan distance between vector1 and vector2."""
    point1 = np.array(vector1)
    point2 = np.array(vector2)

    # Check that the input vectors have the same shape.
    if point1.shape != point2.shape:
        raise ValueError("Input vectors must have the same shape")

    # Compute the Manhattan distance.
    manhattan_distance = np.sum(np.abs(point1 - point2))

    return manhattan_distance


def max_metric(vector1, vector2):
    """Returns max_metric distance between vector1 and vector2."""
    point1 = np.array(vector1)
    point2 = np.array(vector2)

    # Check that the input vectors have the same shape.
    if point1.shape != point2.shape:
        raise ValueError("Input vectors must have the same shape")

    # Compute the max_metric distance.
    max_metric = np.max(np.abs(point1 - point2))

    return max_metric


def replace_missing_with_mean(vector):
    """
    Replace missing value with mean. Returns the new vector and the mean value.

    Parameters:
        vector (array-like): The input vector.

    Returns:
        tuple: A tuple containing the new vector with missing values replaced by the mean and the mean value.
    """
    vector = np.array(vector)
    missing_values = np.isnan(vector)
    num_missing_values = np.count_nonzero(missing_values)
    if num_missing_values == len(vector):
        raise ValueError("All values in vector are missing.")
    if num_missing_values == 0:
        return vector, vector.mean()
    mean_value = vector[~missing_values].mean()
    new_vector = vector.copy()
    new_vector[missing_values] = mean_value
    return new_vector, mean_value


def replace_missing_with_median(vector):
    """
    Replace missing value with median. Returns the new vector and the median value.

    Parameters:
        vector (array-like): The input vector.

    Returns:
        tuple: A tuple containing the new vector with missing values replaced by the median and the median value.
    """
    vector = np.array(vector)
    missing_values = np.isnan(vector)
    num_missing_values = np.count_nonzero(missing_values)
    if num_missing_values == len(vector):
        raise ValueError("All values in vector are missing.")
    if num_missing_values == 0:
        return vector, np.median(vector)
    median_value = np.median(vector[~missing_values])
    vector[missing_values] = median_value
    return vector, median_value


def replace_missing_with_mode(vector):
    """
    Replace missing value with mode. Returns the new vector and the mode value.

    Parameters:
        vector (array-like): The input vector.

    Returns:
        tuple: A tuple containing the new vector with missing values replaced by the mode and the mode value.
    """
    vector = np.array(vector)
    missing_values = np.isnan(vector)
    num_missing_values = np.count_nonzero(missing_values)
    if num_missing_values == len(vector):
        raise ValueError("All values in vector are missing.")
    if num_missing_values == 0:
        unique, counts = np.unique(vector, return_counts=True)
        mode_index = np.argmax(counts)
        mode_value = unique[mode_index]
        return vector, mode_value
    unique, counts = np.unique(vector[~missing_values], return_counts=True)
    if len(unique) == 1:
        mode_value = unique[0]
    else:
        mode_index = np.argmax(counts)
        mode_value = unique[mode_index]
    vector[missing_values] = mode_value
    return vector, mode_value


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
