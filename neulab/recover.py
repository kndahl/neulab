import numpy as np
from neulab.discover import correlation_coefficient
from neulab.discover import euclidean_distance
from neulab.discover import manhattan_distance
from neulab.discover import max_distance

def replace_missing_with_mean(*vectors):
    """
    Replace missing value with mean. Returns the new vector and the mean value.

    Parameters:
        vector (array-like): The input vector.

    Returns:
        tuple: A tuple containing the new vector with missing values replaced by the mean and the mean value.
    """

    # Combine the input vectors into a single 2D array
    if len(vectors) > 1:
        vector = np.vstack(vectors)
    else:
        vector = np.array(vectors)[0]

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


def replace_missing_with_median(*vectors):
    """
    Replace missing value with median. Returns the new vector and the median value.

    Parameters:
        vector (array-like): The input vector.

    Returns:
        tuple: A tuple containing the new vector with missing values replaced by the median and the median value.
    """

    # Combine the input vectors into a single 2D array
    if len(vectors) > 1:
        vector = np.vstack(vectors)
    else:
        vector = np.array(vectors)[0]

    missing_values = np.isnan(vector)
    num_missing_values = np.count_nonzero(missing_values)
    if num_missing_values == len(vector):
        raise ValueError("All values in vector are missing.")
    if num_missing_values == 0:
        return vector, np.median(vector)
    median_value = np.median(vector[~missing_values])
    vector[missing_values] = median_value
    return vector, median_value


def replace_missing_with_mode(*vectors):
    """
    Replace missing value with mode. Returns the new vector and the mode value.

    Parameters:
        vector (array-like): The input vector.

    Returns:
        tuple: A tuple containing the new vector with missing values replaced by the mode and the mode value.
    """

    # Combine the input vectors into a single 2D array
    if len(vectors) > 1:
        vector = np.vstack(vectors)
    else:
        vector = np.array(vectors)[0]

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


def replace_missing_with_corrcoef(*vectors):
    """
    Replace missing values using correlation coefficient. Returns the new vector and the correlation.

    Parameters:
        vector (array-like): The input vector.

    Returns:
        tuple: A tuple containing the new vector with missing values replaced by values calculated with correlation coefficient and the correlation coefficient value.
    """

    # Combine the input vectors into a single 2D array
    if len(vectors) > 1:
        vector = np.vstack(vectors)
    else:
        vector = np.array(vectors)[0]

    missing_values = np.isnan(vector)
    num_missing_values = np.count_nonzero(missing_values)
    if num_missing_values == len(vector):
        raise ValueError("All values in vector are missing.")
        
    # Assuming that missing values are represented by NaNs
    missing_mask = np.isnan(vector)
    missing_indices = np.where(missing_mask)[0]
    nonmissing_indices = np.where(~missing_mask)[0]
    
    if len(missing_indices) == 0:
        # If there are no non-missing values, return the input vector and NaN correlation
        return vector, np.nan
    
    # Calculate the correlation coefficient between non-missing values and missing indices
    corr_coef = correlation_coefficient(nonmissing_indices, vector[nonmissing_indices])[0]
    
    if np.isnan(corr_coef):
        # If the correlation coefficient is NaN, return the input vector and NaN correlation
        return vector, np.nan
    
    # Calculate the new values for missing indices using the correlation coefficient
    new_values = (missing_indices - nonmissing_indices.mean()) * corr_coef + vector[nonmissing_indices].mean()
    
    # Replace the missing values with the new values
    new_vector = vector.copy()
    new_vector[missing_mask] = new_values
    
    return new_vector, round(corr_coef, 6)


def replace_missing_with_distance(*vectors, metric='euclidean', how='vertical'):
    """
    Replace missing values using a given distance metric. Returns the new vector and the distance.

    Parameters:
        *vectors (array-like): The input vectors.
        metric (str): The distance metric to use. Can be 'euclidean', 'manhattan', or 'max'.
        how (str): The direction of deleting a missing value.
            'vertical' - deletes missing values from different columns in each row.
            'horizontal' - deletes missing values from different rows in each column.

    Returns:
        tuple: A tuple containing the new vectors with missing values replaced by values calculated with the given distance metric and the distance.
    """
    # Combine the input vectors into a single 2D array
    if len(vectors) > 1:
        vector_array = np.vstack(vectors)
    else:
        vector_array = np.array(vectors)[0]

    vec_arr_copy = vector_array.copy()
    
    # Find the missing values in the input vectors
    missing_values = np.isnan(vector_array)
    num_missing_values = np.count_nonzero(missing_values)
    if num_missing_values == vector_array.size:
        raise ValueError("All values in vectors are missing.")
    
    if num_missing_values == 0:
        # If there are no non-missing values, return the input vector and NaN correlation
        return vector_array, np.nan
    
    if how == 'vertical':
        vector_array = vector_array.transpose()
        # Find the rows containing missing values
        missing_rows = np.any(missing_values, axis=1)
        # Delete the rows containing missing values
        vector_array = np.delete(vector_array, np.where(missing_rows), axis=1)
    elif how == 'horizontal':
        # Find the columns containing missing values
        missing_cols = np.any(missing_values, axis=0)
        # Delete the columns containing missing values
        vector_array = np.delete(vector_array, np.where(missing_cols), axis=1)

    print(vector_array)

    indices = np.where(missing_values)
        
    distances = []
    for i in range(vector_array.shape[0]-1):
        if  i != indices:
            if metric == 'euclidean':
                distances.append(euclidean_distance(vector_array[i], vector_array[-1])[0])
            if metric == 'manhattan':
                distances.append(manhattan_distance(vector_array[i], vector_array[-1])[0])
            if metric == 'max':
                distances.append(max_distance(vector_array[i], vector_array[-1])[0])

    # Get row with nan
    val_list = vec_arr_copy[indices[0][0]]
    val_list = val_list[~np.isnan(val_list)]
    
    DISTANCE = round(1/(sum([1/i for i in distances])) * sum([v/d for v, d in zip(val_list, distances)]), 2)

    # Replace missing values with DISTANCE
    vec_arr_copy[indices] = DISTANCE

    # Transpose the vector array back to its original shape if necessary
    if how == 'vertical':
        vector_array = vec_arr_copy.transpose()
    
    return vec_arr_copy, DISTANCE
