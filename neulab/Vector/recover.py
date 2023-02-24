import numpy as np

def replace_missing_with_mean(*vectors):
    """
    Replace missing value with mean. Returns the new vector and the mean value.

    Parameters:
        vectors (array-like): One or more input vectors.

    Returns:
        tuple: A tuple containing the new vectors with missing values replaced by the mean and the mean value.
    """

    new_vectors = []
    mean_values = []

    for vector in np.array(vectors):
        missing_values = np.isnan(vector)
        num_missing_values = np.count_nonzero(missing_values)
        if num_missing_values == len(vector):
            raise ValueError("All values in vector are missing.")
        if num_missing_values == 0:
            new_vectors.append(vector)
            mean_values.append(vector.mean())
            continue
        
        mean_value = np.mean(vector[~missing_values])
        new_vector = vector.copy()
        new_vector[missing_values] = mean_value
        new_vectors.append(new_vector)
        mean_values.append(mean_value)

    if len(new_vectors) == 1:
        return new_vectors[0], mean_values[0]
    else:
        return new_vectors, mean_values


def replace_missing_with_median(*vectors):
    """
    Replace missing value with median. Returns the new vector and the median value.

    Parameters:
        vector (array-like): The input vector.

    Returns:
        tuple: A tuple containing the new vector with missing values replaced by the median and the median value.
    """

    new_vectors = []
    median_values = []

    for vector in np.array(vectors):

        missing_values = np.isnan(vector)
        num_missing_values = np.count_nonzero(missing_values)
        if num_missing_values == len(vector):
            raise ValueError("All values in vector are missing.")
        if num_missing_values == 0:
            new_vectors.append(vector)
            median_values.append(np.median(vector))
            continue

        median_value = np.median(vector[~missing_values])
        new_vector = vector.copy()
        new_vector[missing_values] = median_value
        new_vectors.append(new_vector)
        median_values.append(median_value)

    if len(new_vectors) == 1:
        return new_vectors[0], median_values[0]
    else:
        return new_vectors, median_values


def replace_missing_with_mode(*vectors):
    """
    Replace missing value with mode. Returns the new vector and the mode value.

    Parameters:
        vectors (array-like): The input vectors.

    Returns:
        tuple: A tuple containing the new vectors with missing values replaced by the mode and the mode values.
    """

    import numpy as np

    new_vectors = []
    mode_values = []

    for vector in np.array(vectors):

        missing_values = np.isnan(vector)
        num_missing_values = np.count_nonzero(missing_values)
        if num_missing_values == len(vector):
            raise ValueError("All values in vector are missing.")
        if num_missing_values == 0:
            unique, counts = np.unique(vector, return_counts=True)
            mode_index = np.argmax(counts)
            mode_value = unique[mode_index]
        else:
            unique, counts = np.unique(vector[~missing_values], return_counts=True)
            if len(unique) == 1:
                mode_value = unique[0]
            else:
                mode_index = np.argmax(counts)
                mode_value = unique[mode_index]
            vector[missing_values] = mode_value
        new_vectors.append(vector)
        mode_values.append(mode_value)

    if len(new_vectors) == 1:
        return new_vectors[0], mode_values[0]
    else:
        return new_vectors, mode_values


def replace_missing_with_corrcoef(*vectors):
    """
    Replace missing values using correlation coefficient. Returns the new vector and the correlation.

    Parameters:
        vector (array-like): The input vector.

    Returns:
        tuple: A tuple containing the new vector with missing values replaced by values calculated with correlation coefficient and the correlation coefficient value.
    """

    from neulab.Vector.discover import correlation_coefficient

    recovered_vector = []
    corr_coefs = []

    missing_values = np.isnan(vectors)
    num_missing_values = np.count_nonzero(missing_values)
    if num_missing_values == 0:
        # If there are no non-missing values, return the input vector and NaN correlation
        return vectors, np.nan

    for vector in np.array(vectors):

        missing_values = np.isnan(vector)
        num_missing_values = np.count_nonzero(missing_values)
        if num_missing_values == len(vector):
            raise ValueError("All values in vector are missing.")
            
        # Assuming that missing values are represented by NaNs
        missing_mask = np.isnan(vector)
        missing_indices = np.where(missing_mask)[0]
        nonmissing_indices = np.where(~missing_mask)[0]
        
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

        recovered_vector.append(new_vector)
        corr_coefs.append(round(corr_coef, 6))
    
    return recovered_vector, corr_coefs


def replace_missing_with_distance(*vectors, metric='euclidean'):
    """
    Replace missing values using a given distance metric. Returns the new vector and the distance.
    Parameters:
        *vectors (array-like): The input vectors.
        metric (str): The distance metric to use. Can be 'euclidean', 'manhattan', or 'max'.
    Returns:
        tuple: A tuple containing the new vectors with missing values replaced by values calculated with the given distance metric and the distance.
    """

    from neulab.Vector.discover import euclidean_distance
    from neulab.Vector.discover import manhattan_distance
    from neulab.Vector.discover import max_distance

    if np.array(vectors).shape[1] == 1:
        return vectors, np.nan
    elif np.array(vectors).shape[1] > 1:
        vectors = np.array(vectors).squeeze()

    for vt in np.array(vectors):
        # Find the missing values in the input vectors
        missing_values = np.isnan(vt)
        num_missing_values = np.count_nonzero(missing_values)
        if num_missing_values == vt.size:
            raise ValueError("All values in vectors are missing.")
    
    missing_values = np.isnan(vectors)
    num_missing_values = np.count_nonzero(missing_values)
    if num_missing_values == 0:
        # If there are no non-missing values, return the input vector and NaN correlation
        return vectors, np.nan
    
    vectors = np.array(vectors)

    indices = np.where(missing_values)

    distances = []
    dst = []
    for i in range(len(vectors)):
        indices = np.where(np.isnan(vectors[i]))
        num_missing_values = np.count_nonzero(np.isnan(vectors[i]))
        if num_missing_values == 0:
            continue
        distances = []
        for j in range(len(vectors)):
            non_missing_values_i = np.delete(vectors[i], np.where(np.isnan(vectors[i])))
            non_missing_values_j = np.delete(vectors[j], np.where(np.isnan(vectors[j])))
            if i != j:
                v1 = np.where(np.isnan(vectors[i]), np.mean(non_missing_values_i), vectors[i])
                v2 = np.where(np.isnan(vectors[j]), np.mean(non_missing_values_j), vectors[j])
                if metric == 'euclidean':
                    distances.append(euclidean_distance(v1, v2)[0])
                if metric == 'manhattan':
                    distances.append(manhattan_distance(v1, v2)[0])
                if metric == 'max':
                    distances.append(max_distance(v1, v2)[0])

        val_list = np.delete(vectors[i], np.where(np.isnan(vectors[i])))
        DISTANCE = round(1 / (sum([1/i for i in distances])) * sum([v/d for v, d in zip(val_list, distances)]), 2)
        dst.append(DISTANCE)

        for z in indices[0]:
            vectors[i][z] = DISTANCE

    # Transpose the vector array back to its original shape if necessary
    replaced = np.array(vectors)
    
    return replaced, dst
