
def zscore_outliers(*vectors, method='mean', threshold=3):
    """
    Z-score algorithm to remove outliers.

    This function uses the Z-score algorithm to remove outliers from a variable-length list of vectors.

    Parameters:
        *vectors (list of lists): A variable-length list of vectors, each represented as a list of numbers.
        method (str): Specifies the method used to calculate the Z-score. Acceptable values are 'mean' or 'median'.
        threshold (int or float): The threshold for identifying outliers. Values greater than threshold times the standard deviation from the mean (or median) will be considered outliers.

    Returns:
        A tuple containing two lists:
            - A list of cleared vectors, where outliers have been removed.
            - A list of the dropped outliers.
    """

    from neulab.Vector.discover import std_deviation
    import numpy as np
    import warnings

    if method not in ['mean', 'median']:
        raise ValueError('Invalid method specified.')

    cleared_vectors = []
    outliers = []
    for vector in vectors:
        # Convert the list to a numpy array
        vector = np.array(vector)

        if vector.shape[0] < 11:
            warnings.warn(f'The z-score algorithm may not perform well on small vectors. Recommended minimum vector length is 11, received: {vector.shape[0]}')

        # Calculate the median and median absolute deviation of the vector
        if method == 'median':
            # Calculate z-score
            z_score = np.abs((vector - np.median(vector)) / std_deviation(vector))
            # Identify outliers using a threshold of standard deviations from the median
            is_outlier = z_score > threshold
        elif method == 'mean':
            # Calculate z-score
            z_score = np.abs((vector - np.mean(vector)) / std_deviation(vector))
            # Identify outliers using a threshold times of standard deviations from the mean
            is_outlier = z_score > threshold

        if np.any(is_outlier):
            # Add the outliers to the list
            outliers.extend(list(vector[is_outlier]))

            # Remove the outliers from the vector
            not_outlier_indices = np.where(~is_outlier)[0]
            vector = vector[not_outlier_indices]
        cleared_vectors.append(list(vector))

    return cleared_vectors, outliers


def chauvenet_outliers(*vectors):
    """
    Chauvenet algorithm to remove outliers.

    Parameters:
        *vectors: variable-length list of vectors, each represented as a list of numbers.

    Returns:
        A tuple containing two lists:
            - A list of cleared vectors, where outliers have been removed.
            - A list of the dropped outliers.
    """

    from neulab.Vector.discover import std_deviation
    from scipy.special import erfc
    import numpy as np

    cleared_vectors = []
    outliers = []
    for vector in vectors:
        # Convert the list to a numpy array
        vector = np.array(vector)

        mean = np.mean(vector)
        std = std_deviation(vector)

        # Lenght of incoming array
        N = len(vector) 
        # Chauvenet's criterion
        criterion = 1.0 / (2 * N) 
        # Distance of a value to mean in std's
        d = abs(vector - mean) / std
        # Area normal dist.
        prob = erfc(d)

        bools = prob < criterion

        outliers.append(vector[np.where(bools)])

        # Remove the outliers from the data
        cleared_vector = [vector[i] for i in range(len(vector)) if vector[i] not in outliers[0]]
        cleared_vectors.append(cleared_vector)

    # Concatenate outliers
    outliers = np.concatenate(outliers)
    outliers = outliers.flatten()

    return cleared_vectors, outliers


def quartile_outliers(*vectors):
    """
    Use the Quartile algorithm to remove outliers.

    Parameters:
        *vectors: variable-length list of vectors, each represented as a list of numbers.

    Returns:
        A tuple containing two lists:
            - A list of cleared vectors, where outliers have been removed.
            - A list of the dropped outliers.
    """

    import numpy as np

    cleared_vectors = []
    outliers = []
    for vector in vectors:
        # Convert the list to a numpy array
        vector = np.array(vector).flatten()

        # Calculate the first and third quartiles of the data
        q1 = np.percentile(vector, 25)
        q3 = np.percentile(vector, 75)

        # Calculate the interquartile range
        iqr = q3 - q1

        # Calculate the minimum and maximum values that are not outliers
        min_val = q1 - 1.5 * iqr
        max_val = q3 + 1.5 * iqr

        for i in range(len(vector)):
            val = vector[i]
            if val < min_val or val > max_val:
                outliers.append(val)

        # Remove the outliers from the data
        cleared_vector = [val for val in vector if val not in outliers]
        cleared_vectors.append(cleared_vector)

    return cleared_vectors, outliers


def dixon_test_outliers(*vectors):
    """
    Use the Dixon Q-test algorithm to remove outliers.

    Parameters:
        *vectors: variable-length list of vectors, each represented as a list of numbers.

    Returns:
        A tuple containing two lists:
            - A list of cleared vectors, where outliers have been removed.
            - A list of the dropped outliers.
    """

    q90 = [0.941, 0.765, 0.642, 0.56, 0.507, 0.468, 0.437,
        0.412, 0.392, 0.376, 0.361, 0.349, 0.338, 0.329,
        0.32, 0.313, 0.306, 0.3, 0.295, 0.29, 0.285, 0.281,
        0.277, 0.273, 0.269, 0.266, 0.263, 0.26
        ]

    q95 = [0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466,
        0.444, 0.426, 0.41, 0.396, 0.384, 0.374, 0.365, 0.356,
        0.349, 0.342, 0.337, 0.331, 0.326, 0.321, 0.317, 0.312,
        0.308, 0.305, 0.301, 0.29
        ]

    q99 = [0.994, 0.926, 0.821, 0.74, 0.68, 0.634, 0.598, 0.568,
        0.542, 0.522, 0.503, 0.488, 0.475, 0.463, 0.452, 0.442,
        0.433, 0.425, 0.418, 0.411, 0.404, 0.399, 0.393, 0.388,
        0.384, 0.38, 0.376, 0.372
        ]

    Q90 = {n:q for n,q in zip(range(3,len(q90)+1), q90)}
    Q95 = {n:q for n,q in zip(range(3,len(q95)+1), q95)}
    Q99 = {n:q for n,q in zip(range(3,len(q99)+1), q99)}

    outliers = []
    cleared_vectors = []
    
    for vector in vectors:
        n = len(vector)
        vector_sorted = sorted(vector)
        
        Q_val = Q90.get(n) if n in Q90 else (Q95.get(n) if n in Q95 else Q99.get(n))
        R = (vector_sorted[-1] - vector_sorted[0]) / (vector_sorted[-1] - vector_sorted[1])
        
        if R > Q_val:
            outlier = vector_sorted[-1]
            outliers.append(outlier)

        # Remove the outliers from the data
        cleared_vector = [vector[i] for i in range(len(vector)) if vector[i] not in outliers]
        cleared_vectors.append(cleared_vector)
    
    return cleared_vectors, outliers