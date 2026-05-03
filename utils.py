import numpy as np


def calculate_local_mean(matrix, row, col, window_size=3):
    """
    Calculate the local mean around a given point (row, col) with a specified window size.

    Parameters:
    - matrix: Input 2D np.array matrix.
    - row, col: The point for which the local mean needs to be calculated.
    - window_size: Window size defining the neighborhood (default is 3x3).

    Returns:
    - local_mean: The mean of all values within the window.
    """
    # Get the range of the neighborhood
    half_window = window_size // 2
    start_row = max(0, row - half_window)
    end_row = min(matrix.shape[0], row + half_window + 1)
    start_col = max(0, col - half_window)
    end_col = min(matrix.shape[1], col + half_window + 1)

    # Extract the neighborhood matrix
    local_window = matrix[start_row:end_row, start_col:end_col]


    local_mean = np.mean(local_window)

    return local_mean
    
def replace_anomalies_with_local_mean(matrix, anomalies, window_size=3):
    """
    Replace detected anomalies with the local mean.

    Parameters:
    - matrix: Input 2D np.array matrix.
    - anomalies: A boolean matrix of the same shape as the input matrix, where True represents an anomaly and False represents a normal value.
    - window_size: Window size defining the neighborhood (default is 3x3).

    Returns:
    - modified_matrix: The matrix after replacing anomalies with local means.
    """
    
    modified_matrix = matrix.copy()


    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if anomalies[row, col]:  # If it is an anomaly
                local_mean = calculate_local_mean(matrix, row, col, window_size)
                modified_matrix[row, col] = local_mean  # Replace with local mean

    return modified_matrix



def detect_anomalies_stat(matrix, k=3):
    """
    Detect anomalies using a statistical-based method, where values exceeding mean ± k standard deviations are considered anomalies.

    Parameters:
    - matrix: The 2D np.array matrix to detect anomalies in.
    - k: Standard deviation multiplier, default is 3, meaning values greater than mean ± 3 times standard deviation are anomalies.

    Returns:
    - anomalies: Returns a boolean matrix of the same shape as the input, where True represents an anomaly and False represents a normal value.
    - anomaly_count: Returns the count of detected anomalies.
    """
    
    mean = np.mean(matrix)
    std_dev = np.std(matrix)


    lower_bound = mean - k * std_dev
    upper_bound = mean + k * std_dev


    anomalies = (matrix < lower_bound) | (matrix > upper_bound)


    anomaly_count = np.sum(anomalies)

    return anomalies, anomaly_count


def clear_data(data_np):
    
    anomalies, anomaly_count = detect_anomalies_stat(data_np, k=6)
    
    modified_matrix = replace_anomalies_with_local_mean(data_np, anomalies, window_size=7)

    return modified_matrix