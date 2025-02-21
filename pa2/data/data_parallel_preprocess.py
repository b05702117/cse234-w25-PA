import numpy as np

def split_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    mp_size: int,
    dp_size: int,
    rank: int,
):
    """The function for splitting the dataset uniformly across data parallel groups

    Parameters
    ----------
        x_train : np.ndarray float32
            the input feature of MNIST dataset in numpy array of shape (data_num, feature_dim)

        y_train : np.ndarray int32
            the label of MNIST dataset in numpy array of shape (data_num,)

        mp_size : int
            Model Parallel size

        dp_size : int
            Data Parallel size

        rank : int
            the corresponding rank of the process

    Returns
    -------
        split_x_train : np.ndarray float32
            the split input feature of MNIST dataset in numpy array of shape (data_num/dp_size, feature_dim)

        split_y_train : np.ndarray int32
            the split label of MNIST dataset in numpy array of shape (data_num/dp_size, )

    Note
    ----
        - Data is split uniformly across data parallel (DP) groups.
        - All model parallel (MP) ranks within the same DP group share the same data.
        - The data length is guaranteed to be divisible by dp_size.
        - Do not shuffle the data indices as shuffling will be done later.
    """
    data_num = x_train.shape[0] # total number of data

    # Ensure data_num is divisible by dp_size
    assert data_num % dp_size == 0, "data_num is not divisible by dp_size"

    # Determine the DP group ID
    dp_group_id = rank // mp_size

    # Compute the split size per DP group
    split_size = data_num // dp_size

    # Get the data slice corresponding to the current DP group
    start_idx = dp_group_id * split_size
    end_idx = start_idx + split_size

    split_x_train = x_train[start_idx:end_idx]
    split_y_train = y_train[start_idx:end_idx]

    return split_x_train, split_y_train