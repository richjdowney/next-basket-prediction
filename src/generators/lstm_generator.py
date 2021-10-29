import random as rnd
import numpy as np


def lstm_data_generator(
    batch_size: int,
    cust_list_x: list,
    cust_list_y: list,
    shuffle=True,
):
    """Generator function to yield batches of sequences of customer transactions for training an LSTM model, the
    generator outputs batches of sequences of baskets and items into x and y arrays.  Assumes that the input
    is already padded appropriately and y arrays are one-hot encoded

    Parameters
    ----------
    batch_size: int
      Size of the batch to yield
    cust_list_x: list
      List of customers and transactions from which to generate the batch
    cust_list_y: list
      List of customers and baskets to predict
    shuffle: bool
      Controls whether the sequence of customers should be shuffled between batches (defaults to True)

    Yields
    ------
    batch_x_arr: np.array
        Numpy array containing sequence of customers and transactions
    batch_y_arr: np.array
      Numpy array containing items in final basket to predict
    mask_arr: np.array
      Numpy array containing a list identifying if the observation (basket) in the sequence is padded i.e. the
      entire basket was added to pad the transaction sequence

    """

    # Initialize index
    index = 0

    # initialize current batch lists
    cur_x_batch = []
    cur_y_batch = []

    # Initialize batch count
    batch_count = 0

    # count the number of customers in the file
    num_custs = len(cust_list_x)

    # create an array with the indexes of customers that can be shuffled
    cust_index = [*range(num_custs)]

    # shuffle customer indexes if shuffle is set to True
    if shuffle:
        rnd.shuffle(cust_index)

    while True:
        # if the index is greater or equal than to the number of customers in the data
        if index >= num_custs:
            # then reset the index to 0
            index = 0
            # shuffle line indexes if shuffle is set to True
            if shuffle:
                rnd.shuffle(cust_index)

        # get the x and y lists for the chosen customer index
        cust_x = cust_list_x[cust_index[index]]
        cust_y = cust_list_y[cust_index[index]]

        cur_x_batch.append(cust_x)
        cur_y_batch.append(cust_y)

        # Add 1 to the batch count
        batch_count += 1

        # increment the index by one to pull the next customer record
        index += 1

        # if the current batch is now equal to the desired batch size
        if batch_count == batch_size:
            batch_x_arr = np.array(cur_x_batch)
            batch_y_arr = np.array(cur_y_batch)

            # yield batch_x_arr, batch_y_arr, mask_arr
            yield batch_x_arr, batch_y_arr

            # reset the current batch to an empty list
            cur_x_batch = []
            cur_y_batch = []
            batch_count = 0
