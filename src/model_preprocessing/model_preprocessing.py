import random as rnd
from sklearn.preprocessing import MultiLabelBinarizer


def convert_prod_to_index(cust_list: list, prod_dict: dict) -> list:
    """
    Function to convert lists of customers/baskets/products to customer/baskets/product indices
    as required by the downstream sequence models

    Parameters
    ----------
    cust_list: list
      List of customers and PROD_CODE for items purchased in each transaction
    prod_dict: dict
        Dictionary mapping of PROD_CODE to index

    Returns
    -------
    cust_list_index: list
      List of customers and the index for items purchased

    """

    cust_index = list()

    # Customer loop
    for i in range(0, len(cust_list)):

        cust = cust_list[i]

        # Basket loop
        bask_list = list()
        for bask in cust:
            # Product loop
            prod_list = list()
            for prod in bask:

                if prod in prod_dict:
                    item_integer = prod_dict[prod]
                else:
                    item_integer = 0  # dictionary['UNK']
                prod_list.append(item_integer)
            bask_list.append(prod_list)
        cust_index.append(bask_list)

    return cust_index


def pad_baskets(cust_list: list, max_items: int) -> list:
    """
    Function to pad all customer baskets to a standard length and remove any
    baskets that contain more than the maximum specified number of items

    Parameters
    ----------
    cust_list: list
        List of customers and items purchased in each transaction
    max_items: int
        Maximum number of items allowed for basket to be used in modeling

    Returns
    -------
    cust_list_pad: list
        List of customers and items purchased with padding to ensure baskets are
        a standard length

    """

    cust_list_pad = list()

    # Customer loop
    for i in range(0, len(cust_list)):

        cust = cust_list[i]

        # Basket loop
        bask_list = list()
        for bask in cust:

            # Only keep the transaction if it has fewer than the maximum allowed number of items
            if len(bask) < max_items:
                # Pad the basket to a standard length
                pad = [0] * (max_items - len(bask))
                cust_prod_list_pad = bask + pad

                # Append to the customers list of transactions
                bask_list.append(cust_prod_list_pad)

        # append the current customer baskets to the cumulative list of all customers
        cust_list_pad.append(bask_list)

    return cust_list_pad


def create_test_valid(cust_list: list, train_test_valid_perc: list) -> tuple:
    """
    Function to create train, test and validation lists for input to the LSTM

    Parameters
    ----------
    cust_list: list
      List of customers and items purchased in each transaction
    train_test_valid_perc: list
      Proportion of customers for training, testing and validation - proportions should
      sum to 1

    Returns
    -------
    train: list
      List of customers and items purchased in each transaction for train set
    test: list
      List of customers and items purchased in each transaction for test set
    valid: list
      List of customers and items purchased in each transaction for validation set
    """

    assert (
            sum(train_test_valid_perc) == 1
    ), "The sum of train, test and validation proportions must sum to 1"

    # Shuffle the lists before selecting the training and validation sets
    rnd.seed(1234)
    rnd.shuffle(cust_list)

    # Get the observation numbers needed to split the customer list
    train_num = round(len(cust_list) * train_test_valid_perc[0])
    print(train_num)
    test_num = round(
        len(cust_list) * (train_test_valid_perc[0] + train_test_valid_perc[1])
    )
    valid_num = len(cust_list)

    # Get train, test and validation records
    train = cust_list[0:train_num]
    test = cust_list[train_num:test_num]
    valid = cust_list[test_num:valid_num]

    print(
        "There are {} records in train, {} in test and {} in validation".format(
            len(train), len(test), len(valid)
        )
    )

    return train, test, valid


def create_x_y_list(cust_list: list, num_prods: int, max_seq: int) -> tuple:
    """
    Function to create x and y lists for training - the y list will be the customers last basket
    as this is a next basket prediction

    Parameters
    ----------
    cust_list: list
      List of customers and items purchased in each transaction
    num_prods: int
      The number of products being modelled
    max_seq: int
      Length of the sequence required

    Returns
    -------
    cust_list_x: list
      List of customers and items purchased in each transaction for training (x list)
    cust_list_y: list
      List of customers and items purchased in each transaction for prediction (y list)
      NOTE:  y_list is converted into multi binary label for downstream modelling

    """

    cust_list_x = list()
    cust_list_y = list()

    # Loop over customers, keep last basket for y list
    for i in range(0, len(cust_list)):

        cust = cust_list[i]

        # Customer must have had at least 2 transactions and number of baskets must be less than
        # allowable max
        if 2 <= len(cust) < max_seq:
            x = cust[:-1]
            y = cust[-1]

            # Convert prods in y list to multi-label
            y_tuple = [tuple(y)]
            mlb = MultiLabelBinarizer(classes=[*range(0, num_prods + 1, 1)])
            y_multi_label = mlb.fit_transform(y_tuple)
            y_multi_label = y_multi_label.tolist()[0]

            cust_list_x.append(x)
            cust_list_y.append(y_multi_label)

    assert len(cust_list_x) == len(cust_list_y), "length of x and y are not equal"

    return cust_list_x, cust_list_y


def pad_cust_seq(cust_list: list, max_seq: int, max_items: int) -> list:
    """
    Function to pad customer sequences to equal lengths as required by downstream models

    Parameters
    ----------
    cust_list: list
      List of customers and items purchased in each transaction
    max_seq: int
      Length of the sequence required
    max_items: int
        Maximum number of items allowed for basket to be used in modeling

    Returns
    -------
    cust_list_pad: list
      List of customers and items purchased in each transaction with standard (padded) length

    """
    cust_list_pad = list()

    # Padding to insert to pad the sequence == to the max length of a basket
    full_seq_pad = [0] * max_items

    # Customer loop
    for i in range(0, len(cust_list)):
        cust = cust_list[i]

        if len(cust) < max_seq:
            pad = [full_seq_pad] * (max_seq - len(cust))

            # combine the customer list with the pad
            cur_cust_pad = cust + pad

            # append the padded tensor to the batch
            cust_list_pad.append(cur_cust_pad)

    return cust_list_pad
