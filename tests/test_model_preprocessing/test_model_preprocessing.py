import sys

sys.path.append("/home/hadoop/sequence_models")

import pytest
from src.model_preprocessing.model_preprocessing import (
    convert_prod_to_index,
    pad_baskets,
    create_test_valid,
    create_x_y_list,
    pad_cust_seq,
)


@pytest.fixture(scope="function")
def cust_list(spark) -> list:

    """Get mock list of customers and items to test convert_prod_to_index"""
    cust_list = [
        [
            ["PRD0901848", "PRD0902030", "PRD0902242"],
            ["PRD0900912", "PRD0900773"],
            ["PRD0901136", "PRD0900121", "PRD0903256", "PRD0902394", "PRD0900569"],
        ],
        [
            ["PRD0903587", "PRD0901009"],
            ["PRD0901608", "PRD0901586", "PRD0901652", "PRD0900150", "PRD0901009"],
            ["PRD0903587", "PRD0903489", "PRD0900272"],
            ["PRD0900097", "PRD0900360"],
            ["PRD0903678", "PRD0901225", "PRD0900097", "PRD0904421"],
            ["PRD0903806"],
        ],
        [["PRD0903633", "PRD0903532"], ["PRD0902666", "PRD0904723"]],
        [
            ["PRD0901982", "PRD0900417"],
            ["PRD0903678", "PRD0900932", "PRD0900121"],
            ["PRD0903170"],
            ["PRD0901113", "PRD0903228"],
            ["PRD0900121"],
        ],
        [["PRD0903678"]],
    ]

    return cust_list


@pytest.fixture(scope="function")
def prod_dict(spark) -> dict:
    """Get mock product dictionary to test convert_prod_to_index"""

    prod_dict = {
        "UNK": 0,
        "PRD0901848": 1,
        "PRD0900121": 2,
        "PRD0900569": 3,
        "PRD0901009": 4,
    }

    return prod_dict


@pytest.fixture(scope="function")
def train_test_valid_perc():
    """Set the percentages for train, test and validation splits"""
    train_test_valid_perc = [0.33, 0.33, 0.34]
    return train_test_valid_perc


# Get the observation numbers needed to split the customer list
@pytest.fixture(scope="function")
def cust_list_train_test():
    """Get test observations required for test_create_test_valid function"""
    cust_list_train_test = [
        [[2, 3, 4, 0, 0], [3, 2, 5, 7, 4]],
        [[2, 2, 3, 3, 3], [2, 0, 0, 0, 0]],
        [[5, 5, 5, 5, 5], [1, 1, 1, 1, 1]],
        [[4, 4, 4, 4, 0], [0, 9, 0, 8, 7]],
        [[2, 3, 4, 1, 0], [7, 4, 3, 1, 0]],
        [[5, 6, 7, 2, 3], [0, 0, 0, 0, 0]],
    ]
    return cust_list_train_test


def test_convert_prod_to_index(cust_list, prod_dict):
    """Test function convert_prod_to_index
    GIVEN:  Array of lists of product ID's purchased by customers and a dictionary mapping of product ID to an index
    THEN:  Return an array of lists with product ID's mapped to integers in the product dictionary

    """

    expected_cust_index = [
        [[1, 0, 0], [0, 0], [0, 2, 0, 0, 3]],
        [[0, 4], [0, 0, 0, 0, 4], [0, 0, 0], [0, 0], [0, 0, 0, 0], [0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0, 2], [0], [0, 0], [2]],
        [[0]],
    ]

    cust_index = convert_prod_to_index(cust_list, prod_dict)

    msg = f"expected output in cust_index and received output do not match"

    assert cust_index == expected_cust_index, msg


@pytest.mark.parametrize(
    "cust_bask_list, max_items, expected_cust_bask_list_pad",  # parameter for the test function
    [
        ([[[1, 0, 0], [0, 0], [0, 2, 0, 0, 3]]], 5, [[[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
        (
            [[[0, 4], [0, 0, 0, 0, 4], [0, 0, 0], [0, 0], [0, 0, 0, 0], [0]]],
            5,
            [[
                [0, 4, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]],
        ),
        ([[[0, 0], [0, 0]]], 5, [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
        (
            [[[0, 0], [0, 0, 2], [0], [0, 0], [2]]],
            5,
            [[
                [0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0],
            ]],
        ),
        ([[[0]]], 3, [[[0, 0, 0]]]),
    ],
)
def test_cust_bask_list_pad(cust_bask_list, max_items, expected_cust_bask_list_pad):
    """Test function cust_bask_list_pad
    GIVEN:  Array of lists of product ID's purchased by customers and a value for the maximum number of items in
    a basket
    THEN:  Return an array of lists with each list padded to the length of the maximum basket and any basket >= the
    specified maximum removed

    """

    cust_bask_list_pad = pad_baskets(cust_bask_list, max_items)

    msg = (
        f"expected output in cust_bassk_list_pad and received output do not match for "
        f"records {cust_bask_list} and {expected_cust_bask_list_pad}"
    )

    assert cust_bask_list_pad == expected_cust_bask_list_pad, msg


def test_create_test_valid(cust_list_train_test, train_test_valid_perc):
    """Test function cust_bask_list_pad
    GIVEN:  Array of lists of product integers representing items purchased by customers and an array specifying values
    for train, test and validation sets
    THEN:  Tuple of lists containing data for train, test and validation

    """

    expected_train = [
        [[2, 2, 3, 3, 3], [2, 0, 0, 0, 0]],
        [[5, 5, 5, 5, 5], [1, 1, 1, 1, 1]],
    ]
    expected_test = [
        [[5, 6, 7, 2, 3], [0, 0, 0, 0, 0]],
        [[2, 3, 4, 1, 0], [7, 4, 3, 1, 0]],
    ]
    expected_valid = [
        [[2, 3, 4, 0, 0], [3, 2, 5, 7, 4]],
        [[4, 4, 4, 4, 0], [0, 9, 0, 8, 7]],
    ]
    train, test, valid = create_test_valid(cust_list_train_test, train_test_valid_perc)

    msg_train_match = "expected output in train and received output do not match"
    msg_test_match = "expected output in test and received output do not match"
    msg_valid_match = "expected output in valid and received output do not match"

    assert train == expected_train, msg_train_match
    assert test == expected_test, msg_test_match
    assert valid == expected_valid, msg_valid_match


@pytest.mark.parametrize(
    "cust_list_x_y, num_prods, expected_x, expected_y",  # parameter for the test function
    [
        ([[[2, 2, 3, 3, 3], [2, 0, 0, 0, 0]]], 5, [[[2, 2, 3, 3, 3]]], [[1, 0, 1, 0, 0, 0]]),
        ([[[5, 5, 5, 5, 5], [1, 2, 2, 5, 4]]], 5, [[[5, 5, 5, 5, 5]]], [[0, 1, 1, 0, 1, 1]]),
    ],
)
def test_create_x_y_list(cust_list_x_y, num_prods, expected_x, expected_y):
    """Test function create_x_y_list
    GIVEN:  list of product ID's purchased by customers and a value for total number of products
    THEN:  return 2 lists, one with x values that are all list values except the list entry and one with
           the last entry converted into multi binary label

    """

    cust_list_x, cust_list_y = create_x_y_list(cust_list_x_y, num_prods)

    msg_x = "expected output in cust_list_x and received output do not match"
    msg_y = "expected output in cust_list_y and received output do not match"

    assert cust_list_x == expected_x, msg_x
    assert cust_list_y == expected_y, msg_y


@pytest.mark.parametrize(
    "cust_list, max_seq, max_items, expected_cust_list_pad",  # parameter for the test function
    [
        (
            [[[5, 3, 8, 9, 20], [9, 2, 1, 2, 30]]],
            3,
            5,
            [[[5, 3, 8, 9, 20], [9, 2, 1, 2, 30], [0, 0, 0, 0, 0]]]
        ),
        (
            [[[3, 2, 1, 5, 6],[0, 1, 1, 0, 1, 1]]],
            3,
            5,
            [[[3, 2, 1, 5, 6], [0, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0]]],
        ),
    ],
)
def test_pad_cust_seq(cust_list, max_seq, max_items, expected_cust_list_pad):
    """Test function pad_cust_seq
        GIVEN: Array of lists of customer transactions
        THEN: Pad each customer list to have the same length
    """

    cust_list_pad = pad_cust_seq(cust_list, max_seq, max_items)

    msg = "expected output in cust_list_pad and received output do not match"

    assert cust_list_pad == expected_cust_list_pad, msg
