import numpy as np
import pytest
from src.generators.lstm_generator import lstm_data_generator


@pytest.fixture(scope="function")
def cust_list_x():
    """Set the "cust_x_list to test generator"""

    cust_list_x = [
        [[1, 2, 4, 10, 5, 9, 3, 16, 11, 15], [3, 6, 10, 20, 14, 12, 8, 2, 1, 0]],
        [[20, 15, 10, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[11, 8, 1, 2, 5, 3, 16, 0, 0, 0], [2, 18, 0, 0, 0, 0, 0, 0, 0, 0]],
    ]

    return cust_list_x


@pytest.fixture(scope="function")
def cust_list_y():
    """Set the "cust_y_list to test generator"""

    cust_list_y = [
        [
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        ],
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        ],
    ]

    return cust_list_y


def test_lstm_data_generator(cust_list_x, cust_list_y):
    """Test generator for lstm model
    GIVEN:  Array of lists of customer baskets (x values) and binary encoded lists of items in the basket
            (y values)
    THEN:  Return  a specified batch of x and y lists in an array for training

    """

    generator = lstm_data_generator(
        batch_size=2,
        cust_list_x=cust_list_x,
        cust_list_y=cust_list_y,
        shuffle=False,
    )

    test_batch_x, test_batch_y = next(generator)

    expected_test_batch_x = np.array(
        [
            [[1, 2, 4, 10, 5, 9, 3, 16, 11, 15], [3, 6, 10, 20, 14, 12, 8, 2, 1, 0]],
            [[20, 15, 10, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        ]
    )

    expected_test_batch_y = np.array(
        [
            [
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
            ],
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )

    msg_x = f"expected output in test_batch_x {expected_test_batch_x} and received output {expected_test_batch_x} do not match"
    msg_y = f"expected output in test_batch_y {expected_test_batch_y} and received output {expected_test_batch_y} do not match"

    assert (expected_test_batch_x == test_batch_x).all(), msg_x
    assert (expected_test_batch_y == test_batch_y).all, msg_y
