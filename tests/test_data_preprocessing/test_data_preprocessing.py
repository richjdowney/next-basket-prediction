import sys

sys.path.append("/home/hadoop/sequence_models")
import pytest
from pyspark.sql import DataFrame
from src.data_preprocessing.data_preprocessing import (
    create_item_cust_arrays,
    generate_prod_dictionaries,
)


@pytest.fixture(scope="function")
def df(spark) -> DataFrame:
    """Get the sample transaction data for pre-processing tests"""
    df = spark.createDataFrame(
        [
            ("CUST0000000031", "994102600163819", "PRD0901848"),
            ("CUST0000000031", "994102600163819", "PRD0902030"),
            ("CUST0000000031", "994102600163819", "PRD0902242"),
            ("CUST0000000031", "994102700162751", "PRD0900773"),
            ("CUST0000000031", "994102700162751", "PRD0900912"),
            ("CUST0000000031", "994102900161864", "PRD0900121"),
            ("CUST0000000031", "994102900161864", "PRD0900569"),
            ("CUST0000000031", "994102900161864", "PRD0901136"),
            ("CUST0000000031", "994102900161864", "PRD0902394"),
            ("CUST0000000031", "994102900161864", "PRD0903256"),
            ("CUST0000000068", "994102600163843", "PRD0901009"),
            ("CUST0000000068", "994102600163843", "PRD0903587"),
            ("CUST0000000068", "994102600163844", "PRD0900150"),
            ("CUST0000000068", "994102600163844", "PRD0901009"),
            ("CUST0000000068", "994102600163844", "PRD0901586"),
            ("CUST0000000068", "994102600163844", "PRD0901608"),
            ("CUST0000000068", "994102600163844", "PRD0901652"),
            ("CUST0000000068", "994102700162775", "PRD0900272"),
            ("CUST0000000068", "994102700162775", "PRD0903489"),
            ("CUST0000000068", "994102700162775", "PRD0903587"),
            ("CUST0000000068", "994102800162985", "PRD0900097"),
            ("CUST0000000068", "994102800162985", "PRD0900360"),
            ("CUST0000000068", "994102800162986", "PRD0900097"),
            ("CUST0000000068", "994102800162986", "PRD0904421"),
            ("CUST0000000068", "994102800162986", "PRD0903678"),
            ("CUST0000000068", "994102800162986", "PRD0901225"),
            ("CUST0000000068", "994102800162987", "PRD0903806"),
            ("CUST0000000131", "994103700194219", "PRD0903532"),
            ("CUST0000000131", "994103700194219", "PRD0903633"),
            ("CUST0000000131", "994104000154991", "PRD0902666"),
            ("CUST0000000131", "994104000154991", "PRD0904723"),
            ("CUST0000000164", "994102600163913", "PRD0900417"),
            ("CUST0000000164", "994102600163913", "PRD0901982"),
            ("CUST0000000164", "994102600163914", "PRD0900121"),
            ("CUST0000000164", "994102600163914", "PRD0900932"),
            ("CUST0000000164", "994102600163914", "PRD0903678"),
            ("CUST0000000164", "994102800163052", "PRD0903170"),
            ("CUST0000000164", "994102800163053", "PRD0901113"),
            ("CUST0000000164", "994102800163053", "PRD0903228"),
            ("CUST0000000164", "994102800163054", "PRD0900121"),
            ("CUST0000000180", "994102600163931", "PRD0903678"),
        ],
        [
            "CUST_CODE",
            "BASKET_ID",
            "PROD_CODE",
        ],
    )

    return df


@pytest.fixture(scope="function")
def item_list(spark) -> list:
    """Get list of items needed to test the function generate_prod_dictionaries"""
    item_list = [
        "PRD0901848",
        "PRD0900569",
        "PRD0901009",
        "PRD0901848",
        "PRD0902666",
        "PRD0900121",
        "PRD0901848",
        "PRD0900121",
    ]

    return item_list


@pytest.fixture(scope="function")
def num_prods() -> int:
    """Assign the number of products to use to test the function generate_prod_dictionaries"""
    return 5


def test_create_item_cust_arrays(df):

    """Test function create_item_cust_arrays
    GIVEN:  List of unique customer ID's, basket ID's and items
    THEN:  Return tuple of lists:
        1.)  Array of lists of products in each basket from each customer
        2.)  List of items in every basket (not de-duped)
        3.)  List of unique customer_ids in same order as original df
    """

    expected_cust_list = [
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

    expected_item_list = [
        "PRD0901848",
        "PRD0902030",
        "PRD0902242",
        "PRD0900773",
        "PRD0900912",
        "PRD0900121",
        "PRD0900569",
        "PRD0901136",
        "PRD0902394",
        "PRD0903256",
        "PRD0901009",
        "PRD0903587",
        "PRD0900150",
        "PRD0901009",
        "PRD0901586",
        "PRD0901608",
        "PRD0901652",
        "PRD0900272",
        "PRD0903489",
        "PRD0903587",
        "PRD0900097",
        "PRD0900360",
        "PRD0900097",
        "PRD0904421",
        "PRD0903678",
        "PRD0901225",
        "PRD0903806",
        "PRD0903532",
        "PRD0903633",
        "PRD0902666",
        "PRD0904723",
        "PRD0900417",
        "PRD0901982",
        "PRD0900121",
        "PRD0900932",
        "PRD0903678",
        "PRD0903170",
        "PRD0901113",
        "PRD0903228",
        "PRD0900121",
        "PRD0903678",
    ]

    expected_customer_id_list = [
        "CUST0000000031",
        "CUST0000000068",
        "CUST0000000131",
        "CUST0000000180",
        "CUST0000000164",
    ]

    cust_list, item_list, customer_id_list = create_item_cust_arrays(df)

    msg_cust_list = f"expected output in cust_list and received output do not match"
    msg_item_list = f"expected output in item_list and received output do not match"
    msg_customer_id_list = (
        f"expected output in customer_id_list and received output do not match"
    )

    assert cust_list == expected_cust_list, msg_cust_list
    assert item_list == expected_item_list, msg_item_list
    assert customer_id_list == expected_customer_id_list, msg_customer_id_list


def test_generate_prod_dictionaries(item_list, num_prods):
    """Test function generate_prod_dictionaries
        GIVEN:  List of product ID's and total number of items to use
        THEN:  Return tuple of dictionaries:
            1.)  Dictionary mapping the product ID's to an integer in order of most common item to least common
                 with UNK for any items ranked above the value for num_prods
            2.)  A reversed dictionary mapping the integer ID's back to the product ID's

        """

    expected_prod_dictionary = {
        "UNK": 0,
        "PRD0901848": 1,
        "PRD0900121": 2,
        "PRD0900569": 3,
        "PRD0901009": 4,
    }
    expected_reverse_prod_dictionary = {
        0: "UNK",
        1: "PRD0901848",
        2: "PRD0900121",
        3: "PRD0900569",
        4: "PRD0901009",
    }

    prod_dictionary, reversed_prod_dictionary = generate_prod_dictionaries(item_list, num_prods)

    msg_expected_prod_dict = (
        f"expected output in prod_dictionary and received output do not match"
    )
    msg_expected_reversed_prod_dict = (
        f"expected output in reversed_prod_dictionary and received output do not match"
    )

    assert prod_dictionary == expected_prod_dictionary, msg_expected_prod_dict
    assert reversed_prod_dictionary == expected_reverse_prod_dictionary, msg_expected_reversed_prod_dict
