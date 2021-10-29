import collections
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window


def create_item_cust_arrays(df: DataFrame) -> tuple:
    """
    Function to create the customer and item arrays needed for model training

    Parameters
    ----------
    df: DataFrame
      DataFrame containing CUST_CODE, BASKET_ID, PROD_CODE

    Returns
    -------
    cust_list: list
      List containing an array of all customers, baskets and items
    item_list: list
      List containing an array of all items purchased in all baskets by all customers
    customer_id_list: list
      List of customer CUST_CODE in same order as cust_list

    """

    w = Window.partitionBy("CUST_CODE").orderBy("BASKET_ID")

    # First rollup to customer and basket collecting the items as a list
    cust_bask_rollup = df.groupBy("CUST_CODE", "BASKET_ID").agg(
        F.collect_set("PROD_CODE").alias("BASK_SET"),
    )

    # Get the customer, basket, item list
    cust_rollup_items = (
        cust_bask_rollup.withColumn("sorted_list", F.collect_list("BASK_SET").over(w))
        .groupBy("CUST_CODE")
        .agg(F.max("sorted_list").alias("CUST_BASK_SET"))
    )

    cust_list = (
        cust_rollup_items.select("CUST_BASK_SET").rdd.flatMap(lambda x: x).collect()
    )

    customer_id_list = (
        cust_rollup_items.select("CUST_CODE")
        .dropDuplicates()
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    # Create a list of items (not deduped)
    item_list = df.select("PROD_CODE").rdd.flatMap(lambda x: x).collect()

    return cust_list, item_list, customer_id_list


def generate_prod_dictionaries(item_list: list, num_prods: int) -> tuple:
    """
    Function to create dictionaries mapping PROD_CODE to an index and back again

    Parameters
    ----------
    item_list: list
      List containing an array of all items purchased in all baskets by all customers
    num_prods: int
      The number of products to keep (will keep only the top X products by number of transactions
      the product was purchased in)

    Returns
    -------
    prod_dictionary: dict
      Dictionary containing the mapping of PROD_CODE to index
    reversed_prod_dictionary: dict
     Dictionary containing the mapping of index to PROD_CODE

    """

    # Create counts of products
    count = [["UNK", -1]]  # Placeholder for unknown
    count.extend(collections.Counter(item_list).most_common(num_prods - 1))

    # Create a dictionary mapping of product to index
    prod_dictionary = dict()
    for prod, _ in count:
        prod_dictionary[prod] = len(prod_dictionary)

    # Create a reversed mapping of index to product
    reversed_prod_dictionary = dict(
        zip(prod_dictionary.values(), prod_dictionary.keys())
    )

    return prod_dictionary, reversed_prod_dictionary