from pyspark.sql.types import StringType
from pyspark.sql import functions as F
from utils.logging_framework import log


class DataProfiling:
    """Profile the imported data"""

    def __init__(self, df, df_desc):
        """ Initialize the profiling object
        Parameters
        ----
        df : pyspark.sql.DataFrame
            DataFrame to be profiled
        df_desc : str
            String containing the name of the DataFrame being profiled
        """

        self.df = df
        self.df_desc = df_desc

    def top10_records(self):
        """Print the top 10 records from the DataFrame"""

        log.info("Top 10 records from DataFrame {}".format(self.df_desc))
        self.df.show(10)

    def print_schema(self):
        """Print the DataFrame schema"""

        log.info("Schema for DataFrame {}".format(self.df_desc))
        self.df.printSchema()

    def row_column_counts(self):
        """Print the number of rows and columns"""

        log.info(
            "DataFrame {} has {} columns and {} rows".format(
                self.df_desc, len(self.df.columns), self.df.count()
            )
        )

    def check_df_missing(self):
        """Check the entire DataFrame for duplicates across all rows and columns"""

        log.info(
            "Count of rows in DataFrame {}: {}".format(self.df_desc, self.df.count())
        )
        log.info(
            "Count of distinct rows in DataFrame {}: {}".format(
                self.df_desc, self.df.count()
            )
        )

    def check_missing_per_col(self):
        """Get the % of missing values per column"""

        log.info(
            "% of missing values per column in DataFrame {}".format(self.df_desc)
        )
        self.df.select(
            [
                (
                    F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)) / F.count("*")
                ).alias(c)
                for c in self.df.columns
            ]
        ).show()

    def top20_string_values(self):
        """Get the top 20 values for each string column"""

        log.info(
            "Top 20 unique values for string variables in DataFrame {}".format(
                self.df_desc
            )
        )
        str_cols = [
            f.name for f in self.df.schema.fields if isinstance(f.dataType, StringType)
        ]
        for col in str_cols:
            self.df.groupBy(col).count().orderBy(F.col("count").desc()).show(20)

    def num_col_profile(self):
        """Get Mean, min, max and standard deviation of numeric columns """

        log.info(
            "Mean, min, max and standard deviation for variables in DataFrame {}".format(
                self.df_desc
            )
        )
        num_cols = [
            f.name
            for f in self.df.schema.fields
            if not isinstance(f.dataType, StringType)
        ]
        for col in num_cols:
            self.df.agg(
                F.mean(col).alias("mean_" + col),
                F.max(col).alias("max_" + col),
                F.min(col).alias("min_" + col),
                F.stddev(col).alias("std_" + col),
            ).show()