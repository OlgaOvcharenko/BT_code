import functools
import math
import time

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import monotonically_increasing_id, explode, col, row_number, udf, lit
from pyspark.sql.types import IntegerType, ArrayType
from sklearn.utils import shuffle

from error_distribution import ErrorDistribution


class DataGen:
    """A data generator class"""

    def __init__(self):
        self.clean_data_scaled = None
        self.errors_mask = None

    def generate(self, data: pd.DataFrame, scaling_factor: float, new_dim: int, error_distribution: ErrorDistribution, dirty_schema):
        if data.shape[0] < 5000 and round(scaling_factor) != scaling_factor:  # FIXME what is the correct number to argue?
        # if False:
            self._generate_hash(data=data, scaling_factor=scaling_factor, new_dim=new_dim,
                                                    error_distribution=error_distribution, dirty_schema=dirty_schema)
        else:
            self._generate_random_no_transformation(data=data, scaling_factor=scaling_factor,
                                                    new_dim=new_dim, error_distribution=error_distribution, dirty_schema=dirty_schema)

    def _generate_hash(self, data, scaling_factor: float, new_dim: int, error_distribution: ErrorDistribution, dirty_schema):
        rows, cols = data.shape[0], data.shape[1]
        # data_scaled = pd.DataFrame(columns=data.columns)
        data_scaled = pd.DataFrame(columns=dirty_schema)

        # Hash dataset
        hashed_data = {}
        for r in range(rows):
            data_tuple = tuple(data.loc[r])
            if not hashed_data.get(data_tuple):
                hashed_data[data_tuple] = 1
            else:
                hashed_data[data_tuple] += 1

        for tuple_count in hashed_data.items():
            num_copies = round(tuple_count[1] * scaling_factor)
            if num_copies > 0:
                rows_to_add = pd.DataFrame((tuple_count[0],) * num_copies, columns=data.columns)
                data_scaled = pd.concat([data_scaled, rows_to_add], axis=0, ignore_index=True)

        self.clean_data_scaled = data_scaled
        self.clean_data_scaled = shuffle(data_scaled)
        # self.errors_mask = np.full(self.clean_data_scaled.shape, False, dtype=bool)

    def _generate_random_no_transformation(self, data, scaling_factor: float, new_dim: int,
                                           error_distribution: ErrorDistribution, dirty_schema):
        rows, cols = data.shape[0], data.shape[1]
        n_copies = math.floor(scaling_factor)

        # Replicate the data
        data_scaled = data.astype(dirty_schema)
        for i in range(0, n_copies - 1):
            data_scaled = pd.concat([data_scaled, data], axis=0, ignore_index=True)

        # Randomly modify data to exact scaling factor
        if n_copies < scaling_factor:
            rows_to_add = new_dim - (n_copies * rows)
            random_indices = np.random.choice(data.shape[0], size=rows_to_add, replace=False)
            data_scaled = pd.concat([data_scaled, data.loc[random_indices]], axis=0, ignore_index=True)

        self.clean_data_scaled = shuffle(data_scaled)
        # self.errors_mask = np.full(self.clean_data_scaled.shape, False, dtype=bool)

    def _generate_random_full(self, data, scaling_factor: float, new_dim: int,
                                           error_distribution: ErrorDistribution, dirty_schema):
        # FIXME
        rows, cols = data.shape[0], data.shape[1]
        n_copies = math.floor(scaling_factor)

        # Replicate the data
        data_scaled = data.astype(dirty_schema)

        random_indices = np.random.choice(data.shape[0], size=new_dim-rows, replace=True)
        new_data = data.loc[random_indices]
        data_scaled = pd.concat([data_scaled, new_data], axis=0, ignore_index=True)
        self.clean_data_scaled = shuffle(data_scaled)

    def _generate_slice(self, data, scaling_factor: float, new_dim: int,
                                           error_distribution: ErrorDistribution, dirty_schema):
        rows, cols = data.shape[0], data.shape[1]
        n_copies = math.floor(scaling_factor)
        # remaining_rows = new_dim - n_copies * rows

        # Replicate the data
        data_scaled = data
        for i in range(0, n_copies - 1):
            data_scaled = pd.concat([data_scaled, data], axis=0, ignore_index=True)

        # Randomly modify data to exact scaling factor
        if n_copies < scaling_factor:
            rows_to_add = new_dim - (n_copies * rows)
            data_scaled = pd.concat([data_scaled, data.loc[0:rows_to_add]], axis=0, ignore_index=True)

        self.clean_data_scaled = shuffle(data_scaled)
        # self.errors_mask = np.full(self.clean_data_scaled.shape, False, dtype=bool)


class DataGenSP:
    """A data generator class"""

    def __init__(self):
        self.clean_data_scaled = None
        self.errors_mask = None
        self.num_rows = 0
        self.num_cols = 0
        self.index_col_name = "seq_id"

    # def generate(self, spark: SparkSession, data: pd.DataFrame, scaling_factor: float, new_dim: int, dirty_schema):
    #     self._generate_random_no_transformation(spark=spark, data=data, scaling_factor=scaling_factor,
    #                                             new_dim=new_dim, dirty_schema=dirty_schema)

    def generate(self, clean_df: pd.DataFrame, spark: SparkSession, data_path: str, scaling_factor: float, new_dim: int, rows: int, cols: int,
                 dirty_path: str):
        self._generate_copy_row(pandas_df=clean_df, spark=spark, data_path=data_path, scaling_factor=scaling_factor,
                                new_dim=new_dim, rows=rows, cols=cols, dirty_path=dirty_path)

    def _generate_random_no_transformation_append(self, spark: SparkSession, data, scaling_factor: float, new_dim: int, dirty_schema):
        rows, cols = data.shape[0], data.shape[1]
        self.num_rows = rows
        self.num_cols = cols
        n_copies = math.floor(scaling_factor)
        data = data.astype(dirty_schema)

        # Replicate the data
        data_scaled, append_data = spark.createDataFrame(data), spark.createDataFrame(data)
        for i in range(0, n_copies - 1):
            data_scaled = functools.reduce(lambda df1, df2: df1.union(df2),
                                           [data_scaled, append_data])
            self.num_rows += rows

        # Randomly modify data to exact scaling factor
        if n_copies < scaling_factor:
            rows_to_add = new_dim - (n_copies * rows)
            random_indices = np.random.choice(rows, size=rows_to_add, replace=False)
            sliced_part = data.loc[random_indices]
            append_data = spark.createDataFrame(sliced_part)
            data_scaled = functools.reduce(lambda df1, df2: df1.union(df2),
                                           [data_scaled, append_data])
            self.num_rows += len(random_indices)

        # Add index
        data_scaled = data_scaled.select("*").withColumn(self.index_col_name, monotonically_increasing_id())
        # data_scaled = data_scaled.repartition(16, self.index_col_name)
        self.clean_data_scaled = data_scaled.cache()

    def _generate_copy_row(self, pandas_df: pd.DataFrame, spark: SparkSession, data_path: str, scaling_factor: float, new_dim: int,
                           rows: int, cols: int, dirty_path: str):
        start_time = time.time()
        n_copies = math.floor(scaling_factor)
        self.num_rows = n_copies * rows
        self.num_cols = cols

        #  Read dirty schema and clean data
        dirty_schema = spark.read.option("header", "true").csv(dirty_path).schema
        data = spark.read.option("header", "true").schema(dirty_schema).csv(data_path)

        # Get sample
        data = data.rdd.zipWithIndex().toDF()
        for pandas_col in pandas_df.columns:
            data = data.withColumn(pandas_col, data["_1"].getItem(pandas_col))
            data = data.drop(data["_1"].getItem(pandas_col))
        data = data\
            .withColumnRenamed("_2", self.index_col_name)\
            .drop("_1")

        # Replicate the data
        n_to_array = udf(lambda n: [n + m * rows for m in range(n_copies)], ArrayType(IntegerType()))
        data = data.withColumn(self.index_col_name, n_to_array(data.seq_id))
        data = data.withColumn(self.index_col_name, explode(data.seq_id))

        print(f"Created integer part {time.time()-start_time}")
        # Add float part
        # Randomly modify data to exact scaling factor
        if n_copies < scaling_factor:
            rows_to_add = new_dim - (n_copies * rows)
            random_indices = np.random.choice(rows, size=rows_to_add, replace=False)
            sliced_part = pandas_df.loc[random_indices]
            append_data = spark.createDataFrame(sliced_part)

            # FIXME
            append_data = append_data.select("*").withColumn(self.index_col_name, monotonically_increasing_id())
            window_index = Window.partitionBy(self.index_col_name).orderBy(col(self.index_col_name))
            append_data = append_data.withColumn(self.index_col_name, row_number().over(window_index))

            append_data = append_data.rdd.zipWithIndex().toDF()
            for pandas_col in pandas_df.columns:
                append_data = append_data.withColumn(pandas_col, append_data["_1"].getItem(pandas_col))
                append_data = append_data.drop(append_data["_1"].getItem(pandas_col))
            append_data = append_data \
                .withColumn("_2", append_data._2 + (self.num_rows + 1)) \
                .withColumnRenamed("_2", self.index_col_name) \
                .drop("_1")

            val_to_add = self.num_rows
            append_data = append_data.withColumn(self.index_col_name, col(self.index_col_name) + lit(val_to_add))
            data = functools.reduce(lambda df1, df2: df1.union(df2), [data, append_data])

            self.num_rows += len(random_indices)

        data = data.repartition(DataGenSP.get_number_partitions(self.num_rows, self.num_cols), self.index_col_name)
        self.clean_data_scaled = data.cache()

        print(f"Created data {time.time()-start_time}")

        print(f"Num of rows {self.num_rows}")
        # print(f"Distinct values after scaling {self.clean_data_scaled.select(col(self.index_col_name)).distinct().count()}")


    @staticmethod
    def get_number_partitions(rows: int, cols: int):
        num_partitions = math.floor((rows * cols) / 100000)  # TODO 50000

        # num_partitions = math \
        #     .ceil((data.memory_usage(deep=True).sum() * scaling_factor) / 100000000)  # Each block 100 MB

        return max(num_partitions, 16)
