from pyspark.sql import SparkSession


class SparkSessionHolder:
    def __init__(self):
        pass

    @staticmethod
    def get_spark_session():
        return SparkSession.builder\
            .master("local[4]")\
            .appName("sql.functions tests")\
            .getOrCreate()
