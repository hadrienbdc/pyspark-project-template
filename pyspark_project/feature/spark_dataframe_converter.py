import pandas as pd

from pyspark_project.utils import SparkSessionHolder


class SparkDataFrameConverter:
    def __init__(self):
        self.spark = SparkSessionHolder.get_spark_session()

    def convert(self, news_list, news_target):
        news_pdf = pd.DataFrame(data=news_list, columns=["text"])
        news_pdf["label"] = news_target
        return self.spark.createDataFrame(news_pdf)
