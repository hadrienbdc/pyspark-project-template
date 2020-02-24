from hamcrest import *
import unittest2
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from pyspark_project.feature import SparkDataFrameConverter


class TestSparkDataframeConverter(unittest2.TestCase):

    def test_should_convert_simple_list_into_dataframe(self):
        # Given texts and labeld
        texts = ['text1', 'text2']
        labels = ['label1', 'label2']

        # When conversion is applied
        converter = SparkDataFrameConverter()
        df = converter.convert(texts, labels)

        # Then the result should be a dataframe
        assert_that(df, instance_of(DataFrame))
        # And the dataframe should have 2 columns named text and label
        field_names = map(lambda field: field.name, df.schema.fields)
        assert_that(field_names, contains('text', 'label'))
        # And the values should be zipped together
        line_text_1 = df.where(col('text') == 'text1').head()
        assert_that(line_text_1['label'], equal_to('label1'))
        line_text_2 = df.where(col('text') == 'text2').head()
        assert_that(line_text_2['label'], equal_to('label2'))
