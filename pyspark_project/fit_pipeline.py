import logging
import logging.config
import os
import argparse

from pyspark.ml import PipelineModel

from pyspark_project.feature import FeatureEngineering, SparkDataFrameConverter
from pyspark_project.loader import DataLoader
from pyspark_project.model import Model


def arg_parser():
    current_working_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Simple project example')
    parser.add_argument('--fe-path',
                        type=str,
                        help="Where to save the feature engineering pipeline",
                        dest='fe_pipeline_save_path',
                        required=False,
                        default=os.path.join(current_working_dir, '../models/fe_pipeline_spark'))
    parser.add_argument('--classifier-path',
                        type=str,
                        help="Where to save the resulting classifier",
                        dest='classifier_save_path',
                        required=False,
                        default=os.path.join(current_working_dir, '../models/classifier_spark'))
    return parser


def main():
    logging.config.fileConfig('%s/../logging.conf' % os.path.dirname(os.path.abspath(__file__)))
    logger = logging.getLogger(name="simpleExample")

    parser = arg_parser()
    args = parser.parse_args()

    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    fe_pipeline_save_path = args.fe_pipeline_save_path
    classifier_save_path = args.classifier_save_path

    logger.info("Load Data")
    newsgroup_data_loader = DataLoader(categories=categories)
    twenty_train, _ = newsgroup_data_loader.load_data()

    logger.info("Transform raw data into Spark DataFrame")
    spark_df_converter = SparkDataFrameConverter()
    twenty_train_df = spark_df_converter.convert(twenty_train.data, twenty_train.target)

    logger.info("Fit and Save Feature Engineering Pipeline")
    fe = FeatureEngineering()
    fe.fit_and_save_pipeline(
        df_train=twenty_train_df,
        save_path=fe_pipeline_save_path)

    logger.info("Load Feature Engineering Pipeline and apply transformations on train set")
    fe_pipeline_model = PipelineModel.load(fe_pipeline_save_path)
    twenty_train_counts_df = fe_pipeline_model.transform(twenty_train_df)

    logger.info("Training a classifier")
    model = Model(features_col="text_count_vect", label_col="label_indexed")
    model.fit_and_save_model(
        df_train=twenty_train_counts_df,
        save_path=classifier_save_path
    )

    logger.info("Model trained and saved")


if __name__ == "__main__":
    main()
