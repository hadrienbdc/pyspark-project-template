from pyspark.ml.feature import CountVectorizer, Tokenizer, StringIndexer
from pyspark.ml import Pipeline


class FeatureEngineering:
    def __init__(self):
        self.tokenizer = Tokenizer(inputCol="text", outputCol="text_token")
        self.count_vect = CountVectorizer(inputCol="text_token", outputCol="text_count_vect")
        self.label_indexer = StringIndexer(inputCol="label", outputCol="label_indexed")
        self.pipeline = Pipeline(stages=[self.tokenizer, self.count_vect, self.label_indexer])

    @staticmethod
    def fit_pipeline(pipeline, df_train):
        return pipeline.fit(df_train)

    @staticmethod
    def save_pipeline_model(pipeline_model, save_path):
        pipeline_model.write().overwrite().save(save_path)

    def fit_and_save_pipeline(self, df_train, save_path):
        pipeline_model = self.fit_pipeline(self.pipeline, df_train)
        self.save_pipeline_model(pipeline_model, save_path)
