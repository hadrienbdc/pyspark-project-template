# PySpark project template

Boilerplate template for machine learning projects in PySpark.

To build docker image :
```bash
IMAGE_NAME=simple-ml-project
IMAGE_VERSION=1.0.0
docker build -t ${IMAGE_NAME}:${IMAGE_VERSION} -f Dockerfile .
```

To run model fit :
```bash
IMAGE_NAME=simple-ml-project
IMAGE_VERSION=1.0.0
PYTHON_SCRIPT=pyspark_project/fit_pipeline.py
docker run -d --name IMAGE_NAME ${IMAGE_NAME}:${IMAGE_VERSION} python ${PYTHON_SCRIPT} ${PARAMS}
```
This script will take parameters :
```
usage: fit_pipeline.py [-h] [--fe-path FE_PIPELINE_SAVE_PATH]
                        [--classifier-path CLASSIFIER_SAVE_PATH]
 
Simple project example
 
optional arguments:
  -h, --help            show this help message and exit
  --fe-path FE_PIPELINE_SAVE_PATH
                        Where to save the feature engineering pipeline
  --classifier-path CLASSIFIER_SAVE_PATH
                        Where to save the resulting classifier
```
 
To run prediction
```bash
IMAGE_NAME=simple-ml-project
IMAGE_VERSION=1.0.0
PYTHON_SCRIPT=pyspark_project/predict.py
docker run -d --name IMAGE_NAME ${IMAGE_NAME}:${IMAGE_VERSION} python ${PYTHON_SCRIPT} ${PARAMS}
```
This script will take parameters :
```
usage: predict.py [-h] [--fe-path FE_PIPELINE_SAVE_PATH]
                       [--classifier-path CLASSIFIER_SAVE_PATH]
  
Simple project example
  
optional arguments:
  -h, --help            show this help message and exit
  --fe-path FE_PIPELINE_SAVE_PATH
                        Where to save the feature engineering pipeline
  --classifier-path CLASSIFIER_SAVE_PATH
                        Where to save the resulting classifier
```

To run tests
```
docker run -d --name IMAGE_NAME ${IMAGE_NAME}:${IMAGE_VERSION} python setup.py test
```