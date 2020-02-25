FROM continuumio/miniconda3:4.7.12

RUN mkdir /opt/simple-ml-project/
ADD pyspark_project /opt/simple-ml-project/pyspark_project/

WORKDIR /opt/simple-ml-project/
ENV PYTHONPATH /opt/simple-ml-project
RUN python setup.py install
