FROM python:3.9-slim-buster

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# pip leaves the install caches populated which uses a
# significant amount of space. These optimizations save a fair
# amount of space in the image, which reduces start up time.
RUN pip --no-cache-dir install langchain==0.0.354 opensearch-py==2.2.0 sagemaker==2.148.0 beautifulsoup4==4.12.2

# Include python script for retrieving credentials
# from AWS SecretsManager and Sagemaker helper classes
ADD credentials.py /code/
ADD sm_helper.py /code/

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE