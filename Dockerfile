FROM nvcr.io/nvidia/pytorch:21.11-py3  

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install correct data in container
RUN  apt-get update &&  \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
     apt-get install -y npm

# add requirements 
ADD requirements.txt /requirements.txt

# create virtual env for the project and install requirements
RUN python3 -m venv /venv && source /venv/bin/activate 
RUN pip install --upgrade pip && \
    pip install wheel && \
    pip install -r /requirements.txt

