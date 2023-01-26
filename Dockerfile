FROM nvcr.io/nvidia/pytorch:22.03-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONHASHSEED=0
ENV HOME /cluster/home/esh
ENV PYTHONPATH="/workspace"

RUN apt-get update -y && apt-get install -y unrar

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
