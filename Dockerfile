FROM continuumio/anaconda3:latest
WORKDIR /
COPY requirements.txt /titanic.yml
RUN /bin/bash \
/opt/conda/bin/conda env create -f /titanic.yml
ENTRYPOINT /bin/bash
