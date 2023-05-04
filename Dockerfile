FROM continuumio/anaconda3:2022.10
WORKDIR /
COPY requirements.txt /titanic.yml
RUN /bin/bash \
/opt/conda/bin/conda env create -f /titanic.yml
ENTRYPOINT /bin/bash
