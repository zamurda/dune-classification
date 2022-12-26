# image name uproot-extract
# made to run the uproot_to_csv.py script for data mining select feautures from .root files


FROM python:3.10-slim-buster

WORKDIR /dune_classification

# copy requirements and install
COPY requirements.txt requirements.txt
RUN pip3 install -v -r requirements.txt

# copy the source code into the image
COPY .  .

RUN pip install -e .

# change workdir to ./scripts for executing the required script
WORKDIR /dune_classification/scripts
CMD ["python", "-m", "uproot_to_csv"]