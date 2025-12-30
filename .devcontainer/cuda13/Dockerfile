FROM nvidia/cuda:13.1.0-devel-ubuntu24.04
RUN apt -y update
RUN apt -y install curl git

RUN apt -y install python3.12

RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/usr/local python3.12 -

WORKDIR /workspace
COPY pyproject.toml README.md /workspace/
COPY nn_trainer /workspace/nn_trainer
RUN poetry install --with=dev --extras=cuda13
