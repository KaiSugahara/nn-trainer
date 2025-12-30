FROM python:3.12-slim

RUN apt -y update
RUN apt -y install curl git
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/usr/local python3 -

WORKDIR /workspace
COPY pyproject.toml README.md /workspace/
COPY nn_trainer /workspace/nn_trainer
RUN poetry install --with=dev
