FROM python:3.12-slim
RUN pip install --no-cache-dir pytest
COPY . /tests
WORKDIR /workspace
