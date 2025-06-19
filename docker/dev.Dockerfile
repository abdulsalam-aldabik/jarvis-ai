FROM python:3.12-slim

WORKDIR /workspace

COPY pyproject.toml ./
RUN pip install poetry && poetry install

CMD ["/bin/bash"]
