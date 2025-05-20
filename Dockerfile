FROM cubertgmbh/cuvis_python:3.3.1-ubuntu22.04

WORKDIR /app
COPY cuvis_ai cuvis_ai/
COPY entrypoint.sh entrypoint.sh
COPY pyproject.toml pyproject.toml
COPY docs docs/
ENV CUVIS=/lib/cuvis