FROM cubertgmbh/cuvis_python:3.4.0-ubuntu24.04

WORKDIR /app
COPY cuvis_ai cuvis_ai/
COPY entrypoint.sh entrypoint.sh
COPY build_docs.sh build_docs.sh
COPY pyproject.toml pyproject.toml
ENV CUVIS=/lib/cuvis