FROM ttl.sh/91a06dbb-c918-4f4b-be3c-10843f73d3b0:2h

WORKDIR /app
COPY cuvis_ai cuvis_ai/
COPY entrypoint.sh entrypoint.sh
COPY pyproject.toml pyproject.toml
ENV CUVIS=/lib/cuvis