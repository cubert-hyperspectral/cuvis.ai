FROM ttl.sh/606ac5b5-0f8a-4e75-855b-9774ce7b801e

WORKDIR /app
COPY cuvis_ai cuvis_ai/
COPY entrypoint.sh entrypoint.sh
COPY pyproject.toml pyproject.toml
COPY docs docs/
ENV CUVIS=/lib/cuvis