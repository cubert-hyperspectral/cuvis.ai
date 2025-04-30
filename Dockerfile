FROM ttl.sh/efcdae76-2ba9-401b-bfc5-18c79a89a860

WORKDIR /app
COPY cuvis_ai cuvis_ai/
COPY entrypoint.sh entrypoint.sh
COPY pyproject.toml pyproject.toml
ENV CUVIS=/lib/cuvis