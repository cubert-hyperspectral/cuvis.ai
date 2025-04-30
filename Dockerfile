FROM ttl.sh/5a2c954f-7f29-4782-b829-4d7812e6736d

WORKDIR /app
COPY cuvis_ai cuvis_ai/
COPY entrypoint.sh entrypoint.sh
COPY pyproject.toml pyproject.toml
ENV CUVIS=/lib/cuvis