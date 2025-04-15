FROM python:3.13-slim-bookworm

ENV PIP_NO_CACHE_DIR=1

WORKDIR  /src

COPY . /src

RUN python3 -m venv .venv
ENV PATH="/src/.venv/bin:$PATH"
RUN pip3 install .

ENTRYPOINT [ "python3", "main.py" ]