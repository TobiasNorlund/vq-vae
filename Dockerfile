FROM python:3.8

RUN groupadd -r docker-user && useradd -r -m -s /bin/false -g docker-user docker-user

# hadolint ignore=DL3027
RUN apt update && apt install -y less nano jq git

COPY bash.bashrc /etc/bash.bashrc

ARG DOCKER_WORKSPACE_PATH
RUN mkdir -p $DOCKER_WORKSPACE_PATH/src $DOCKER_WORKSPACE_PATH/.home
WORKDIR $DOCKER_WORKSPACE_PATH/src
ENV HOME=$DOCKER_WORKSPACE_PATH/.home

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt