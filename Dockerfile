FROM python:3.8-slim-buster

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED 1

ARG UID=1000
ARG GID=1000

RUN groupadd -g "${GID}" python \
  && useradd --create-home --no-log-init -u "${UID}" -g "${GID}" python
WORKDIR /home/python

COPY --chown=python:python requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get -y install libglib2.0-0 -y

# USER 변경은 반드시 pip 패키지 설치 스크립트 이후에 작성되어야 함
USER python:python
ENV PATH="/home/${USER}/.local/bin:${PATH}"
COPY --chown=python:python . .

ARG FLASK_ENV

ENV FLASK_ENV=${FLASK_ENV}

EXPOSE 5000

CMD ["gunicorn", "--timeout", "600", "-b", "0.0.0.0:5000", "app:app"]
