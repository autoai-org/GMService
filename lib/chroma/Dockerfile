FROM python:3.10

#RUN apt-get update -qq
#RUN apt-get install python3.10 python3-pip -y --no-install-recommends && rm -rf /var/lib/apt/lists_/*

WORKDIR /chroma

RUN git clone https://github.com/chroma-core/chroma.git .
RUN pip install --no-cache-dir --upgrade -r requirements.txt


EXPOSE 8000

CMD ["/docker_entrypoint.sh"]