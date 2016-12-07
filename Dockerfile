FROM python:2.7-slim

# Update the sources list
RUN apt-get update && apt-get install -y apt-transport-https

# Install basic applications
RUN apt-get install -y tar git curl nano wget dialog net-tools build-essential

# Install postgresql-devel for psycopg2
RUN apt-get install -y libpq-dev

# Install Tkinter for matplotlib
RUN apt-get install -y python-tk

ADD . /code

WORKDIR /code

# Fix PYTHONPATH
ENV PYTHONPATH="$PYTHONPATH:/code"

ENV DJANGO_SECRET_KEY="thisisthesecretkey"

RUN pip install -r requirements.txt

# Download stopwords
RUN python -m nltk.downloader stopwords

RUN python server/manage.py migrate

CMD ["python", "server/start_server.py"]
