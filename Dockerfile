#FROM steveltn/https-portal:1
FROM python:3.7.4





# https://github.com/joyzoursky/docker-python-chromedriver/blob/master/py3/py3.6-xvfb-selenium/Dockerfile
RUN apt-get update 
# install selenium
RUN apt-get install -y python3-software-properties
RUN apt-get install -y software-properties-common
RUN apt-get -y install apt-transport-https ca-certificates
RUN apt-get -y install apt-transport-https curl
RUN apt-get -y install wget curl
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        bzip2 \
        libfontconfig \
    && apt-get clean

RUN apt-get install --fix-missing

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN apt-get update

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8   
# Copy local code to the container image.
# --------------- Install python packages using `pip` ---------------
# Installing requirements this way allows you to leverage cache ADD is cache busting.

RUN bash -c 'echo -e "\
dask==2.5.2\n\
efel==3.0.72\n\
lxml==4.4.1\n\
elephant==0.4.1\n\
scipy==1.3.1\n\
tqdm==4.48.2\n\
neo==0.5.2\n\
plotly==4.5.2\n\
allensdk==0.16.3\n\
simplejson==3.17.0\n\
future==0.17.1\n\
streamlit==0.52.2\n\
asciiplotlib==0.2.3\n\
pandas==1.1.0\n\
matplotlib==3.1.1\n\
seaborn==0.9.0\n\
quantities==0.12.4\n\
Jinja2==2.10.3\n\
numpy==1.17.2\n\
Pebble==4.5.3\n\
backports.statistics==0.1.0\n\
python_dateutil==2.8.1\n\
scikit_learn==0.23.2\n\
quantities==0.12.4\n\
asciiplotlib==0.2.3\n\
requests==2.22.0\n\
multiprocess==0.70.10\n\
numpy==1.17.2\n\
scipy==1.3.1\n\
dask==2.5.2\n\
natsort==7.0.1\n\
elephant==0.4.1\n\
tables==3.5.2\n\
deap==1.3.0\n\
joblib==0.13.2\n\
allensdk==0.16.3\n\
seaborn==0.9.0\n\
frozendict==1.2\n\
neo==0.5.2\n\
lxml==4.4.1\n\
numba==0.45.1\n\
maps==5.1.1\n\
plotly==4.5.2\n\
tqdm==4.48.2\n\
lmfit==1.0.0\n\
Cython==0.29.13\n\
backports.statistics==0.1.0\n\
beautifulsoup4==4.9.1\n\
bmtk==0.0.7\n\
scikit_learn==0.23.2\n\
git+https://github.com/russelljjarvis/neuronunit\n\
git+https://github.com/scidash/sciunit\n\
" > requirements.txt'

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt \
	&& rm -rf requirements.txt
#RUN pip install --upgrade streamlit

ENV APP_HOME /app
WORKDIR $APP_HOME

WORKDIR $APP_HOME/data
# This may be more correct app doesn't mind
# WORKDIR $APP_HOME																	

RUN pip install --upgrade streamlit
RUN pip install git+https://github.com/fun-zoological-computing/BluePyOpt
RUN pip install dask[bag] --upgrade 
RUN pip install maps frozendict
ADD . .
WORKDIR examples/app

# --------------- Configure Streamlit ---------------
RUN mkdir -p .streamlit
RUN touch .streamlit/config.toml

#RUN wget https://raw.githubusercontent.com/MarcSkovMadsen/awesome-streamlit/master/.streamlit/config.prod.toml >> /root/.streamlit/config.toml
#RUN wget https://raw.githubusercontent.com/MarcSkovMadsen/awesome-streamlit/master/.streamlit/config.local.toml >> /root/.streamlit/config.toml
RUN touch .streamlit/credentials.toml
RUN echo "[general]" >> .streamlit/credentials.toml
RUN echo 'email = "colouredstatic@gmail.com"' >> .streamlit/credentials.toml

#RUN bash -c 'echo -e "\
#	[server]\n\
#	enableCORS = false\n\
#	enableXsrfProtection = false\n\
#	\n\
#	[browser]\n\
#	serverAddress = \"0.0.0.0\"\
#	" > .streamlit/config.toml'
RUN bash -c 'echo -e "\
	[server]\n\
	enableCORS = false\n\
	enableXsrfProtection = false\n\
	[browser]\n\
    " > .streamlit/config.toml'

#File "/usr/local/lib/python3.7/site-packages/neuronunit/models/backends/__init__.py", line 69, in <module>

#RUN cp /usr/local/lib/python3.7/site-packages/neuronunit/models/__init__.py foo.py.tmp
#RUN sed '$ d' foo.py.tmp > /usr/local/lib/python3.7/site-packages/neuronunit/models/__init__.py
#RUN rm -f foo.py.tmp
RUN head -n -1 /usr/local/lib/python3.7/site-packages/neuronunit/models/__init__.py > temp.txt ; 
RUN mv temp.txt /usr/local/lib/python3.7/site-packages/neuronunit/models/__init__.py
RUN mkdir /usr/local/lib/python3.7/site-packages/neuronunit/models/backends/config
ADD params.json .
ADD data_transport_container.py .
ADD optimization_management.py .

RUN mv params.json /usr/local/lib/python3.7/site-packages/neuronunit/models/backends/config
RUN mv data_transport_container.py /usr/local/lib/python3.7/site-packages/neuronunit/optimisation
RUN mv optimization_management.py /usr/local/lib/python3.7/site-packages/neuronunit/optimisation

EXPOSE 8501


# --------------- Export envirennement variable ---------------
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
#RUN streamlit version
# enviroment variable ensures that the python output is set straight
# to the terminal without buffering it first
RUN pip install llvmlite==0.32.1
RUN pip install numba==0.49.1

ENV PYTHONUNBUFFERED 1
CMD ["streamlit", "run", "--server.port", "8501", "app.py"]
