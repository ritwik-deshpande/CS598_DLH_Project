# Use an official Python runtime as a parent image
FROM ubuntu:latest

# Install updates and necessary packages
 
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.7 openjdk-8-jdk python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* 
    
RUN apt update
RUN apt install -y git-core

WORKDIR /cs598

# Copy the current directory contents into the container at /app
COPY . /cs598

RUN pip3 install git+https://github.com/LeeKamentsky/python-javabridge
RUN pip3 install -r requirements.txt
RUN python3.7 -m spacy download en-core-web-sm==3.5.0



# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Run Jupyter Notebook when the container launches
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser"]
