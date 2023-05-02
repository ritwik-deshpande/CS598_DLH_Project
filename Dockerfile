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
RUN apt install wget
RUN apt install unzip

WORKDIR /app
COPY . .
RUN ls -lrt
RUN wget http://www.java2s.com/Code/JarDownload/weka/weka.jar.zip
RUN ls -lrt
RUN unzip weka.jar.zip
RUN cd CS598_DLH_Project/
RUN ls -lrt 
RUN pip install -r requirements.txt
# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Run Jupyter Notebook when the container launches
CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser"]
