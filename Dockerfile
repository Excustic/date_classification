FROM continuumio/miniconda3

# # Set the working directory to /app
WORKDIR /app
#
# # Copy the current directory contents into the container at /app
ADD . /app

# Create the environment:
COPY environment.yml /app
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "date_classification_docker", "/bin/bash", "-c"]

# Make sure the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

# The code to run when container is started:
COPY app.py .
EXPOSE 5000
ENTRYPOINT ["conda", "run", "-n", "date_classification_docker", "python", "app.py"]