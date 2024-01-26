FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install pip
RUN apt-get update && \
    apt-get install -y python3-pip

# Install any needed packages specified in requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the current directory contents into the container at /app
COPY ./ /app

# Specify the command to run on container start
CMD ["uvicorn", "fastapp:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

