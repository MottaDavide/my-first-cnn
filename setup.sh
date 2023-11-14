#!/bin/bash

# Check if Poetry is installed
if ! command -v poetry &> /dev/null
then
    echo "Poetry not found, installing..."
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
else
    echo "Poetry is already installed."
fi

# Initialize the environment
poetry install

#Run
#chmod +x setup.sh
