#!/bin/bash

# create a vvirtual enviorment
if [ ! -d "venv"]; then
    python -m venv venv
    echo "Virtual enviorment 'venv' created."
else
    echo "Virtual enviorment 'venv' already exists."

fi 

# activate the virtual enviorment
#  for Unix-based system( linex,macOS)

sorce venv/script/activate

# install required package 
if [ -f "requirements.txt"]; then
    pip install -r requirements.txt
    echo "Installed packages from requirements.txt."
else
    pip install Flask
    echo "requirement.txt not found. installed Flask by defoult."
fi

echo "VIrtual enviorment setup and package intallation complete"