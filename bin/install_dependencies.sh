#!/bin/bash

aws s3 cp s3://sequence-models/bootstrap/requirements.txt .
sudo python3 -m pip install python-dev .
sudo python3 -m pip install -r requirements.txt