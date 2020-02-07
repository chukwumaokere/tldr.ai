#!/usr/bin/env bash

apt-get-update
sudo apt upgrade python3
sudo apt-get install python3.7
python3 --version
python3.7 --version
cd /vagrant/
# wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh
exec bash
conda list
conda install -c pytorch -c fastai fastai
pip install -U scikit-learn
python3.7 testfastai.py