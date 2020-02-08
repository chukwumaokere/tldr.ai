#!/usr/bin/env bash
anaconda=Anaconda3-2019.10-Linux-x86_64.sh

apt-get update
sudo apt upgrade python3
sudo apt-get install -y python3.7
python3 --version
python3.7 --version
cd /vagrant/
if [[ ! -f $anaconda ]]; then
    wget --quiet https://repo.anaconda.com/archive/$anaconda
fi
cd /vagrant/
sudo chmod +x ./$anaconda
cd /vagrant/
mkdir ./anaconda3
sh ./$anaconda -b -p /home/vagrant/anaconda3/
eval "$(/home/vagrant/anaconda3/bin/conda shell.bash hook)"
conda list
conda install -c pytorch -c fastai fastai
pip install -U scikit-learn
python3.7 testfastai.py

