# Set up the fast.ai Vagrant dev system for tldr.ai

1) Install [VirtualBox 6.0](https://www.virtualbox.org/wiki/Download_Old_Builds_6_0) (IT WILL NOT WORK WITH 6.1+)
2) Install Vagrant from [vagrantup.com](https://www.vagrantup.com/downloads.html)
3) Clone/download this repo
4) Run `vagrant up`
5) If anything fails, run `vagrant provision` to rerun the provisioning step or edit the `bootstrap.sh` file
6) When its done, you should see a really long read out (thats the shell running the python script for fastai which pulls 20 random articles to ensure everything was installed correctly) and it will end in `default: (11314, 2)` which is the last print out in the python script.
7) Run `vagrant ssh` to get into the vagrant environment. If any bootstrap lines failed to run, open `bootstrap.sh` and manually run the remaining lines. Or resolve the issues in the `bootstrap.sh` file and run `vagrant provision` from the host computer.
8) When you run `vagrant ssh` you MAY need to run this in the shell `eval "$(/home/vagrant/anaconda3/bin/conda shell.bash hook)"` if you get `No module named fastai` or `conda list` or any other conda functions don't seem to work. This will start your shell in the conda environment which is needed to use fastai functions.
9) So that its not annoying, you can add this line `eval "$(/home/vagrant/anaconda3/bin/conda shell.bash hook)"` to `/home/vagrant/.bashrc` so it automatically puts you in the conda environment when you ssh in
10) Feel free to also add `cd /vagrant/` after that line too, to start in `/vagrant/`
11) To make it easier to execute python scripts using python3.7, you can also add `alias python=python3.7` to your `.bashrc` file so you can use `python testfastai.py` to execute scripts faster.