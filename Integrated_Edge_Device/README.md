# Integrated Edge Device

The purpose of this is to document the development of an edge device which can acquire, parse, publish, and analyze vibration data 

## Preparing the Beaglebone


### Install Tensorflow (Requires Python 3.5)

The wheel file in this directory has been generated with the help of [https://github.com/lhelontra/tensorflow-on-arm](this repository). If you wish to use a different version of Tensorflow or Python, you will need to build Tensorflow on your own.

Before installing tensorflow, you need to install a couple of libraries.

```bash
python3 -m pip install cython
sudo apt-get update
sudo apt-get install libhdf5-dev
```

After you install these, you should be able to install Tensorflow without any errors. 

```bash
python3 -m pip install tensorflow-2.0.0-cp35-none-linux_armv7l.whl
```

At a couple of points, the installation hangs for a _very_ long time. Specifically, when either of these commands are showing:

```bash
Running setup.py bdist_wheel for numpy ...
Running setup.py bdist_wheel for h5py ...
```

I was able to cancel these specific actions by CTRL+C. The terminal will show an error as below, but installation can continue successfully.

```bash
  Running setup.py bdist_wheel for numpy ... \^error
  Failed building wheel for numpy
  Running setup.py clean for numpy
  Complete output from command /usr/bin/python3 -u -c "import setuptools, tokenize;__file__='/tmp/pip-build-8cbq0qnv/numpy/setup.py';f=getattr(tokenize, 'open', open)(__file__);code=f.read().replace('\r\n', '\n');f.close();exec(compile(code, __file__, 'exec'))" clean --all:
  Running from numpy source directory.
  
  `setup.py clean` is not supported, use one of the following instead:
  
    - `git clean -xdf` (cleans all files)
    - `git clean -Xdf` (cleans all versioned files, doesn't touch
                        files that aren't checked into the git repo)
  
  Add `--force` to your command to use it anyway if you must (unsupported).
  
  
  ----------------------------------------
  Failed cleaning build dir for numpy
  Running setup.py bdist_wheel for h5py ... \^error
  Failed building wheel for h5py
  Running setup.py clean for h5py
```

After you are done, you should be able to use tensorflow

```bash
debian@beaglebone:~$ python3
Python 3.5.3 (default, Sep 27 2018, 17:25:39) 
[GCC 6.3.0 20170516] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> 
```

## Install Scipy

Installing scipy will take a long time (several hours)

```bash
sudo apt-get update
sudo apt-get install libatlas-base-dev
python3 -m pip install scipy
```

## Install Scikit-Learn

It's also difficult to install scikit-learn on a beaglebone device. To properly install the most recent releases, it is necessary to first install the dependencies (numpy, scipy, etc.) and then install the library without trying to install the dependencies. 

Installing Scikit-Learn will also take several hours.


```bash
python3 -m pip install joblib
python3 -m pip install -U scikit-learn --no-deps
```


## Python API Setup

This repository also contains code to extract statistical features from a vibration payload. It leverages a Flask REST API to efficiently do so. 

The python code requires libraries such as numpy and scipy. Start by installing them

```bash
sudo apt-get update
python3 -m pip install wheel uwsgi flask
```

Create a service file:

```bash
sudo nano /etc/systemd/system/python_api.service
```

and paste this code into it

```bash
[Unit]
Description=uWSGI instance to serve api
After=network.target

[Service]
User=debian
Group=debian
WorkingDirectory=/<path_to_pthon_files>/
Environment="PATH=/usr/bin/python3"
ExecStart=/home/debian/.local/bin/uwsgi --ini api.ini


[Install]
WantedBy=multi-user.target

```

Be sure to change the WorkingDirectory line to point to the directory which houses ```BBB-PRU_ADC/BBB``` from this repository.

Start this service

```bash
sudo systemctl start python_api
sudo systemctl enable python_api
```

It will take several seconds to fully load. You can verify it is running by ```systemctl status python_api```:

```bash
root@beaglebone:/home/debian/# systemctl status edge_ml_api
● python_api.service - uWSGI instance to serve api
   Loaded: loaded (/etc/systemd/system/python_api.service; enabled; vendor pres
   Active: active (running) since Thu 2020-04-02 15:49:34 UTC; 3 days ago
 Main PID: 821 (uwsgi)
    Tasks: 7 (limit: 4915)
   CGroup: /system.slice/edge_ml_api.service
           ├─ 821 /home/debian/.local/bin/uwsgi --ini api.ini
           ├─1418 /home/debian/.local/bin/uwsgi --ini api.ini
           ├─1419 /home/debian/.local/bin/uwsgi --ini api.ini
           ├─1420 /home/debian/.local/bin/uwsgi --ini api.ini
           ├─1421 /home/debian/.local/bin/uwsgi --ini api.ini
           ├─1422 /home/debian/.local/bin/uwsgi --ini api.ini
           └─1423 /home/debian/.local/bin/uwsgi --ini api.ini

Apr 06 12:08:15 beaglebone uwsgi[821]: [pid: 1421|app: 0|req: 38543/80926] 127.0
Apr 06 12:08:19 beaglebone uwsgi[821]: [pid: 1422|app: 0|req: 39798/80927] 127.0
Apr 06 12:08:23 beaglebone uwsgi[821]: [pid: 1421|app: 0|req: 38544/80928] 127.0
Apr 06 12:08:27 beaglebone uwsgi[821]: [pid: 1422|app: 0|req: 39799/80929] 127.0
```
