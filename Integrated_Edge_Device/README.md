# Integrated Edge Device

The purpose of this is to document the development of an edge device which can acquire, parse, publish, and analyze vibration data 

## Preparing the Beaglebone

Install Python 3.7 

```bash 
wget https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tar.xz
tar xf Python-3.7.3.tar.xz
cd ./Python-3.7.3 
./configure
make
make install
update-alternatives --install /usr/bin/python python /usr/local/bin/python3.7 10
```


## Install Tensorflow

```bash
python3 -m pip install cython
sudo apt-get update
sudo apt-get install libhdf5-dev
```
