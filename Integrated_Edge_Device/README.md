# Integrated Edge Device

The purpose of this is to document the development of an edge device which can acquire, parse, publish, and analyze vibration data 

## Preparing the Beaglebone


### Install Tensorflow (Requires Python 3.5 right now)

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

