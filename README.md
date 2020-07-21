# BBB-PRU-ADC
My attempt at setting up and documenting the use of the Beaglebone Black PRU for reading an ADC

I recommend cloning this repository to your working device:

```bash
debian@beaglebone:~$ cd ~/
debian@beaglebone:~$ mkdir Git && cd Git
debian@beaglebone:~/Git$ git clone https://github.com/danielnewman09/BBB-PRU-ADC.git
```

__This tutorial has been tested on the following images:__

https://debian.beagleboard.org/images/bone-debian-9.5-iot-armhf-2018-10-07-4gb.img.xz

https://debian.beagleboard.org/images/bone-debian-9.9-lxqt-armhf-2019-08-03-4gb.img.xz

https://debian.beagleboard.org/images/bone-debian-10.3-iot-armhf-2020-04-06-4gb.img.xz

__NOTE:__ I believe that any newer debian image will work for the Beaglebone Black. I have tried using the Beaglebone AI but cannot find the PRUSS options on the current images for that device.

__NOTE:__ The 10.3 image includes numpy and scipy in the Python3 installation. This is a fantastic addition to the Beaglebone image, and I recommend using it.

Flash the image to an SD card and install on the Beaglebone. Log in to the device

Then, follow the prompts at

[http://users.freebasic-portal.de/tjf/Projekte/libpruio/doc/html/ChaPreparation.html]

to install libpruio on the beaglebone. These steps are repeated below:

```bash
sudo nano /etc/apt/sources.list
```

Add these lines to the file:

```bash
deb [trusted=yes] http://beagle.tuks.nl/debian jessie/
deb-src [trusted=yes] http://beagle.tuks.nl/debian jessie/
```

Then:

```bash
wget -qO - http://beagle.tuks.nl/debian/pubring.gpg | sudo apt-key add -
sudo apt-get update
```

To install the python libraries:

```bash
sudo apt-get install python-pruio libpruio-lkm libpruio-doc
```

Copy the python example files to a desired directory by:

```bash
cp -r /usr/share/doc/python-pruio/examples .
```

When you try to run a python file, you should get an error like:

```bash
initialisation failed (cannot open /dev/uio5)
```

To mitigate this, you need to modify the boot file:
[https://www.freebasic.net/forum/viewtopic.php?t=22501&start=240]

```bash
sudo nano /boot/uEnv.txt
```

There will be a section that looks like the one below. Uncomment the line below the one that begins with "pruio_uio". Your file should look identical to the following code block when done.

```bash
###PRUSS OPTIONS
###pru_rproc (4.4.x-ti kernel)
#uboot_overlay_pru=/lib/firmware/AM335X-PRU-RPROC-4-4-TI-00A0.dtbo
###pru_rproc (4.14.x-ti kernel)
#uboot_overlay_pru=/lib/firmware/AM335X-PRU-RPROC-4-14-TI-00A0.dtbo
###pru_uio (4.4.x-ti, 4.14.x-ti & mainline/bone kernel)
uboot_overlay_pru=/lib/firmware/AM335X-PRU-UIO-00A0.dtbo
###
###Cape Universal Enable
#enable_uboot_cape_universal=1
###
```

Then reboot.

Now, you can begin running example code in python.

## Setting up Node Red

In order to get the files to run properly in Node Red, you must run them as the 'debian' user. I've found that changing some folder permissions and defaults for Node Red works for this. 

First, find the service file used to run Node Red. This seems to change from one distribution to the next, so a good way to know for sure is by checking the service:

```bash
debian@beaglebone:~/Git/BBB-PRU-ADC/PRU$ systemctl status nodered
● nodered.service - Node-RED graphical event wiring tool
   Loaded: loaded (/lib/systemd/system/nodered.service; disabled; vendor preset: enabled)
   Active: active (running) since Tue 2020-07-21 13:05:36 UTC; 11min ago
     Docs: https://nodered.org/docs/
 Main PID: 16084 (node-red)
    Tasks: 11 (limit: 1028)
   Memory: 104.7M
   CGroup: /system.slice/nodered.service
           └─16084 node-red
```

NOTE: the name of the service may be "nodered" or "node-red." I don't see a rhyme or reason to why it's sometimes one or the other

The service file from the above command is: ```/lib/systemd/system/nodered.service```. When we look at that file, we see where the Node Red installation is located on the machine:

```bash
debian@beaglebone:~/Git/BBB-PRU-ADC/PRU$ cat /lib/systemd/system/nodered.service
# systemd service file to start Node-RED
# From: https://github.com/node-red/linux-installers/blob/master/resources/nodered.service

[Unit]
Description=Node-RED graphical event wiring tool
Wants=network.target
Documentation=https://nodered.org/docs/
After=multi-user.target

[Service]
Type=simple
# Run as normal pi user - change to the user name you wish to run Node-RED as
User=node-red
Group=node-red
WorkingDirectory=/var/lib/node-red
```

With this information, we want to change the User and Group lines to 'debian':

```bash
[Service]
Type=simple
# Run as normal pi user - change to the user name you wish to run Node-RED as
User=debian
Group=debian
WorkingDirectory=/var/lib/node-red
```
Also note the WorkingDirectory. We want to make sure the debian user has ownership of this directory.

```bash
debian@beaglebone:~$ sudo chown -R debian /var/lib/node-red
debian@beaglebone:~$ sudo chgrp -R debian /var/lib/node-red
```

After you do this, your Node Red working directory should look like so:

```bash
debian@beaglebone:~$ cd /var/lib/node-red
debian@beaglebone:/var/lib/node-red$ ls -al
total 16
drwxr-xr-x  4 debian debian 4096 Jul 21 13:31 .
drwxr-xr-x 30 root   root   4096 Jul 21 01:19 ..
drwx------  3 debian debian 4096 Apr  6 13:28 .config
drwxr-xr-x  5 debian debian 4096 Jul 21 13:06 .node-red
```
restart Node Red

```bash
debian@beaglebone:~$ sudo systemctl restart nodered
```

## Install Scikit-Learn

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
WorkingDirectory=/<path_to_python_files>/
Environment="PATH=/usr/bin/python3"
ExecStart=/home/debian/.local/bin/uwsgi --ini api.ini


[Install]
WantedBy=multi-user.target

```

Be sure to change the WorkingDirectory line to point to the directory which houses ```BBB-PRU_ADC/BBB/Python_API``` from this repository.

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

