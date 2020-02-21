# BBB-PRU-ADC
My attempt at setting up and documenting the use of the Beaglebone Black PRU for reading an ADC

__Using this Image:__
https://debian.beagleboard.org/images/bone-debian-9.5-iot-armhf-2018-10-07-4gb.img.xz

Flash the image to an SD card and install on the Beaglebone. Log in to the device

Then, follow the prompts at

[http://users.freebasic-portal.de/tjf/Projekte/libpruio/doc/html/ChaPreparation.html]

to install libpruio on the beaglebone. These steps are repeated below:

```bash
sudo nano /etc/apt/sources.list
```

Add these lines to the file:

```bash
deb http://beagle.tuks.nl/debian jessie/
deb-src http://beagle.tuks.nl/debian jessie/
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

Now, you can begin running example code in python.