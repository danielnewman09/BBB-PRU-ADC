# Python Files

To sample AIN-1 at 5kHz for one second, run rb_file.py

```bash
python rb_file.py
```

This will create a raw buffer file "output.0"

To convert the output to a text file, run read_files.py (requires numpy and Python 3 because I'm lazy)

```bash
python3 read_files.py
```

This will create a text file "output.0.txt". This script successfully used the PRU to capture a signal at 100Hz!

![FFT](https://raw.githubusercontent.com/danielnewman09/BBB-PRU-ADC/master/Images/FFT_PRU.png)


![raw signal](https://raw.githubusercontent.com/danielnewman09/BBB-PRU-ADC/master/Images/adc_reading_full.png)
