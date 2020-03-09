import os
from subprocess import call
import argparse
import json
from numpy import frombuffer
from numpy import uint16

cwd = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':

    # Create an argParser to parse through the user-given arugments
    argParser = argparse.ArgumentParser()

    # Add an arugment to parse through the raw vibration data.
    argParser.add_argument(
        '-n', '--samplePoints', type=str,
        help='comma-separated string of raw accelerometer data'
        )

    # Add an arugment to parse the spindle rpm
    argParser.add_argument(
        '-dt', '--samplingInterval', type=float,
        help='',
        )

    # pack the args into a nice list
    args = vars(argParser.parse_args())
    samplingInterval = args['samplingInterval']
    samplePoints = args['samplePoints']

    samplingIntervalNS = int(samplingInterval * 1e9)

    call([cwd + '/rb_file',str(samplePoints),str(samplingIntervalNS)])

    raw_file = open(cwd + '/output.0','rb')
    raw_data = raw_file.read()

    numpy_data = frombuffer(raw_data,dtype=uint16)

    print({'Vibration':numpy_data.tolist()})

