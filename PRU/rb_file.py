#!/usr/bin/python
## \file
# \brief Example: fetch ADC samples in a ring buffer and save to file.
#
# This file contains an example on how to use the ring buffer mode of
# libpruio. A fixed step mask of AIN-0, AIN-1 and AIN-2 get configured
# for maximum speed, sampled in to the ring buffer and from there saved
# as raw data to some files. Find a functional description in section
# \ref sSecExaRbFile.
#
# Licence: GPLv3, Copyright 2017-\Year by \Mail
#
# Run by: `python rb_file.py`
#
# \since 0.6.0

from __future__ import print_function
from libpruio import *
import argparse
import time

## For file operations
libc = CDLL("libc.so.6")


parser = argparse.ArgumentParser()

parser.add_argument('-n', action='store', dest='samplingSize', type=int, default=1000)
parser.add_argument('-dt', action='store', dest='samplingInterval', type=int, default=100000)

results = parser.parse_args()

## The number of samples in the files (per step).
tSamp = results.samplingSize
## The sampling rate in ns (20000 -> 50 kHz).
tmr = results.samplingInterval
## The number of active steps (must match setStep calls and mask).
NoStep = 1
## The number of files to write.
NoFile = 1
## The output file names format.
NamFil = "output.%u"

## Create a ctypes pointer to the pruio structure
io = pruio_new(PRUIO_DEF_ACTIVE, 0, 0, 0)
try:
  ## The pointer dereferencing, using contents member
  IO = io.contents
  if IO.Errr: raise AssertionError("pruio_new failed (%s)" % IO.Errr)

  #if pruio_adc_setStep(io, 9, 0, 0, 0, 0): #               step 9, AIN-0
  #  raise AssertionError("step 9 configuration failed: (%s)" % IO.Errr)
  if pruio_adc_setStep(io,10, 1, 0, 0, 0): #               step 9, AIN-1
    raise AssertionError("step 10 configuration failed: (%s)" % IO.Errr)
  #if pruio_adc_setStep(io,11, 2, 0, 0, 0): #               step 9, AIN-2
  #  raise AssertionError("step 11 configuration failed: (%s)" % IO.Errr)

  ## The active steps (9 to 11)
  mask = 0b1 << 10
  ## The maximum total index
  tInd = tSamp * NoStep
  ## Index half ring buffer
  half = ((IO.ESize >> 2) // NoStep) * NoStep

  if half > tInd: half = tInd #               adapt size for small files
  ## The number of samples (per step)
  samp = (half << 1) // NoStep

  if pruio_config(io, samp, mask, tmr, 0): #  upload settings, prepare MM/RB mode
    raise AssertionError("config failed (%s)" % IO.Errr)

  if pruio_rb_start(io): #                                start sampling
    raise AssertionError("rb_start failed (%s)" % IO.Errr)

  ## A pointer to the start of the ring buffer
  p0 = IO.Adc.contents.Value
  ## Pointer to middle of the ring buffer
  p1 = cast(byref(p0.contents, (half << 1)), POINTER(c_ushort))
  for n in range(0, NoFile):
    ## The file name
    fName = NamFil % n
    print("Creating file %s" % fName)
    ## The output file descriptor
    oFile = libc.fopen(fName, "wb")
    ## Start index
    i = 0
    while i < tInd:
      i += half
      if i > tInd: #             fetch the rest(maybe no complete chunk)
        ## The bites left in ring buffer
        rest = tInd + half - i
        ## Last index in buffer
        iEnd = rest if p1 >= p0 else rest + half
        while IO.DRam[0] < iEnd: time.sleep(0.001)
        print("  writing samples %u-%u" % (tInd -rest, tInd-1))
        libc.fwrite(p0, sizeof(UInt16), rest, oFile) #        write data
        p0, p1 = p1, p0 #                           swap buffer pointers
        break
      if p1 > p0:
        while IO.DRam[0] < half: time.sleep(0.001) # wait for completion
      else:
        while IO.DRam[0] > half: time.sleep(0.001) # wait for completion
      print("  writing samples %u-%u" % (i-half, i-1))
      libc.fwrite(p0, sizeof(UInt16), half, oFile) #      write bin data
      p0, p1 = p1, p0 #                             swap buffer pointers
    libc.fclose(oFile)
    print("Finished file %s" % fName)
finally:
  pruio_destroy(io)
