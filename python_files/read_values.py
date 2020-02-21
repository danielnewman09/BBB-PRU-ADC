import numpy as np

f = open('/home/debian/python-pru/examples/output.0','rb')

raw_data = f.read()

numpy_data = np.frombuffer(raw_data,dtype=np.uint16)
numpy_data = np.atleast_2d(numpy_data)
print(numpy_data.shape)

np.savetxt('/home/debian/python-pru/examples/output.0.txt',numpy_data,delimiter=',')
