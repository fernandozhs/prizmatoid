import numpy as np

slice_data = np.load('slices.npy', allow_pickle=True)
offset_data = np.load('offsetting.npy', allow_pickle=True)

data = np.array([ [slice, offset] for slice, offset in zip(slice_data, offset_data) ])

print(data)

np.save('offsetting2.npy', data)
