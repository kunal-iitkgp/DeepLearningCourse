import nibabel as nib
import numpy as np
import scipy.io
import random
from tempfile import TemporaryFile

mat = scipy.io.loadmat('mask.mat')
mask = mat['Mask_universal']
c = 0
index = []
for i in xrange(0,13):
	for j in xrange(0,16):
		for k in xrange(0,12):
			if(mask[i,j,k]!=0):
				index.append((i,j,k))
				c+=1
print "number of non zero voxels calculated from the mask= %d" %c

x_data = np.zeros((380,137,569))
y_data = np.zeros((380))

labels = scipy.io.loadmat('FinalSubjectsList.mat')

for i in range(0, 380, 4):
	
	y_data[i] = y_data[i+1] = y_data[i+2] = y_data[i+3] = labels['Labels'][i/4]

	for j in xrange(0,4):
		sample = np.array_str(labels['IDs380'][i][0])
		sample = '../data/' + sample[3:-2] +'_0003_AO_'+ str(j+1) +'.nii.gz'
		img = nib.load(sample)
		# print sample
		img_data = img.get_data()
		#print "4D shape = %s" % str(img_data.shape)
		for time in xrange(137):
			for voxel in xrange(len(index)):
				x,y,z = index[voxel]
				x_data[i+j][time][voxel] = img_data[x][y][z][time]
	
y_data = y_data.reshape((380))

np.save("processed_voxels.npy", x_data)

np.save("processed_labels.npy", y_data)
