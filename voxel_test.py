import nibabel as nib
import numpy as np
import scipy.io
import random

sample_l = nib.load('../data/000303269784_0003_AO_1.nii.gz')
sample = sample_l.get_data()
print "4D shape = %s" % str(sample.shape)
#print type(img_data)

mat = scipy.io.loadmat('mask.mat')
#print type(mat['Mask_universal'])
mask = mat['Mask_universal']
#print mask.shape
#print mask

c = 0
voxel = mask
print mask.shape
for i in xrange(0,13):
	for j in xrange(0,16):
		for k in xrange(0,12):
			voxel[i,j,k] = 0
			if(mask[i,j,k]!=0):
				c= c + 1
				print"ks"
print "number of non zero voxels calculated from the mask= %d" %c

x = random.randint(0,136)
print "x=%d" %x
c = 0
for i in xrange(0,13):
	for j in xrange(0,16):
		for k in xrange(0,12):
			if(voxel[i,j,k]==1):
				if(sample[i,j,k,x]!=0):
					c+=1
print "number of non zero voxels calculated from the random image at some time step for this 4D sample= %d" %c
