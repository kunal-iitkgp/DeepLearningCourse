import nibabel as nib
import numpy as np
import scipy.io
import random

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

x_train = np.zeros((380,137,569))
y_train = np.zeros((95))

labels = scipy.io.loadmat('FinalSubjectsList.mat')
y_train = labels['Labels'].reshape((95))

for i in xrange(95):
	sample = np.array_str(labels['IDs380'][i][0])
	sample = '../data/' + sample[3:-2] +'_0003_AO_1.nii.gz'
	img = nib.load(sample)
	#print sample
	img_data = img.get_data()
	#print "4D shape = %s" % str(img_data.shape)
	for time in xrange(137):
		for voxel in xrange(len(index)):
			x,y,z = index[voxel]
			x_train[i][time][voxel] = img_data[x][y][z][time]
			
	# try:
	# 	img = nib.load(sample)
	# except Exception, e:
	# 	continue
	# img_data = img.get_data()
	# print "4D shape = %s" % str(img_data.shape)
# for i in xrange(88):
# 	sample = np.array_str(labels['IDs380'][i][0])
# 	sample = '../data/' + sample[3:-2] +'_0003_AO_2.nii.gz'
# 	#print sample
# 	try:
# 		img = nib.load(sample)
# 	except Exception, e:
# 		continue
# 	img_data = img.get_data()
# 	print "4D shape = %s" % str(img_data.shape)

# for i in xrange(88):
# 	sample = np.array_str(labels['IDs380'][i][0])
# 	sample = '../data/' + sample[3:-2] +'_0003_AO_3.nii.gz'
# 	#print sample
# 	try:
# 		img = nib.load(sample)
# 	except Exception, e:
# 		continue
# 	img_data = img.get_data()
# 	print "4D shape = %s" % str(img_data.shape)
# for i in xrange(88):
# 	sample = np.array_str(labels['IDs380'][i][0])
# 	sample = '../data/' + sample[3:-2] +'_0003_AO_4.nii.gz'
# 	#print sample
# 	try:
# 		img = nib.load(sample)
# 	except Exception, e:
# 		continue
# 	img_data = img.get_data()
# 	print "4D shape = %s" % str(img_data.shape)

# x = random.randint(0,136)
# print "x=%d" %x
# c = 0
# voxel = mask
# for i in xrange(0,13):
# 	for j in xrange(0,16):
# 		for k in xrange(0,12):
# 			if(voxel[i,j,k]!=0):
# 				if(img_data[i,j,k,x]!=0):
# 					c+=1
# print "number of non zero voxels calculated from the random image at some time step for this 4D sample= %d" %c
