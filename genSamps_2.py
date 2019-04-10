import os
import numpy as np
from scipy import misc


HOW_MANY_TO_USE = 100000

count = 0

for f in os.listdir('samps_1/'):
	if f == '.DS_Store':
		continue
	print(f)
	myImage = misc.imread('samps_1/'+f)
	mySliceY = myImage[33:66,33:66]

	a = myImage[0:33,33:66]
	b = myImage[66:99,33:66]
	c = myImage[33:66,0:33]
	d = myImage[33:66,66:99]

	mySliceX = np.concatenate((a,b,c,d), axis=1)
	count+=1
	if count > HOW_MANY_TO_USE:
		exit()
	misc.imsave('samps_2/y'+f,mySliceY)
	misc.imsave('samps_2/x'+f,mySliceX)

