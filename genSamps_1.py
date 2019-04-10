import os
import random
import numpy as np
from scipy import misc

HOW_MANY_OF_EACH = 5
HOW_MANY_FILES = 100000

prefShape = (99,99,3)

fcount = -1
for f in os.listdir('samps_0/'):
	try:
		if f == '.DS_Store':
			continue
		fcount+=1
		print(f)
		totalCount = 0
		for i in range(HOW_MANY_OF_EACH):
			myImage = misc.imread('samps_0/'+f)
			xStart = random.randint(0,myImage.shape[0] - 99)
			yStart = random.randint(0,myImage.shape[1] - 99)
			myslice = myImage[xStart:xStart+99,yStart:yStart+99]
			if(myslice.shape == prefShape):
				save = misc.imsave('samps_1/'+str(fcount)+"_"+str(totalCount)+'.jpg',myslice)
				totalCount+=1
			else:
				print("BOOOO!!!!")
				print(f)
	except:
		pass
	if fcount > HOW_MANY_FILES:
		exit()