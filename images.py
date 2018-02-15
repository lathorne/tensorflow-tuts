import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
import scipy.misc


def threshold(img):
	mask =  np.logical_and((img[:,:,2] > 200), (img[:,:,1] < 40),  (img[:,:,0] <  40)) * 1
	return mask

for i in range(1, 20):

	fig, ax = plt.subplots()

	#plotting red circles
	for x in range(random.randint(1,10)):
		circle1 = plt.Circle((random.uniform(0,1), random.uniform(0,1)), random.uniform(0,0.2), color = 'red', clip_on = False)
		ax.add_artist(circle1)

	#plotting blue circles
	for x in range(random.randint(1,10)):
		circle2 = plt.Circle((random.uniform(0,1), random.uniform(0,1)), random.uniform(0,0.2), color = "blue", clip_on = False)
		ax.add_artist(circle2)

	#plotting green circles
	for x in range(random.randint(1,10)):
		circle3 = plt.Circle((random.uniform(0,1), random.uniform(0,1)), random.uniform(0,0.2), color = "green", clip_on = False)
		ax.add_artist(circle3)


	plt.axis('off')

	#plt.show()
	fig.savefig("input_images/input" + str(i) + ".jpg")
	plt.close(fig)

	#create binary mask
	im = cv2.imread("input_images/input" + str(i) + ".jpg", 1) 

	im_mask = threshold(im)
	#print(im_mask.shape)

	a = np.stack(((mask, (1-mask))), asix = 2) #create a vector out of the two masks

	#look at shape here WxHx2
	#this isn't set to a probability so how will this work??
	#output_im = np.concatenate(im_mask, 1 - im_mask) # only length-1 arrays can be converted to python scalars


	scipy.misc.imsave("output_images/output" + str(i) + ".jpg", im_mask)

	#FIGURE OUT THE SECOND DIMENSION AND PROPERLY GENERATE ABOUT 1000 images



