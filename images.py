import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
import scipy.misc


def threshold(img):
	mask =  np.logical_and((img[:,:,0] > 200), (img[:,:,2] < 40),  (img[:,:,1] <  40)) * 1
	return mask

def fig2data (fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

for i in range(1, 2):

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
	fig.canvas.draw()
	im = fig2data(fig)

	# to output to a file
	# fig.savefig("input_images/large" + str(i) + ".jpg")
	# scipy.misc.imsave("input_images/test" + str(i) + ".jpg", im)

	plt.close(fig)

	# to read in from a file
	# im = cv2.imread("input_images/test" + str(i) + ".jpg", 1) 
	# print(im.shape)

	#make everything smaller
	im = im[::18,::18,:]	 #THIS IS THE INPUT (27, 36, 3)
	# print(im.shape)
	# scipy.misc.imsave("input_images/input" + str(i) + ".jpg", im)


	im_mask = threshold(im)
	# print(im_mask.shape)

	output = np.stack(((im_mask, (1-im_mask))), axis = 2) #THIS IS THE RESULTING OUTPUT (27, 36, 2), hopefully they are stacking in the right order
	# print(output.shape)




