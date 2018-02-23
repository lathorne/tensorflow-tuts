import numpy as np
import random
import matplotlib.pyplot as plt

##### need to edit next_function and make sure the data is being loaded in correctly

DATASET_SIZE = 5000

def threshold(img):
	mask =  np.logical_and((img[:,:,0] > 200), (img[:,:,2] < 40),  (img[:,:,1] <  40)) * 1
	return mask

def fig2data (fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def load_data(ratio = 0.8):

	for i in range(1, DATASET_SIZE):

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
		plt.close(fig)

		
		im = im[::18,::18,:]	 #THIS IS THE INPUT (27, 36, 3)
		im_mask = threshold(im)
		output = np.stack(((im_mask, (1-im_mask))), axis = 2) #THIS IS THE RESULTING OUTPUT (27, 36, 2), hopefully they are stacking in the right order
		
		# not sure if I am doing this right, I think I have to create a float here
		_x.append(im) #input array
		_y.append(output) #output array

	# Split data into test/training sets
	index = int(ratio * len(_x)) # Split index
	x_train = _x[0:index, :]
	y_train = _y[0:index]
	x_test = _x[index:,:]
	y_test = _y[index:]	

	# Print out data sizes for train/test batches
	print("Data Split: ", ratio)
	print("Train => x:", x_train.shape, " y:", len(y_train))
	print("Test  => x:", x_test.shape, " y:", len(y_test))

	return [x_train, y_train, x_test, y_test]


class DataLoader():
	""" data loader for custom dataset """
	def __init__(self):
		self.x_train, self.y_train, self.x_test, self.y_test = load_data()

	def next_batch(self, batch_size):
		length = self.x_train.shape[0]
		indices = np.random.randint(0, length, batch_size) # Grab batch_size values randomly
		return [self.x_train[indices], self.y_train[indices]]