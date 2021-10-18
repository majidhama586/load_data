import glob


def read_data(path = "./labelled/*.jpg", no_of_train = 275):
	x = []
	y = []
	import tensorflow as tf
	for image in glob.glob(path):
		x.append(image)
		y.append(float(image.split('_')[2].split('.jpg')[0]))

	assert (no_of_train < len(x))

	return x[0:no_of_train], y[0:no_of_train], x[no_of_train:], y[no_of_train:]
