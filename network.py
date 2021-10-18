import tensorflow as tf

import tf_util as U


class Model:
	def __init__(self, loss = "weighted"):
		self.loss = loss
		self.prob = tf.placeholder(shape = (), dtype = tf.float32)
		self.lr = tf.placeholder(shape = (), dtype = tf.float32)
		self.n_patches = 64
		self.output = self.loss_op = self.train_op = None

	@staticmethod
	def block(net, filters):
		with tf.variable_scope('block_' + str(filters)):
			net = U.conv2d(net, filters, 'conv1', (3, 3))
			net = U.swish(net)
			net = U.conv2d(net, filters, 'conv2', (3, 3))
			net = U.swish(net)
			net = U.maxpool(net, 2)
		return net

	def build(self, image, mos_score):
		net = tf.reshape(image, [-1, 32, 32, 3])
		net = self.block(net, 32)
		net = self.block(net, 64)
		net = self.block(net, 128)
		net = self.block(net, 256)
		net = self.block(net, 512)

		net1 = tf.reshape(net, (-1, 512))
		net1 = U.dense(net1, 512, 'fc1')
		net1 = U.swish(net1)
		net1 = tf.nn.dropout(net1, keep_prob = self.prob)
		net1 = U.dense(net1, 1, 'fc2')

		net2 = tf.reshape(net, (-1, 512))
		net2 = U.dense(net2, 512, 'fc1_weight')
		net2 = U.swish(net2)
		net2 = tf.nn.dropout(net2, keep_prob = self.prob)
		net2 = U.dense(net2, 1, 'fc2_weight')
		net2 = tf.nn.relu(net2) + 1e-6

		self.loss_op = self.weighted_loss(net1, net2, mos_score)

		optimizer = tf.train.AdamOptimizer(self.lr)
		self.train_op = optimizer.minimize(self.loss_op)

	def weighted_loss(self, h, a, t):
		self.output = 0

		h = tf.reshape(h, (-1, self.n_patches))
		a = tf.reshape(a, (-1, self.n_patches))
		ha = tf.multiply(h, a)
		ha_sum = tf.reduce_sum(ha, axis = 1)
		a_sum = tf.reduce_sum(a, axis = 1)

		y = tf.divide(ha_sum, a_sum)
		diff = tf.abs(y - t)
		loss = tf.reduce_mean(diff)
		return loss
