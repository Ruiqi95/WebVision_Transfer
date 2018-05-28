import tensorflow as tf
import envvar as envVar

def print_activations(t):
	print(t.op.name, ' ', t.get_shape().as_list())

def inference(images, dropout_rate):
	with tf.name_scope("conv_1") as scope:
		if not (scope in envVar.FIX_LAYER):
			kernels = tf.Variable(tf.truncated_normal([11, 11, 3, 96], stddev=0.01))
			biases  = tf.Variable(tf.constant(0.1, shape=[96]))
			weight_decay = tf.multiply(tf.nn.l2_loss(kernels), 0.0005, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		else:
			kernels = tf.Variable(tf.truncated_normal([11, 11, 3, 96], stddev=0.01), trainable=False)
			biases  = tf.Variable(tf.constant(0.1, shape=[96]), trainable=False)
		conv   = tf.nn.conv2d(images, kernels, strides=[1,4,4,1], padding='SAME')
		conv_1 = tf.nn.relu(conv + biases)
		tf.summary.histogram(scope + "weights", kernels)
		tf.summary.histogram(scope + "bias", biases)
		print_activations(conv_1)

	with tf.name_scope("max_2") as scope:
		max_2 = tf.nn.max_pool(conv_1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
		print_activations(max_2)

	with tf.name_scope("conv_3") as scope:
		if not (scope in envVar.FIX_LAYER):
			kernels = tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.01))
			biases  = tf.Variable(tf.constant(0.1, shape=[256]))
			weight_decay = tf.multiply(tf.nn.l2_loss(kernels), 0.0005, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		else:
			kernels = tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.01), trainable=False)
			biases  = tf.Variable(tf.constant(0.1, shape=[256]), trainable=False)
		conv   = tf.nn.conv2d(max_2, kernels, strides=[1,1,1,1], padding='SAME')
		conv_3 = tf.nn.relu(conv + biases)
		tf.summary.histogram(scope + "weights", kernels)
		tf.summary.histogram(scope + "bias", biases)
		print_activations(conv_3)

	with tf.name_scope("max_4") as scope:
		max_4 = tf.nn.max_pool(conv_3, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
		print_activations(max_4)

	with tf.name_scope("conv_5") as scope:
		if not (scope in envVar.FIX_LAYER):
			kernels = tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01))
			biases  = tf.Variable(tf.constant(0.1, shape=[384]))
			weight_decay = tf.multiply(tf.nn.l2_loss(kernels), 0.0005, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		else:
			kernels = tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01), trainable=False)
			biases  = tf.Variable(tf.constant(0.1, shape=[384]), trainable=False)
		conv   = tf.nn.conv2d(max_4, kernels, strides=[1,1,1,1], padding='SAME')
		conv_5 = tf.nn.relu(conv + biases)
		tf.summary.histogram(scope + "weights", kernels)
		tf.summary.histogram(scope + "bias", biases)
		print_activations(conv_5)

	with tf.name_scope("conv_6") as scope:
		if not (scope in envVar.FIX_LAYER):
			kernels = tf.Variable(tf.truncated_normal([4, 4, 384, 384], stddev=0.01))
			biases  = tf.Variable(tf.constant(0.1, shape=[384]))
			weight_decay = tf.multiply(tf.nn.l2_loss(kernels), 0.0005, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		else:
			kernels = tf.Variable(tf.truncated_normal([4, 4, 384, 384], stddev=0.01), trainable=False)
			biases  = tf.Variable(tf.constant(0.1, shape=[384]), trainable=False)
		conv   = tf.nn.conv2d(conv_5, kernels, strides=[1,1,1,1], padding='SAME')
		conv_6 = tf.nn.relu(conv + biases)
		tf.summary.histogram(scope + "weights", kernels)
		tf.summary.histogram(scope + "bias", biases)
		print_activations(conv_6)

	with tf.name_scope("conv_7") as scope:
		if not (scope in envVar.FIX_LAYER):
			kernels = tf.Variable(tf.truncated_normal([4, 4, 384, 256], stddev=0.01))
			biases  = tf.Variable(tf.constant(0.1, shape=[256]))
			weight_decay = tf.multiply(tf.nn.l2_loss(kernels), 0.0005, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		else:
			kernels = tf.Variable(tf.truncated_normal([4, 4, 384, 256], stddev=0.01), trainable=False)
			biases  = tf.Variable(tf.constant(0.1, shape=[256]), trainable=False)
		conv   = tf.nn.conv2d(conv_6, kernels, strides=[1,1,1,1], padding='SAME')
		conv_7 = tf.nn.relu(conv + biases)
		tf.summary.histogram(scope + "weights", kernels)
		tf.summary.histogram(scope + "bias", biases)
		print_activations(conv_7)

	with tf.name_scope("fc8") as scope:
		if not (scope in envVar.FIX_LAYER):
			reshape = tf.reshape(conv_7, [-1, 43264])
			kernels = tf.Variable(tf.truncated_normal([43264, 4096], stddev=0.01))
			biases  = tf.Variable(tf.constant(0.1, shape=[4096]))
			weight_decay = tf.multiply(tf.nn.l2_loss(kernels), 0.0005, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		else:
			reshape = tf.reshape(conv_7, [-1, 43264])
			kernels = tf.Variable(tf.truncated_normal([43264, 4096], stddev=0.01), trainable=False)
			biases  = tf.Variable(tf.constant(0.1, shape=[4096]), trainable=False)
		fc_8 = tf.nn.relu(tf.matmul(reshape, kernels) + biases)
		fc_8_drop = tf.nn.dropout(fc_8, dropout_rate)
		tf.summary.histogram(scope + "weights", kernels)
		tf.summary.histogram(scope + "bias", biases)
		print_activations(fc_8)

	with tf.name_scope("fc9") as scope:
		if not (scope in envVar.FIX_LAYER):
			kernels = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.01))
			biases  = tf.Variable(tf.constant(0.1, shape=[4096]))
			weight_decay = tf.multiply(tf.nn.l2_loss(kernels), 0.0005, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		else:
			kernels = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.01), trainable=False)
			biases  = tf.Variable(tf.constant(0.1, shape=[4096]), trainable=False)
		fc_9 = tf.nn.relu(tf.matmul(fc_8_drop, kernels) + biases)
		fc_9_drop = tf.nn.dropout(fc_9, dropout_rate)
		tf.summary.histogram(scope + "weights", kernels)
		tf.summary.histogram(scope + "bias", biases)
		print_activations(fc_9)

	with tf.name_scope("readout") as scope:
		kernels = tf.Variable(tf.truncated_normal([4096, 500], stddev=0.01))
		biases  = tf.Variable(tf.constant(0.1, shape=[500]))
		weight_decay = tf.multiply(tf.nn.l2_loss(kernels), 0.0005, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
		readout = tf.nn.relu(tf.matmul(fc_9_drop, kernels) + biases)
		tf.summary.histogram(scope + "weights", kernels)
		tf.summary.histogram(scope + "bias", biases)
		print_activations(readout)

	return readout

def cost(logits, labels):
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
						labels=labels, logits=logits, name="cross_entropy_per_example")
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def predict(logits, labels):
	correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accuracy

def predictTop5(logits, labels):
	top_prediction = tf.nn.in_top_k(logits, labels, k=5)
	accuracy = tf.reduce_mean(tf.cast(top_prediction, tf.float32))
	return accuracy

def train(loss, lr, global_step):
	lr_decay = tf.train.exponential_decay(lr, global_step, 100000, 0.1, staircase=True)
	optimizer = tf.train.MomentumOptimizer(lr_decay, momentum=0.9)
	gvs = optimizer.compute_gradients(loss)
	capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
	train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
	return train_op
