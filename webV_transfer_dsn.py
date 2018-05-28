import tensorflow as tf
import webV_input as readInput
import webV_net_transfer_dsn as net
import envvar as envVar
import re

def getSTEP(string):
	return int(re.findall("\d+", string)[-1])

with tf.Graph().as_default() as graph:

	lr = 0.01
	batch_size  = 32
	EPOCH = 100
	train_dir = './tf.Events/'

	dataset, data_amount = readInput.inputs_dsn(False, batch_size)
	dataset_source = dataset[0]
	iterator_source = dataset_source.make_initializable_iterator()
	next_examples_source, next_labels_source = iterator_source.get_next()

	images_source = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
	labels_source = tf.placeholder(tf.int64, shape=[None])
	keep_prob    = tf.placeholder(tf.float32, name = 'keep_prob')
	labels_source = tf.placeholder(tf.int64, shape=[None])
	shape_source = tf.placeholder(tf.float32, shape=[])
	global_step  = tf.Variable(0, name='global_step', trainable=False)

	with tf.variable_scope('Share') as scope:
		repShE_S = net.inference_SharedEncoder(images_source, shape_source)

	logits = net.inference_Classifier(repShE_S, keep_prob)
	accuracy=net.predict(logits, labels_source)
	loss = net.cost_entropy(logits, labels_source)
	train  = net.train(loss, lr, global_step)
	accuracy=net.predict(logits, labels_source)

	tf.summary.scalar('Total Loss', loss)
	tf.summary.scalar("Accuracy", accuracy)
	summary = tf.summary.merge_all()
	
	writer = tf.summary.FileWriter(train_dir)
	writer.add_graph(graph)
	saver = tf.train.Saver(max_to_keep=40)

	with tf.Session() as sess:

		sess.run(iterator_source.initializer)
		saver.restore(sess, tf.train.latest_checkpoint("./DSN"))
		print(tf.GraphKeys.TRAINABLE_VARIABLES)
		sess.run(global_step.initializer)
		sess.run([i.initializer for i in tf.trainable_variables()])

		
		while(True):
			x, y = sess.run([next_examples_source, next_labels_source])
			print(len(x))
			if not (len(x) == 32):
				continue
			feed_dict={ images_source:x, 
						labels_source:y,
						keep_prob:0.5,
						shape_source:len(x),}

			_, _loss, _acc, step = sess.run([train, loss, accuracy, global_step], 
												feed_dict = feed_dict)
			if(step%1000==0):
				summary_str = sess.run(summary, feed_dict = feed_dict)
				writer.add_summary(summary_str, step)
				
			if(step%50000==0):
				saver.save(sess, train_dir, global_step=step)

			j = step % ( data_amount / batch_size)
			print('STEP <{}> : BATCH [{}] Loss = {}, Accuracy = {}'.format(step, j, _loss, _acc))


			
