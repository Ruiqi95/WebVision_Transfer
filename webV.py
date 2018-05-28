import tensorflow as tf
import webV_input as readInput
import webV_net as net

with tf.Graph().as_default() as graph:

	lr = 0.01
	batch_size  = 64
	EPOCH = 100
	train_dir = './tf.Events/'

	dataset, data_amount = readInput.inputs(False, batch_size)
	iterator = dataset.make_initializable_iterator()
	next_examples, next_labels = iterator.get_next()

	test_dataset, test_amount = readInput.inputs(True, batch_size)
	test_iterator = test_dataset.make_initializable_iterator()
	test_examples, test_labels = test_iterator.get_next()

	images_batch = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
	labels_batch = tf.placeholder(tf.int64, shape=[None])
	keep_prob    = tf.placeholder(tf.float32, name = 'keep_prob')
	global_step  = tf.Variable(0, name='global_step', trainable=False)

	logits = net.inference(images_batch, keep_prob)
	loss   = net.cost(logits, labels_batch)
	train  = net.train(loss, lr, global_step)
	accuracy=net.predict(logits, labels_batch)

	tf.summary.scalar('Loss', loss)
	tf.summary.scalar("Accuracy", accuracy)
	tf.summary.image("Image", images_batch)
	summary = tf.summary.merge_all()

	writer = tf.summary.FileWriter(train_dir)
	writer.add_graph(graph)
	saver = tf.train.Saver(max_to_keep=40)

	print("Amount of examples : {}".format(data_amount))
	print("Amount of test_ex  : {}".format(test_amount))

	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())
		sess.run(iterator.initializer)
		
		while(True):
			x, y = sess.run([next_examples, next_labels])
			feed_dict={ images_batch:x, labels_batch:y, keep_prob:0.5 }

			_, _loss, _acc, step = sess.run([train, loss, accuracy, global_step], 
												feed_dict = feed_dict)
			if(step%100==0):
				summary_str = sess.run(summary, feed_dict = feed_dict)
				writer.add_summary(summary_str, step)
				
			if(step%500==0):
				saver.save(sess, train_dir, global_step=step)

			j = step % ( data_amount / batch_size)
			print('STEP <{}> : BATCH [{}] Loss = {}, Accuracy = {}'.format(step, j, _loss, _acc))

			
