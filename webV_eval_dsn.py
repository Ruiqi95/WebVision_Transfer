import tensorflow as tf
import webV_input as readInput
import webV_net_dsn as net
import envvar as envVar
import re

def getSTEP(string):
	return int(re.findall("\d+", string)[-1])

with tf.Graph().as_default() as graph:

	lr = 0.01
	batch_size  = 32
	EPOCH = 100
	train_dir = './tf.Events/'

	dataset, data_amount = readInput.inputs(False, batch_size)
	iterator = dataset.make_initializable_iterator()
	next_examples, next_labels = iterator.get_next()

	test_dataset, test_amount = readInput.inputs_dsn(True, batch_size)
	test_iterator = test_dataset.make_initializable_iterator()
	test_examples, test_labels = test_iterator.get_next()

	images_source = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
	labels_source = tf.placeholder(tf.int64, shape=[None])
	keep_prob    = tf.placeholder(tf.float32, name = 'keep_prob')
	shape_source = tf.placeholder(tf.float32, shape=[])
	global_step  = tf.Variable(0, name='global_step', trainable=False)

	with tf.variable_scope('Share') as scope:
		repShE_S = net.inference_SharedEncoder(images_source, shape_source)

	logits = net.inference_Classifier(repShE_S, keep_prob)
	accuracy=net.predict(logits, labels_source)

	acc_placeholder = tf.placeholder(tf.float32)
	summary = tf.summary.scalar("Test_Accuracy", acc_placeholder)

	writer = tf.summary.FileWriter(train_dir)
	writer.add_graph(graph)
	saver = tf.train.Saver(max_to_keep=40)

	chkpts = tf.train.get_checkpoint_state("./DSN").all_model_checkpoint_paths


	with tf.Session() as sess:
		for z in chkpts:
			print(z)
			divider = 0
			total_acc = 0
			saver.restore(sess, z)
			sess.run(test_iterator.initializer)
		
			while(True):
				#print(total_acc)
				x, y = sess.run([test_examples, test_labels])
				if not (len(x) == 32):
					break;
				feed_dict={ images_source:x, 
							labels_source:y,
							keep_prob:1,
							shape_source:len(x)}

				acc = sess.run(accuracy, feed_dict=feed_dict)
				print(acc)
				total_acc = total_acc + acc
				divider = divider + 32

			summary_str = sess.run(summary, 
									feed_dict = {acc_placeholder:total_acc/divider})
			writer.add_summary(summary_str, getSTEP(str(z)))


			
