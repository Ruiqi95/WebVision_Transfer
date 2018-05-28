import tensorflow as tf
import webV_input as readInput
import webV_net as net
import envvar as envVar
import re

def getSTEP(string):
	return int(re.findall("\d+", string)[-1])

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
	labels_batch = tf.placeholder(tf.int64, shape=(None))
	keep_prob    = tf.placeholder(tf.float32, name = 'keep_prob')

	logits = net.inference(images_batch, keep_prob)
	accuracy=net.predictTop5(logits, labels_batch)

	acc_placeholder = tf.placeholder(tf.float32)
	summary = tf.summary.scalar("Test_Accuracy", acc_placeholder)

	writer = tf.summary.FileWriter(train_dir)
	writer.add_graph(graph)
	saver = tf.train.Saver()
	
	chkpts = tf.train.get_checkpoint_state(envVar.SOURCE).all_model_checkpoint_paths

	with tf.Session() as sess:
		for z in chkpts:
			print(z)
			total_acc = 0
			saver.restore(sess, z)
			sess.run(test_iterator.initializer)
			for i in range(5):
				print(getSTEP(str(z)))
				x, y = sess.run([test_examples, test_labels])
				for i in range(int(len(x)/100)):
					print(i)
		
					feed_dict={images_batch:x[100*i:100+100*i],
								labels_batch:y[100*i:100+100*i],
								keep_prob:1}
					acc = sess.run(accuracy, feed_dict=feed_dict)
					total_acc = total_acc + acc
			summary_str = sess.run(summary, 
									feed_dict = {acc_placeholder:total_acc/(int(len(x)*5)/100)})
			writer.add_summary(summary_str, getSTEP(str(z)))
		print("Done")