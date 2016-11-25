'''
I used mainly the tensorflow translation example:
https://github.com/tensorflow/tensorflow/

and semi-based this off the sentiment analyzer here:
http://deeplearning.net/tutorial/lstm.html

Written by: Dominik Kaukinen
----------------------------------------------------------------------

python ptrain.py --job_name="ps" --task_index=0 --workers=mac-To-be-filled-by-O-E-M:2223,mac-To-be-filled-by-O-E-M:2224
python ptrain.py --job_name="worker" --task_index=0 --workers=mac-To-be-filled-by-O-E-M:2223,mac-To-be-filled-by-O-E-M:2223
python ptrain.py --job_name="worker" --task_index=1 --workers=mac-To-be-filled-by-O-E-M:2224,mac-To-be-filled-by-O-E-M:2224
'''
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell, seq2seq
from tensorflow.python.platform import gfile
import numpy as np
import sys
import math
import os
import ConfigParser
import random
import time
from six.moves import xrange
import util.dataprocessor
import util.vocabmapping

# Defaults for network parameters

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "data/", "Path to main data directory.")
flags.DEFINE_string("checkpoint_dir", "data/checkpoints/", "Directory to store/restore checkpoints")

'''
sentiment_network_params
'''
hidden_size = 50      # number of units in a hidden layer
#hidden_size = 25
num_layers = 2        # number of hidden lstm layers
#num_layers = 1
max_gradient_norm = 5   # maximum size of gradient -> pram grad_clip
lr_decay_factor = 0.97
batch_size = 200
#batch_size = 50
max_epoch = 50
train_frac = 0.7
dropout = 0.5
max_vocab_size = 20000
#max_vocab_size = 10000
forward_only = False    # whether to run backward pass or not

'''
general params
'''
max_seq_length = 200    # the maximum length of the input sequence
use_config_file_if_checkpoint_exists = True
steps_per_checkpoint = 50
'''
ClusterSpec
'''
#parameter_server = ["mac-To-be-filled-by-O-E-M:2222"]
parameter_server = ["sn26:2222"]

tf.app.flags.DEFINE_string("workers", "", "hostname : port")

'''
Server
'''
tf.app.flags.DEFINE_string("job_name", "", "ps || worker")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

def main():
    workers = FLAGS.workers.split(",")

    # Create a cluster from the parameter_server and workers
    cluster = tf.train.ClusterSpec({"ps": parameter_server, "worker": workers})

    # Create and start a server for the local task
    server = tf.train.Server(cluster,
                            job_name=FLAGS.job_name,
                            task_index=FLAGS.task_index)

    util.dataprocessor.run(max_seq_length, max_vocab_size)

    # create model
    print "Creating model with..."
    print "Number of hidden layers: {0}".format(num_layers)
    print "Number of units per layer: {0}".format(hidden_size)
    print "Dropout: {0}".format(dropout)
    vocabmapping = util.vocabmapping.VocabMapping()
    vocab_size = vocabmapping.getSize()
    print "Vocab size is: {0}".format(vocab_size)
    path = os.path.join(FLAGS.data_dir, "processed/")
    infile = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # randomize data order
    print infile
    data = np.load(os.path.join(path, infile[0]))
    for i in range(1, len(infile)):
        data = np.vstack((data, np.load(os.path.join(path, infile[i]))))
    np.random.shuffle(data)
    # data = data[:3000]
    num_batches = len(data) / batch_size
    # 70/30 splir for train/test
    train_start_end_index = [0, int(train_frac * len(data))]
    test_start_end_index = [int(train_frac * len(data)) + 1, len(data) - 1]
    print "Number of training examples per batch: {0}, \
    \nNumber of batches per epoch: {1}".format(batch_size, num_batches)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        #For each local worker do...
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):

            #writer = tf.train.SummaryWriter("/tmp/tb_logs", sess.graph)
            num_classes = 2
            vocab_size = vocab_size
            #learning_rate = tf.Variable(float(learning_rate), trainable=False)
            learning_rate = tf.Variable(float(0.01), trainable=False)
            learning_rate_decay_op = learning_rate.assign(learning_rate * lr_decay_factor)
            initializer = tf.random_uniform_initializer(-1, 1)
            batch_pointer = 0
            seq_input = []
            seq_lengths = []
            projection_dim = hidden_size
            global_step = tf.Variable(0, trainable=False)

            # seq_input: list of tensors, each tensor is size max_seq_length
            # target: a list of values betweeen 0 and 1 indicating target scores
            # seq_lengths:the early stop lengths of each input tensor
            str_summary_type = tf.placeholder(tf.string, name="str_summary_type")
            seq_input = tf.placeholder(tf.int32, shape=[None, max_seq_length],name="input")
            target = tf.placeholder(tf.float32, name="target", shape=[None,num_classes])
            seq_lengths = tf.placeholder(tf.int32, shape=[None], name="early_stop")

            dropout_keep_prob_embedding = tf.constant(dropout)
            dropout_keep_prob_lstm_input = tf.constant(dropout)
            dropout_keep_prob_lstm_output = tf.constant(dropout)

            with tf.variable_scope("embedding"), tf.device("/cpu:0"):
                W = tf.get_variable("W",[vocab_size, hidden_size],initializer=tf.random_uniform_initializer(-1.0, 1.0))
                embedded_tokens = tf.nn.embedding_lookup(W, seq_input)
                embedded_tokens_drop = tf.nn.dropout(embedded_tokens, dropout_keep_prob_embedding)

            rnn_input = [embedded_tokens_drop[:, i, :] for i in range(max_seq_length)]
            with tf.variable_scope("lstm") as scope:
                single_cell = rnn_cell.DropoutWrapper(
                    rnn_cell.LSTMCell(hidden_size,
                                                  initializer=tf.random_uniform_initializer(-1.0, 1.0),
                                                  state_is_tuple=True),
                                                  input_keep_prob=dropout_keep_prob_lstm_input,
                                                  output_keep_prob=dropout_keep_prob_lstm_output)
                cell = rnn_cell.MultiRNNCell([single_cell] * num_layers, state_is_tuple=True)

                initial_state = cell.zero_state(batch_size, tf.float32)

                rnn_output, rnn_state = rnn.rnn(cell, rnn_input,initial_state=initial_state,
                                                                sequence_length=seq_lengths)

                states_list = []
                for state in rnn_state[-1]:
                    states_list.append(state)
                avg_states = tf.reduce_mean(tf.pack(states_list), 0)

            with tf.variable_scope("output_projection"):
                W = tf.get_variable("W",[hidden_size, num_classes],initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable("b",[num_classes],initializer=tf.constant_initializer(0.1))
                scores = tf.nn.xw_plus_b(rnn_state[-1][0], W, b)
                y = tf.nn.softmax(scores)
                predictions = tf.argmax(scores, 1)

            with tf.variable_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(scores, target, name="ce_losses")
                total_loss = tf.reduce_sum(losses)
                mean_loss = tf.reduce_mean(losses)

            with tf.variable_scope("accuracy"):
                correct_predictions = tf.equal(predictions, tf.argmax(target, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_predictions,"float"),name="accuracy")

            params = tf.trainable_variables()
            if not forward_only:
                with tf.name_scope("train") as scope:
                    opt = tf.train.AdamOptimizer(learning_rate)
                gradients = tf.gradients(losses, params)
                clipped_gradients, norm = tf.clip_by_global_norm(
                    gradients, max_gradient_norm)
                with tf.name_scope("grad_norms") as scope:
                    grad_summ = tf.scalar_summary("grad_norms", norm)
                update = opt.apply_gradients(zip(clipped_gradients,params), global_step=global_step)
                loss_summ = tf.scalar_summary("{0}_loss".format(str_summary_type),mean_loss)
                acc_summ = tf.scalar_summary("{0}_accuracy".format(str_summary_type),accuracy)
                #merged = tf.merge_summary([loss_summ, acc_summ])
                merged = tf.merge_all_summaries()
            saver = tf.train.Saver(tf.all_variables())

        #ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        #if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        #    print "Reading model parameters from {0}".format(ckpt.model_checkpoint_path)
         #   saver.restore(session, ckpt.model_checkpoint_path)
        #else:
            print "Created model with fresh parameters."
            init_op = tf.initialize_all_variables()

        # train model and save to checkpoint
            print "Beggining training..."
            print "Maximum number of epochs to train for: {0}".format(max_epoch)
            print "Batch size: {0}".format(batch_size)
            print "Starting learning rate: {0}".format(learning_rate)
            print "Learning rate decay factor: {0}".format(lr_decay_factor)

            step_time, loss = 0.0, 0.0
            previous_losses = []
            tot_steps = num_batches * max_epoch

            # initData
            '''
            Split data into train/test sets and load into memory
            '''
            train_batch_pointer = 0
            test_batch_pointer = 0
            # cutoff non even number of batches
            targets = (data.transpose()[-2]).transpose()
            onehot = np.zeros((len(targets), 2))
            onehot[np.arange(len(targets)), targets] = 1
            sequence_lengths = (data.transpose()[-1]).transpose()
            data = (data.transpose()[0:-2]).transpose()

            train_data = data[train_start_end_index[0]: train_start_end_index[1]]
            test_data = data[test_start_end_index[0]:test_start_end_index[1]]
            test_num_batch = len(test_data) / batch_size

            num_train_batches = len(train_data) / batch_size
            num_test_batches = len(test_data) / batch_size
            train_cutoff = len(train_data) - (len(train_data) % batch_size)
            test_cutoff = len(test_data) - (len(test_data) % batch_size)
            train_data = train_data[:train_cutoff]
            test_data = test_data[:test_cutoff]

            train_sequence_lengths = sequence_lengths[train_start_end_index[0]:train_start_end_index[1]][:train_cutoff]
            train_sequence_lengths = np.split(train_sequence_lengths,num_train_batches)
            train_targets = onehot[train_start_end_index[0]:train_start_end_index[1]][:train_cutoff]
            train_targets = np.split(train_targets, num_train_batches)
            train_data = np.split(train_data, num_train_batches)

            print "Test size is: {0}, splitting into {1} batches".format(len(test_data), num_test_batches)
            test_data = np.split(test_data, num_test_batches)
            test_targets = onehot[test_start_end_index[0]:test_start_end_index[1]][:test_cutoff]
            test_targets = np.split(test_targets, num_test_batches)
            test_sequence_lengths = sequence_lengths[test_start_end_index[0]:test_start_end_index[1]][:test_cutoff]
            test_sequence_lengths = np.split(test_sequence_lengths,num_test_batches)

            sv = tf.train.Supervisor(#is_chief=(FLAGS.task_index == 0),
                                     logdir="/tmp/tb_logs_sup",
                                     saver=None,
                                     global_step=global_step,
                                     summary_op=None,
                                     init_op=init_op)

            #with sv.managed_session(server.target) as sess:
            with sv.prepare_or_wait_for_session(server.target) as sess:

                writer = tf.train.SummaryWriter("/tmp/tb_logs", sess.graph)
                # starting at step 1 to prevent test set from running after first batch
                for step in xrange(1, tot_steps):
                    # Get a batch and make a step.
                    start_time = time.time()
                    #getbatch()
                    input_feed = {}
                    # for i in xrange(max_seq_length):
                    input_feed[seq_input.name] = train_data[train_batch_pointer]  # .transpose()
                    input_feed[target.name] = train_targets[train_batch_pointer]
                    input_feed[seq_lengths.name] = train_sequence_lengths[train_batch_pointer]

                    train_batch_pointer += 1
                    train_batch_pointer = train_batch_pointer % len(train_data)

                    input_feed[str_summary_type.name] = "train"
                    output_feed = [merged, mean_loss, update]
                    outputs = sess.run(output_feed, input_feed)

                    str_summary, step_loss = outputs[0], outputs[1]

                    step_time += (time.time() - start_time) / steps_per_checkpoint
                    loss += step_loss / steps_per_checkpoint
                    # writer.add_summary(str_summary, step)
                    # Once in a while, we save checkpoint, print statistics, and run evals.
                    if step % steps_per_checkpoint == 0:
                        writer.add_summary(str_summary, step)
                        # Print statistics for the previous epoch.
                        print ("global step %d learning rate %.7f step-time %.2f loss %.4f"
                               % (global_step.eval(), learning_rate.eval(),step_time, loss))
                        # Decrease learning rate if no improvement was seen over last 3 times.
                        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                            sess.run(learning_rate_decay_op)
                        previous_losses.append(loss)
                        # Save checkpoint and zero timer and loss.
                        step_time, loss, test_accuracy = 0.0, 0.0, 0.0
                        # Run evals on test set and print their accuracy.
                        print "Running test set"
                        for test_step in xrange(len(test_data)):

                            _input_feed = {}
                            # for i in xrange(max_seq_length):
                            _input_feed[seq_input.name] = test_data[test_batch_pointer]  # .transpose()
                            _input_feed[target.name] = test_targets[test_batch_pointer]
                            _input_feed[seq_lengths.name] = test_sequence_lengths[test_batch_pointer]

                            test_batch_pointer += 1
                            test_batch_pointer = test_batch_pointer % len(test_data)

                            _input_feed[str_summary_type.name] = "test"
                            _output_feed = [merged, mean_loss, y, accuracy]

                            outputs = sess.run(_output_feed, _input_feed)
                            str_summary, test_loss, _, _accuracy = outputs[0], outputs[1], outputs[2], outputs[3]

                            loss += test_loss
                            test_accuracy += _accuracy
                        normalized_test_loss, normalized_test_accuracy = loss / \
                            len(test_data), test_accuracy / len(test_data)
                        #checkpoint_path = os.path.join(FLAGS.checkpoint_dir,"sentiment{0}.ckpt".format(normalized_test_accuracy))
                        #saver.save(sess,checkpoint_path,global_step=global_step)
                        writer.add_summary(str_summary, step)
                        print "Avg Test Loss: {0}, Avg Test Accuracy: {1}".format(normalized_test_loss, normalized_test_accuracy)
                        print "-------Step {0}/{1}------".format(step, tot_steps)
                        loss = 0.0
                        sys.stdout.flush()
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir,"sentiment{0}.ckpt".format(normalized_test_accuracy))
                saver.save(sess,checkpoint_path,global_step=global_step)
            sv.stop()

if __name__ == '__main__':
    main()
