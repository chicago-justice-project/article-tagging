import numpy as np
import tensorflow as tf


class NumericRNN():
    """A numeric RNN model."""
    def __init__(self, config):
        """
        A simple numeric RNN model.
        """
        self.config = config

        # Set up computation graph
        self._set_up_graph()

        # How many training steps have we performed? A.k.a. "epochs" in literature.
        self.global_step = 0


    def fit(self, X_matrices, Y_matrix, session, learning_rate=0.001, verbose=False):
        """
        Trains the numeric RNN model on the training data in X_matrices and Y_vectors.

        Successive calls to `NumericRNN.fit()` will iteratively train the model without
        resetting values learned in previous calls to `NumericRNN.fit()`. If you wish to
        reset all learned values, use `NumericRNN.reset()`.

        Args:
            X_matrices: List of 2D numpy arrays, so that X_matrices[i][j,k] is
                the i'th training input matrix, at the j'th timestep and k'th feature.
            Y_matrix: A 2D numpy array, the true tags corresponding to each matrix
                in the X_matrices input.
            session: A tf.Session() object.
            learning_rate: How much the weights should change w.r.t. the gradients.
        """
        if self.global_step == 0:
            # if we have not yet trained, intialize the variables.
            session.run(tf.initialize_all_variables())

        self.global_step += 1

        X = X_matrices
        Y = Y_matrix

        # Get the length of each input in X
        seq_lens = [x.shape[0] for x in X]

        # Find the maximum sequence length
        max_len = max(seq_lens)

        # Pad each sequence in X so that all sequences have length max_len
        X = [np.vstack([x, np.zeros((max_len - x.shape[0], x.shape[1]))])
             for x in X]
        X = np.stack(X)

        # Get a boolean matrix with the same shape as X that describe
        # which parts of the sequence are real vs. added on during padding.
        mask = np.concatenate([np.concatenate([np.ones(l), np.zeros(max_len - l)])
                               for l in seq_lens]).astype(int)

        # These are all of the things we are going to input into our computation
        # graph.
        feed_dict = {
            self.X: X,
            self.Y: Y,
            self.mask: mask,
            self.learning_rate: learning_rate,
            self.sequence_length: seq_lens
        }

        # These are all of the things we want to get out of the compuation.
        # The computation graph will only go as far as needed to populate all
        # elements of this fetch_dict, so if we didn't request the training
        # operation then it would never actually train! However, this also
        # makes it convenient during testing time since we can use the same
        # computation graph, and just choose not to run it all the way to the
        # training step.
        fetch_dict = {
            'train_op': self.train_op,
            'cost': self.cost
        }
        # TODO: retrieve and use self.curr_state during testing

        out_dict = session.run(fetch_dict, feed_dict=feed_dict)

        if verbose:
            print('step {}:\n  train cost: {}'.format(self.global_step,
                                                      out_dict['cost']))


    def predict_proba(self, X, session):
        """
        Returns the probability of belonging to each class for each element in X.
        You can get the predicted class with np.arg_max()

        The state of the RNN will be preserved between successive calls to predict
        (and will be set to zero the first time predict or predict_proba is called).

        In other words, let's say that the current state of the RNN is C0. Then,
        NumericRNN.predict() is called with data X1. The state of the RNN after
        processing this data is C1. Then, predict is called again with data X2,
        the processing is started with state C1.

        If reset_state is True, then the state of the RNN will be zeroed out before
        processing.

        If freeze_state is True, then the current state will not be updated after
        running through the data in X. For example, in the above example with states
        C0, C1, C2 and data X1 and X2, if the first call to predict with data X1
        also has freeze_state set to True, then the next call to predict with X2
        will start with state C0, not state C1.

        Note: the name comes from the standard names used by sci-kit learn.

        Args:
            X: 2D data matrix with shape sequence_length x num_features.
            session: A tf.Session object.
            reset_state: Resets the state before processing the data X.
            freeze_state: The current state is not updated after processing the data X.
        """
        # The computation graph expects X to be 3-dimensionsal, an extra
        # dimension for each batch. We add a singular dimension (i.e. batch
        # size of 1).
        X = X[np.newaxis,:,:]

        # In order to simplify this example, I'm not saving state between
        # runs. The more advanced RNN example will go through that process.

        # Notice that we don't need to feed in as much here because we're
        # not running any of the training steps.
        feed_dict = {
            self.X: X,
            self.sequence_length: [X.shape[1]]
        }

        # Return the probability of belonging to each class at each
        # time step in X.
        fetch_dict = {
            'out': self.probs
        }
        # TODO: retrieve and use self.curr_state during testing

        outputs = session.run(fetch_dict, feed_dict=feed_dict)

        return outputs['out']


    def _set_up_graph(self):
        """
        Set up the computation graph.
        """
        config = self.config

        # Create placeholders for training data.
        # X is a batch_size x sequence_size x num_features tensor, i.e.
        # X[i,j,k] is the k'th dimension of the j'th element in the i'th sequence
        # We leave batch_size and sequence_size undefined for now.
        self.X = tf.placeholder(tf.float32, [None, None, config.num_features], name='X')
        # Y is a batch_size x num_tags matrix of the true classes of your data.
        self.Y = tf.placeholder(tf.int32, [None, None], name='Y')


        ############################################################
        # Create a LSTM RNN
        ############################################################

        # Create a Long-Short Term Memory cell with rnn_hidden_size hidden neurons
        self.rnn_cell = tf.nn.rnn_cell.LSTMCell(config.rnn_hidden_size,
                                                state_is_tuple=True)

        # Layer these cells rnn_layers number of times.
        # TODO: test if this creates different LSTM cells
        self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell([self.rnn_cell] * config.rnn_layers,
                                                    state_is_tuple=True)

        # The sequence_length parameter allows us to pass in sequences
        # of different lengths during training. It is a batch_size length
        # vector.
        self.sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')

        # batch_size is the size of the first dimension of X.
        batch_size = tf.shape(self.X)[0]

        # Initialize the state to zeros for each sequence in this batch.
        self.init_state = self.rnn_cell.zero_state(batch_size, tf.float32)

        # Get the outputs (values returned by the LSTM) of running all the batches
        # through the RNN. outputs is a batch_size x sequence_length x lstm_size tensor.
        # We do not need the state during training, so ignore it.
        outputs, self.curr_state = tf.nn.dynamic_rnn(self.rnn_cell, self.X,
                                                     sequence_length=self.sequence_length,
                                                     time_major=False,
                                                     initial_state=self.init_state,
                                                     scope='RNN')

        # We need to flatten the 3 dimensional output into a 2 dimensional output,
        # we don't need to distinguish between the different batches when
        # calculating the cost.
        self.output = tf.reshape(outputs, [-1, config.rnn_hidden_size])


        ############################################################
        # Convert RNN output to probability of belonging to class
        ############################################################

        # Recall that for each Y[i], we have an rnn_hidden_size sized vector
        # that is output by the RNN. We want to convert that vector
        # into a list of probabilities that describe our belief that
        # the input corresponding with Y[i] belongs to each class.
        #
        # To do this, We will first convert the rnn_hidden_size sized vector
        # to a num_classes sized vector through a simple linear model
        # (i.e. matrix multiplication).
        self.w = tf.Variable(tf.truncated_normal([config.rnn_hidden_size,
                                                  config.num_classes]),
                             name='softmax_w')
        self.b = tf.Variable(tf.truncated_normal([config.num_classes]),
                             name='softmax_b')

        # These are both (batch_size * sequence_size) x num_classes matrices
        self.logits = tf.matmul(self.output, self.w) + self.b
        self.probs = tf.nn.softmax(self.logits)


        ############################################################
        # Compute/optimize cost function
        ############################################################

        # Flatten Y
        self.Y_vec = Y_vec = tf.reshape(self.Y, [-1])

        # Placeholder for the mask that will be fed in.
        self.mask = tf.placeholder(tf.float32, [None], name='mask')

        # We use a built-in helper function that essentially
        # computes how far the probabilities are from their
        # expected values (either 0 or 1).
        self.cost = tf.nn.seq2seq.sequence_loss(
            [self.logits], [self.Y], [self.mask]
        )

        # Now optimize on that cost using (Stochastic) Gradient Descent

        # We will assing a value to learning_rate during each training run
        self.learning_rate = tf.placeholder(tf.float32)

        # We instantiate an optimizer, and tell it to minimize on the cost.
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
