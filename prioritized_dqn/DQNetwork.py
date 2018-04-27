import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import keras.backend as K


class DQNetwork:
    def __init__(self, actions, input_shape,
                 minibatch_size=32,
                 learning_rate=0.00025,
                 discount_factor=0.99,
                 dropout_prob=0.1,
                 load_path=None,
                 logger=None):

        # Parameters
        self.actions = actions  # Size of the network output
        self.discount_factor = discount_factor  # Discount factor of the MDP
        self.minibatch_size = minibatch_size  # Size of the training batches
        self.learning_rate = learning_rate  # Learning rate
        self.dropout_prob = dropout_prob  # Probability of dropout
        self.logger = logger
        self.training_history_csv = 'training_history.csv'

        if self.logger is not None:
            self.logger.to_csv(self.training_history_csv, 'Loss,Accuracy')

        # Deep Q Network as defined in the DeepMind article on Nature
        # Ordering channels first: (samples, channels, rows, cols)
        self.model = Sequential()

        # First convolutional layer
        self.model.add(Conv2D(32, 8, strides=(4, 4),
                              padding='valid',
                              activation='relu',
                              input_shape=input_shape,
                              data_format='channels_first'))

        # Second convolutional layer
        self.model.add(Conv2D(64, 4, strides=(2, 2),
                              padding='valid',
                              activation='relu',
                              input_shape=input_shape,
                              data_format='channels_first'))

        # Third convolutional layer
        self.model.add(Conv2D(64, 3, strides=(1, 1),
                              padding='valid',
                              activation='relu',
                              input_shape=input_shape,
                              data_format='channels_first'))

        # Flatten the convolution output
        self.model.add(Flatten())

        # First dense layer
        self.model.add(Dense(512, activation='relu'))

        # Output layer
        self.model.add(Dense(self.actions))

        # Load the network weights from saved model
        if load_path is not None:
            self.load(load_path)

        self.model.compile(loss='mean_squared_error',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        # print("test, predict")
        # print(self.model.predict(np.ones(shape=(1, 4, 84, 84)), batch_size=1))

    def train(self, batch_idx, batch, ISWeights, DQN_target,
               max_loss_memory=None):
        """
        Generates inputs and targets from the given batch, trains the model on
        them.
        :param batch: iterable of dictionaries with keys 'source', 'action',
        'dest', 'reward'
        :param DQN_target: a DQNetwork instance to generate targets
        :param batch_idx: batch_idx
        :param ISWeights: importance sampling weight
        :param min_loss_memory: min loss memory
        :param max_loss_memory: max loss memory
        :param paced: whether learn paced
        """

        state_train = []
        t_train = []
        next_states = []
        actions = []
        for datapoint in batch:
            datapoint = datapoint[0]
            # Inputs are the states
            state_train.append(datapoint['source'].astype(np.float64))

            # Apply the DQN or DDQN Q-value selection
            next_state = datapoint['dest'].astype(np.float64)
            next_states.append(next_state)
            next_state_pred = DQN_target.predict(next_state).ravel()
            next_q_value = np.max(next_state_pred)

            actions.append(datapoint['action'])

            # The error must be 0 on all actions except the one taken
            t = list(self.predict(datapoint['source'])[0])
            if datapoint['final']:
                t[datapoint['action']] = datapoint['reward']
            else:
                t[datapoint['action']] = datapoint['reward'] + \
                                         self.discount_factor * next_q_value
            t_train.append(t)

        # Prepare inputs and targets
        x_train = np.asarray(state_train).squeeze()   # evaluated q
        t_train = np.asarray(t_train).squeeze()       # target q
        x_next = np.asarray(next_states).squeeze()    # next state
        actions = np.asarray(actions).squeeze()       # chosen actions index
        ISWeights = ISWeights.squeeze()               # importance weight

        # Train the model for one epoch
        h = self.model.fit(x_train,
                           t_train,
                           sample_weight=ISWeights,
                           batch_size=self.minibatch_size,
                           epochs=1)

        # update the priority
        index = np.arange(len(batch))
        q_update = self.model.predict(x_train, batch_size=32)[index, actions]
        q_target = DQN_target.predict(x_next, batch_size=32)[index, actions]
        abs_error = np.abs(q_target - q_update)
        for i in range(len(batch)):
            max_loss_memory.update(batch_idx[i], abs_error[i])

        # Log loss and accuracy
        if self.logger is not None:
            self.logger.to_csv(self.training_history_csv,
                               [h.history['loss'][0], h.history['acc'][0]])

    def predict(self, state, batch_size=1):
        """
        Feeds state to the model, returns predicted Q-values.
        :param state: a numpy.array with same shape as the network's input
        :param batch_size: batch size
        :return: numpy.array with predicted Q-values
        """
        state = state.astype(np.float32)
        return self.model.predict(state, batch_size)

    def save(self, filename=None, append=''):
        """
        Saves the model weights to disk.
        :param filename: file to which save the weights (must end with ".h5")
        :param append: suffix to append after "model" in the default filename
            if no filename is given
        """
        f = ('model%s.h5' % append) if filename is None else filename
        if self.logger is not None:
            self.logger.log('Saving model as %s' % f)
        self.model.save_weights(self.logger.path + f)

    def load(self, path):
        """
        Loads the model's weights from path.
        :param path: h5 file from which to load teh weights
        """
        if self.logger is not None:
            self.logger.log('Loading weights from file...')
        self.model.load_weights(path)

if __name__ == "__main__":
    dqn = DQNetwork(actions=9, input_shape=(4, 84, 84))