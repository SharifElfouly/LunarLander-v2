import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

class NN:
    """Navigates the ship.

    A simple 3 layered Neural Network (NN) used to navigate the ship
    (Lunar Lander).
    """
    def __init__(self, n_hidden_units=32, batch_size=128, epochs=2):
        """Initializes a Sequential NN with the keras Sequential object.

        It takes as input the state of the ship (shape=(8,)) and returns one of
        four actions.

        Args:
            n_hidden_units (int, optional): Number of hidden units. Defaults to
                15.
            batch_size (int, optional): Batch size while training.
            epochs (int, optional): Number of epochs to train.
        """
        self.n_hidden_units = n_hidden_units
        self.batch_size = batch_size
        self.epochs = epochs

        self.model = Sequential()
        self.model.add(Dense(units=n_hidden_units, activation='sigmoid', input_dim=8))
        self.model.add(Dense(units=4, activation='softmax'))

        optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        self.model.compile(loss='kullback_leibler_divergence', optimizer=optimizer, metrics=['accuracy'])

    def train(self, X, y):
        """Trains the NN.

        Training is always done on the whole normalized training set of one
        generation. The labels are the actions that where taken.

        Args:
            X (np.array): States (Shape=(None, 8)).
            y (np.array): Actions (Shape=(None, 4)).
        """
        self.model.fit(X, y, epochs=self.epochs, batch_size=None, steps_per_epoch=2000)

    def prediction(self, X):
        """Predicts action on given input.

        Args:
            X (np.array): Input of the environment.

        Returns:
            action (np.array): One hot vector (Shape=(4,)) representing one of
            the four available actions.
        """
        # otherwise keras has problems.
        X = X.reshape(1,8)

        prediction = self.model.predict(X)
        action = np.zeros(4,)
        action[np.argmax(prediction)] = 1

        return action

    def save(self, name):
        """Saves the model.

        Args:
            name (str): Name of the model.
        """
        # this is where are models are saved.
        FULL_PATH = '/Users/Shafou/Desktop/LunarLander-v2/nn/'
        self.model.save(FULL_PATH + name)

    def eval(self, X, y):
        """Returns the accuracy of the NN on the test data.

        Args:
            X (np.array): States (Shape=(None, 8)).
            y (np.array): Actions (Shape=(None, 4)).
        """
        accuracy = self.model.evaluate(X, y, batch_size=self.batch_size)
        return accuracy[1]

if __name__ == '__main__':
    random_X = np.random.normal(size=(500, 8))
    random_y = np.random.normal(size=(500, 4))
    nn = NN()
    nn.train(random_X, random_y)
    print(nn.eval(random_X, random_y))
