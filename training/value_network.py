import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
from keras import layers



class ValueNetwork():

    def __init__(self, lr=0.001, optimizer='adam', loss='mse', conv_number=3, filters=32, kernel_size=(3,3), dense_size=128, activation='relu'):
        self.lr = lr
        self.optimizer = optimizer
        self.loss = loss
        self.conv_number = conv_number
        self.filters = filters
        self.kernel_size = kernel_size
        self.dense_size =dense_size
        self.activation = activation
        self.model = self.create_model()

    def create_model(self):
        inputs = tf.keras.Input(shape=(6, 6, 7))
        x = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding='same', activation='relu')(inputs)
        for _ in range(self.conv_number - 1):
            x = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding='same', activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(self.dense_size, activation=self.activation)(x)
        outputs = layers.Dense(1, activation='tanh')(x)
        model = tf.keras.Model(inputs, outputs)
        optimizer = tf.keras.optimizers.get({"class_name": self.optimizer,"config": {"learning_rate": self.lr}})
        model.compile(optimizer=optimizer, loss=self.loss)
        return model
    
    def td_coherent_learning(self, states, next_states, rewards, terminal, gamma=0.99):
        next_values = self.model(next_states, training=False).numpy()
        targets = rewards + gamma * next_values * (1 - terminal)
        return targets
    
    def train_model(self, states, next_states, rewards, dones, gamma=0.99):

        # TD target
        next_values = self.model(next_states, training=False).numpy()
        targets = rewards + gamma * next_values * (1 - dones)

        # train
        self.model.fit(states, targets, batch_size=32, epochs=1, verbose=0)