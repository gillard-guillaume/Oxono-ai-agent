import tensorflow as tf
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
    
    def train_self_play(self):
        """self play td1 ??"""
        return

    def train_td_one(self):
        """supervised td1"""
        return
    

    def train_supervised(self, X_train, y_train, sample_weight=None, batch_size=32, epochs=20, callbacks=None, validation_data=None):
        history = self.model.fit(X_train, y_train, sample_weight=sample_weight, validation_data=validation_data, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0)
        return history