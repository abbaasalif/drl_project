from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, AdamW
from tensorflow.keras.activations import tanh, relu, sigmoid
#building the brain

class Brain(object):
    def __init__(self, learning_rate=0.001, weight_decay=1e-2 , layers=[64,32],num_actions=5, dropout=0.1, activation='relu', optimizer = 'adamw'):
        self.learning_rate = learning_rate
        states = Input(shape=(3,))
        for i in range(len(layers)):
            if i == 0:
                x = Dense(layers[i], activation=activation)(states)
                x = Dropout(dropout)(x)
            else:
                x = Dense(layers[i], activation=activation)(x)
                x = Dropout(dropout)(x)
        q_values = Dense(num_actions, activation='softmax')(x)
        self.model = Model(inputs=states, outputs=q_values)
        if optimizer == 'adam':
            self.model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        elif optimizer == 'rmsprop':
            self.model.compile(loss='mse', optimizer=RMSprop(learning_rate=learning_rate))
        elif optimizer == 'sgd':
            self.model.compile(loss='mse', optimizer=SGD(learning_rate=learning_rate))
        elif optimizer == 'adamw':
            self.model.compile(loss='mse', optimizer=AdamW(learning_rate=learning_rate, weight_decay=weight_decay))
        else:
            print('Invalid optimizer')
            return