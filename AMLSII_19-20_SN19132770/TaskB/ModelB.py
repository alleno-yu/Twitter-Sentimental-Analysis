from tensorflow.compat.v1 import reset_default_graph
from tensorflow.compat.v2.random import set_seed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, \
    Bidirectional, LSTM, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import TensorBoard
import time


def modelB(vocab_size, train_data, val_data, test_data):

    reset_default_graph()
    set_seed(42)

    NAME = "Final_Model-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir=r"Datasets\TaskB\logs\{}".format(NAME))

    model = Sequential()

    model.add(Embedding(vocab_size, 200))

    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.1))

    for l in range(2):
        model.add(Dense(32, activation='relu'))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer=Adam(8e-5),
                  loss=BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_data, epochs=10, validation_data=val_data, callbacks=[tensorboard])

    train_loss, train_acc = model.evaluate(train_data)
    val_loss, val_acc = model.evaluate(val_data)
    test_loss, test_acc = model.evaluate(test_data)

    return train_acc, val_acc, test_acc

