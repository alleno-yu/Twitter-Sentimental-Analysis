from tensorflow.compat.v1 import reset_default_graph
from tensorflow.compat.v2.random import set_seed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, \
    Activation, Bidirectional, LSTM, GaussianNoise, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard
import time


def modelA(vocab_size, train_data, val_data, test_data):
    # for reproducibility
    reset_default_graph()
    set_seed(42)

    # name of file is defined here
    # tensorboard file address
    NAME = "Final_Model-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir=r"Datasets\TaskA\logs\{}".format(NAME))

    # sequential model defined
    model = Sequential()

    # embeeding layer with 200 nodes and 0.1 gaussian noise rate
    model.add(Embedding(vocab_size, 200))
    model.add(GaussianNoise(0.1))

    # bilstm layer with 0.1 dropout rate before and after
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.1))

    # dense to three categories to make classification
    model.add(Dense(3))
    model.add(Activation('softmax'))

    # compile the model with optimizer, loss, and metrics
    model.compile(optimizer=Adam(1e-4),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_data, epochs=14, validation_data=val_data, callbacks=[tensorboard])

    train_loss, train_acc = model.evaluate(train_data)
    val_loss, val_acc = model.evaluate(val_data)
    test_loss, test_acc = model.evaluate(test_data)

    return train_acc, val_acc, test_acc
