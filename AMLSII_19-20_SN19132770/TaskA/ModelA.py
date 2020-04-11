import tensorflow as tf

def model_building(vocab_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, 200))
    model.add(tf.keras.layers.GaussianNoise(0.3))
    # model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
    # model.add(tf.keras.layers.Dropout(0.3))
    # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

    # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
    # model.add(tf.keras.layers.Dropout(0.3))
    for units in [64, 64]:
        model.add(tf.keras.layers.Dense(units, activation='relu'))

    # Output layer. The first argument is the number of labels.
    model.add(tf.keras.layers.Dense(3))
    return model

def model_training(model, train_data, val_data, test_data):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_data, epochs=3, validation_data=val_data)
    eval_loss, eval_acc = model.evaluate(test_data)
    print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))
    return eval_acc, eval_loss

def modelA(vocab_size, train_data, val_data, test_data):
    model = model_building(vocab_size)
    eval_acc, eval_loss = model_training(model, train_data, val_data, test_data)
