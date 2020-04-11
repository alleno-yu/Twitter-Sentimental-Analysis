import tensorflow as tf

def model_building(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 200),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

def model_training(model, train_data, val_data, test_data):
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])
    model.fit(train_data, epochs=3, validation_data=val_data)
    eval_loss, eval_acc = model.evaluate(test_data)
    print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))
    return eval_acc, eval_loss

def modelB(vocab_size, train_data, val_data, test_data):
    model = model_building(vocab_size)
    eval_acc, eval_loss = model_training(model, train_data, val_data, test_data)
