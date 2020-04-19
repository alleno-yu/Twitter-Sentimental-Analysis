import pandas as pd
import tensorflow as tf
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import tensorflow_datasets as tfds
import pickle

def read_txt():
    df = pd.read_table(r"Datasets/TaskB.txt", sep="\t", header=None)
    df = df.drop(columns=[4])
    df = df.drop(columns=[0])
    df.columns = ["topic", "polarity", "tweet"]
    return df

def categorize(df):
    df["polarity"].replace({"positive": 1, "negative": 0}, inplace=True)
    return df

def form_dataset(df):
    target = df.pop('polarity')
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
    return dataset

def encode(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy()[1])
    return encoded_text, label

def encode_map_fn(text, label):
    encoded_text, label = tf.py_function(encode,
                                         inp=[text, label],
                                         Tout=(tf.int64, tf.int64))
    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label

def build_vocab(dataset):
    vocabulary_set = set()
    text_processor = TextPreProcessor(
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                   'time', 'url', 'date', 'number'],
        annotate={"hashtag", "allcaps", "elongated", "repeated",
                  'emphasis', 'censored'},
        fix_html=True,
        segmenter="twitter",
        corrector="twitter",

        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        dicts=[emoticons]
    )
    for text_tensor, _ in dataset:
        text = str(text_tensor.numpy()[1], 'utf-8')
        some_tokens = text_processor.pre_process_doc(text)
        vocabulary_set.update(some_tokens)

    return vocabulary_set

def train_val_test_split(all_encoded_data, TEST_SIZE, VAL_SIZE, BUFFER_SIZE, BATCH_SIZE):
    all_encoded_data.shuffle(BUFFER_SIZE, seed=42)
    train_data = all_encoded_data.skip(TEST_SIZE + VAL_SIZE)
    train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([None], []))
    test_val_data = all_encoded_data.take(TEST_SIZE + VAL_SIZE)
    test_data = test_val_data.skip(VAL_SIZE)
    test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([None], []))
    val_data = test_val_data.take(VAL_SIZE)
    val_data = val_data.padded_batch(BATCH_SIZE, padded_shapes=([None], []))
    return train_data, val_data, test_data

def preprocessingB(TEST_SIZE, VAL_SIZE, BUFFER_SIZE, BATCH_SIZE):
    df = read_txt()
    df = categorize(df)
    data_set = form_dataset(df)
    #===================================
    # vocab_set = build_vocab(data_set)
    # vocab_list = list(vocab_set)
    # file = open('vocab_list.pickle', 'wb')
    # pickle.dump(vocab_list, file)
    # file.close()
    #===================
    file = open('TaskB/vocab_list.pickle', 'rb')
    vocab_list = pickle.load(file)
    file.close()
    #====================
    print(vocab_list)
    vocab_size = len(vocab_list)
    global encoder
    encoder = tfds.features.text.TokenTextEncoder(vocab_list)
    encoded_data = data_set.map(encode_map_fn)
    train_data, val_data, test_data = train_val_test_split(
        encoded_data, TEST_SIZE, VAL_SIZE, BUFFER_SIZE, BATCH_SIZE)
    vocab_size += 2
    return train_data, val_data, test_data, vocab_size


