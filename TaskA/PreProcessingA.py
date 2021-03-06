import pandas as pd
import tensorflow as tf
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import tensorflow_datasets as tfds
import pickle

def read_txt():
    # use pandas package to convert txt file to dataframe and process
    df = pd.read_table(r"Datasets/TaskA.txt", sep="\t", header=None)
    df = df.drop(columns=[3])
    df = df.drop(columns=[0])
    df.columns = ["polarity", "tweet"]
    return df

def categorize(df):
    # categories to numbers for better processing
    df["polarity"].replace({"positive": 1, "negative": 2, "neutral": 0}, inplace=True)
    return df

def form_dataset(df):
    # define target and convert dataframe to tensorflow dataset
    target = df.pop('polarity')
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
    return dataset

def encode(text_tensor, label):
    # encode on text tensors, which is the twitter message
    encoded_text = encoder.encode(text_tensor.numpy()[0])
    return encoded_text, label

def encode_map_fn(text, label):
    # map the encoded text with labels
    encoded_text, label = tf.py_function(encode,
                                         inp=[text, label],
                                         Tout=(tf.int64, tf.int64))
    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label

def build_vocab(dataset):
    # use text processing tool to do word normalization, annotation, segmentation, tokenization, and spell correction
    # return a vocabulary set
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
        text = str(text_tensor.numpy()[0], 'utf-8')
        some_tokens = text_processor.pre_process_doc(text)
        vocabulary_set.update(some_tokens)

    return vocabulary_set

def train_val_test_split(all_encoded_data, TEST_SIZE, VAL_SIZE, BUFFER_SIZE, BATCH_SIZE):
    # shuffle the dataset with reproducibility
    # split the dataset into training, validation, and test set
    # padding the data with zeros
    all_encoded_data.shuffle(BUFFER_SIZE, seed=42)
    train_data = all_encoded_data.skip(TEST_SIZE + VAL_SIZE)
    train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([None], []))

    test_val_data = all_encoded_data.take(TEST_SIZE + VAL_SIZE)

    test_data = test_val_data.skip(VAL_SIZE)
    test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([None], []))

    val_data = test_val_data.take(VAL_SIZE)
    val_data = val_data.padded_batch(BATCH_SIZE, padded_shapes=([None], []))
    return train_data, val_data, test_data

def preprocessingA(TEST_SIZE, VAL_SIZE, BUFFER_SIZE, BATCH_SIZE):
    # the preprocessing function which to be called at the main.py
    df = read_txt()
    df = categorize(df)
    data_set = form_dataset(df)
    file = open(r'TaskA/vocab_list.pickle', 'rb')
    vocab_list = pickle.load(file)
    vocab_size = len(vocab_list)
    file.close()
    global encoder
    encoder = tfds.features.text.TokenTextEncoder(vocab_list)
    encoded_data = data_set.map(encode_map_fn)
    train_data, val_data, test_data = train_val_test_split(
        encoded_data, TEST_SIZE, VAL_SIZE, BUFFER_SIZE, BATCH_SIZE)
    for i in train_data.take(1):
        print(i)
    vocab_size += 2
    return train_data, val_data, test_data, vocab_size