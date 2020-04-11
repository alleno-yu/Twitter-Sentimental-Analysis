import pandas as pd
import tensorflow as tf
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import tensorflow_datasets as tfds

def read_txt():
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    df = pd.read_table(r"Datasets/TaskA.txt", sep="\t", header=None)
    df = df.drop(columns=[3])
    df = df.drop(columns=[0])
    df.columns = ["polarity", "tweet"]
    # print(df.head())
    return df

def categorize(df):
    df["polarity"].replace({"positive": 1, "negative": 2, "neutral": 0}, inplace=True)
    return df

def form_dataset(df):
    target = df.pop('polarity')
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
    return dataset

def encode(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy()[0])
    return encoded_text, label

def encode_map_fn(text, label):
    # py_func doesn't set the shape of the returned tensors.
    encoded_text, label = tf.py_function(encode,
                                         inp=[text, label],
                                         Tout=(tf.int64, tf.int64))

    # `tf.data.Datasets` work best if all components have a shape set
    #  so set the shapes manually:
    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label

def build_vocab(dataset):
    vocabulary_set = set()
    text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                   'time', 'url', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "allcaps", "elongated", "repeated",
                  'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens
        segmenter="twitter",
        corrector="twitter",
        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )
    for text_tensor, _ in dataset:
        text = str(text_tensor.numpy()[0], 'utf-8')
        some_tokens = text_processor.pre_process_doc(text)
        vocabulary_set.update(some_tokens)

    # encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
    # vocab_size = len(vocabulary_set)
    # all_encoded_data = dataset.map(encode_map_fn)

    return vocabulary_set

def train_val_test_split(all_encoded_data, TEST_SIZE, VAL_SIZE, BUFFER_SIZE, BATCH_SIZE):
    train_data = all_encoded_data.skip(TEST_SIZE + VAL_SIZE).shuffle(BUFFER_SIZE)
    train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([None], []))

    test_val_data = all_encoded_data.take(TEST_SIZE + VAL_SIZE)
    # test_val_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([None],[]))

    test_data = test_val_data.skip(VAL_SIZE)
    test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([None], []))

    val_data = test_val_data.take(VAL_SIZE)
    val_data = val_data.padded_batch(BATCH_SIZE, padded_shapes=([None], []))
    return train_data, val_data, test_data

def preprocessingA(TEST_SIZE, VAL_SIZE, BUFFER_SIZE, BATCH_SIZE):
    df = read_txt()
    df = categorize(df)
    data_set = form_dataset(df)
    vocab_set = build_vocab(data_set)
    vocab_size = len(vocab_set)
    global encoder
    encoder = tfds.features.text.TokenTextEncoder(vocab_set)
    encoded_data = data_set.map(encode_map_fn)
    train_data, val_data, test_data = train_val_test_split(
        encoded_data, TEST_SIZE, VAL_SIZE, BUFFER_SIZE, BATCH_SIZE)
    vocab_size += 2
    return train_data, val_data, test_data, vocab_size


