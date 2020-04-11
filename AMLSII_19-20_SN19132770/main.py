import pandas as pd
import tensorflow as tf
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import tensorflow_datasets as tfds
from TaskA.ModelA import modelA
from TaskB.ModelB import modelB
from TaskA.PreProcessingA import preprocessingA
from TaskB.PreProcessingB import preprocessingB

# =============================================
# constant
BUFFER_SIZE = 50000
BATCH_SIZE = 64
DATASET_SIZE_A = 20632
TEST_SIZE_A = int(0.15 * DATASET_SIZE_A)
VAL_SIZE_A = int(0.15 * DATASET_SIZE_A)

DATASET_SIZE_B = 14253
TEST_SIZE_B = int(0.15 * DATASET_SIZE_B)
VAL_SIZE_B = int(0.15 * DATASET_SIZE_B)

# ======================================================================================================================
# Task A Data PreProcessing
train_data_A, val_data_A, test_data_A, vocab_size_A = preprocessingA(
    TEST_SIZE=TEST_SIZE_A, VAL_SIZE=VAL_SIZE_A, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE)
# ======================================================================================================================
# Task B Data PreProcessing
train_data_B, val_data_B, test_data_B, vocab_size_B = preprocessingB(
    TEST_SIZE=TEST_SIZE_B, VAL_SIZE=VAL_SIZE_B, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE)
# ======================================================================================================================
# Task A
model_A = modelA(vocab_size=vocab_size_A, train_data=train_data_A, val_data=val_data_A, test_data=test_data_A)

# ======================================================================================================================
# Task B
# model_B = modelB(vocab_size=vocab_size_B, train_data=train_data_B, val_data=val_data_B, test_data=test_data_B)
# model_B = B(args...)
# acc_B_train = model_B.train(args...)
# acc_B_test = model_B.test(args...)
# Clean up memory/GPU etc...




# ======================================================================================================================
## Print out your results with following format:
# print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test,
#                                                         acc_B_train, acc_B_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A_train = 'TBD'
# acc_B_test = 'TBD'