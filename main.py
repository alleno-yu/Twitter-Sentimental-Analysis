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
DATASET_SIZE_B = 10551
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
train_acc_A, val_acc_A, test_acc_A = modelA\
    (vocab_size=vocab_size_A, train_data=train_data_A, val_data=val_data_A, test_data=test_data_A)
# ======================================================================================================================
# Task B
train_acc_B, val_acc_B, test_acc_B = modelB\
    (vocab_size=vocab_size_B, train_data=train_data_B, val_data=val_data_B, test_data=test_data_B)
# ======================================================================================================================
# Print out your results with following format:
print("""TA_training_acc:{:.2f},    TA_test_acc:{:.2f},     TA_val_acc:{:.2f};
TB_training_acc:{:.2f},    TB_test_acc:{:.2f},     TB_val_acc:{:.2f};""".format(train_acc_A, test_acc_A, val_acc_A,
                                                                                 train_acc_B, test_acc_B, val_acc_B))