# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import io
import json
import os
import re
import sklearn
import sys

# Please use python 3.5 or above
import numpy as np
import tensorflow as tf
from emoji import UNICODE_EMOJI
from keras import optimizers
from keras.backend import argmax
from keras.layers import Dense, Embedding, LSTM, LeakyReLU, Concatenate, Input, Bidirectional, Conv1D, MaxPooling1D, \
    Reshape, MaxPooling2D, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy, binary_accuracy
from keras.models import Sequential, Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from spellchecker import SpellChecker

# Path to training and testing data file. This data can be downloaded from a link, details of which will be provided.
trainDataPath = ""
testDataPath = ""
validDataPath = ""
# Output file that will be generated. This file can be directly submitted.
solutionPath = ""
# Path to directory where GloVe file is saved.
gloveDir = ""
# Path to GloVe embedding matrix
embeddingMatrixGlovePath = ""
# Path to directory where sswe file is saved
ssweDir = ""
# Path to sswe embedding matrix
embeddingMatrixSSWEPath = ""

NUM_FOLDS = None  # Value of K in K-fold Cross Validation
NUM_CLASSES = None  # Number of classes - Happy, Sad, Angry, Others
MAX_NB_WORDS = None  # To set the upper limit on the number of tokens extracted using keras.preprocessing.text.Tokenizer
MAX_SEQUENCE_LENGTH = None  # All sentences having lesser number of words than this will be padded
EMBEDDING_DIM_GLOVE = None  # The dimension of the GloVe word embeddings
EMBEDDING_DIM_SSWE = None  # The dimension of the SSWE word embeddings
BATCH_SIZE = None  # The batch size to be chosen for training the model.
LSTM_DIM = None  # The dimension of the representations learnt by the LSTM model
DROPOUT = None  # Fraction of the units to drop for the linear transformation of the inputs. Ref - https://keras.io/layers/recurrent/
NUM_EPOCHS = None  # Number of epochs to train a model for

label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}

emoji2emoticons = {'😑': ':|', '😖': ':(', '😯': ':o', '😝': ':p', '😐': ':|',
                   '😈': ':)', '🙁': ':(', '😎': ':)', '😞': ':(', '♥️': '<3', '💕': 'love',
                   '😀': ':d', '😢': ":(", '👍': 'ok', '😇': ':)', '😜': ':p',
                   '💙': 'love', '☹️': ':(', '😘': ':)', '🤔': 'hmm', '😲': ':o',
                   '🙂': ':)', '\U0001f923': ':d', '😂': ':d', '👿': ':(', '😛': ':p',
                   '😉': ';)', '🤓': '8-)'}


def clean_word(word):
    """Takes a word, separates emoticons, corrects spelling
    :param word: a string
    :return: list of emoticons, words with correct spellings
    """
    final_list = []
    temp_word = ""
    for k in range(len(word)):
        if word[k] in UNICODE_EMOJI:
            # check if word[k] is an emoji
            if (word[k] in emoji2emoticons):
                final_list.append(emoji2emoticons[word[k]])
        else:
            temp_word = temp_word + word[k]

    if len(temp_word) > 0:
        # "righttttttt" -> "right"
        temp_word = re.sub(r'(.)\1+', r'\1\1', temp_word)
        # correct spelling
        spell = SpellChecker()
        temp_word = spell.correction(temp_word)
        # lemmatize
        temp_word = WordNetLemmatizer().lemmatize(temp_word)
        final_list.append(temp_word)

    return final_list


def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    indices = []
    conversations = []
    labels = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '
                line = cSpace.join(lineSplit)

            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)

            conv = ' <eos> '.join(line[1:4])

            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)

            indices.append(int(line[0]))
            conversations.append(conv.lower())

    if mode == "train":
        return indices, conversations, labels
    else:
        return indices, conversations


def getMetrics(ground, predictions):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = tf.one_hot(predictions, depth=4)
    # ground = tf.constant(ground)#,dtype=tf.float32)
    # if tf.equal(discretePredictions.shape, ground.shape) is True:
    #     print ('All well here')
    truePositives = np.sum((discretePredictions * ground), axis=0)
    falsePositives = np.sum(tf.clip_by_value(discretePredictions - ground, 0.0, 1.0), axis=0)
    falseNegatives = np.sum(tf.clip_by_value(ground - discretePredictions, 0.0, 1.0), axis=0)

    # print("True Positives per class : ", truePositives)
    # print("False Positives per class : ", falsePositives)
    # print("False Negatives per class : ", falseNegatives)

    # # ------------- Macro level calculation ---------------
    # macroPrecision = 0
    # macroRecall = 0
    # # We ignore the "Others" class during the calculation of Precision, Recall and F1
    # for c in range(1, NUM_CLASSES):
    #     precision = truePositives[c] / (truePositives[c] + falsePositives[c])
    #     macroPrecision += precision
    #     recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
    #     macroRecall += recall
    #     if tf.greater(precision+recall,0) is True:
    #         f1 = ( 2 * recall * precision ) / (precision + recall)
    #     else:
    #         f1 = 0
    #     #print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))
    #
    # macroPrecision /= 3
    # macroRecall /= 3
    #
    # if tf.greater(macroPrecision+macroRecall, 0) is True:
    #     macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall)
    # else:
    #     macroF1 = 0
    # #print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))
    #
    # ------------- Micro level calculation ---------------
    truePositives = tf.reduce_sum(truePositives[1:])
    falsePositives = tf.reduce_sum(falsePositives[1:])
    falseNegatives = tf.reduce_sum(falseNegatives[1:])

    # print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))

    microPrecision = tf.divide(truePositives, tf.add_n([truePositives, falsePositives]))
    microRecall = tf.divide(truePositives, tf.add_n([truePositives, falseNegatives]))

    # if tf.greater(tf.add_n([microPrecision, microRecall]), tf.constant(0.0)) is True:
    microF1 = tf.divide(tf.multiply(2.0, tf.multiply(microRecall, microPrecision)),
                        tf.add_n([microPrecision, microRecall]))
    # else:
    #    microF1 = tf.constant(0.0, dtype=tf.float32)
    # -----------------------------------------------------

    # predictions = predictions.argmax(axis=1)
    # ground = ground.argmax(axis=1)
    # accuracy = np.mean(predictions==ground)

    # print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return ("%.2f" % microF1)


def writeNormalisedData(dataFilePath, texts):
    """Write normalised data to a file
    Input:
        dataFilePath : Path to original train/test file that has been processed
        texts : List containing the normalised 3 turn conversations, separated by the <eos> tag.
    """
    normalisedDataFilePath = dataFilePath.replace(".txt", "_normalised.txt")
    with io.open(normalisedDataFilePath, 'w', encoding='utf8') as fout:
        with io.open(dataFilePath, encoding='utf8') as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                line = line.strip().split('\t')
                normalisedLine = texts[lineNum].strip().split('<eos>')
                fout.write(line[0] + '\t')
                # Write the original turn, followed by the normalised version of the same turn
                fout.write(line[1] + '\t' + normalisedLine[0] + '\t')
                fout.write(line[2] + '\t' + normalisedLine[1] + '\t')
                fout.write(line[3] + '\t' + normalisedLine[2] + '\t')
                try:
                    # If label information available (train time)
                    fout.write(line[4] + '\n')
                except:
                    # If label information not available (test time)
                    fout.write('\n')


def getEmbeddingMatrix_glove(wordIndex, load=False):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
        load: If set to True, then loads the embedding matrix from a file
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    if load:
        embeddingMatrix = np.load(embeddingMatrixGlovePath)
        return embeddingMatrix

    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join(gloveDir, 'glove.6B.100d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector

    print('Found %s word vectors.' % len(embeddingsIndex))

    bad_words_glove_1 = set([])
    bad_words_glove_2 = set([])
    counter = 0

    # Minimum word index of any word is 1. 
    embeddingMatrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM_GLOVE))
    for word, i in wordIndex.items():
        # print(".",)
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector
        else:
            bad_words_glove_1.add(word)
            good_from_bad = clean_word(word)
            # sum all word vectors, obtained after clean_word
            temp_embedding_vector = np.zeros((1, EMBEDDING_DIM_GLOVE))
            for www in good_from_bad:
                eee = embeddingsIndex.get(www)
                if eee is not None:
                    temp_embedding_vector = temp_embedding_vector + eee
            if not temp_embedding_vector.all():
                bad_words_glove_2.add(word)
            embeddingMatrix[i] = temp_embedding_vector
        if (counter % 1000 == 0):
            print(counter)
        counter += 1

    print("Bad words in GloVe 1 - %d" % len(bad_words_glove_1))
    print("Bad words in GloVe 2 - %d" % len(bad_words_glove_2))

    #     np.save(embeddingMatrixGlovePath, embeddingMatrix)

    return embeddingMatrix


def getEmbeddingMatrix_sswe(wordIndex, load=False):
    """
    :param wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    :param load: If set to True, then loads the embedding matrix from a file
    :return: embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    if load:
        embeddingMatrix = np.load(embeddingMatrixSSWEPath)
        return embeddingMatrix

    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join(ssweDir, 'sswe-u.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector

    print('Found %s word vectors.' % len(embeddingsIndex))

    bad_words_sswe_1 = set([])
    bad_words_sswe_2 = set([])
    counter = 0

    # Minimum word index of any word is 1.
    embeddingMatrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM_SSWE))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector
        else:
            bad_words_sswe_1.add(word)
            good_from_bad = clean_word(word)
            # sum all word vectors, obtained after clean_word
            temp_embedding_vector = np.zeros((1, EMBEDDING_DIM_SSWE))
            for www in good_from_bad:
                eee = embeddingsIndex.get(www)
                if eee is not None:
                    temp_embedding_vector = temp_embedding_vector + eee
            if not temp_embedding_vector.all():
                bad_words_sswe_2.add(word)
            embeddingMatrix[i] = temp_embedding_vector
        if (counter % 1000 == 0):
            print(counter)
        counter += 1

    print("Bad words in SSWE 1 - %d" % len(bad_words_sswe_1))
    print("Bad words in SSWE 2 - %d" % len(bad_words_sswe_2))

    #     np.save(embeddingMatrixSSWEPath, embeddingMatrix)

    return embeddingMatrix


def buildModel(embeddingMatrix1, embeddingMatrix2):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    main_input_1 = Input(shape=(100,), name='main_input_1')
    embeddingLayer1 = Embedding(embeddingMatrix1.shape[0],
                                EMBEDDING_DIM_GLOVE,
                                weights=[embeddingMatrix1],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)(main_input_1)
    model1 = LSTM(LSTM_DIM, dropout=DROPOUT, return_sequences=True)(embeddingLayer1)
    model1 = LSTM(LSTM_DIM, dropout=DROPOUT)(model1)

    main_input_2 = Input(shape=(100,), name='main_input_2')
    embeddingLayer2 = Embedding(embeddingMatrix2.shape[0],
                                EMBEDDING_DIM_SSWE,
                                weights=[embeddingMatrix2],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)(main_input_2)
    model2 = LSTM(LSTM_DIM, dropout=DROPOUT, return_sequences=True)(embeddingLayer2)
    model2 = LSTM(LSTM_DIM, dropout=DROPOUT)(model2)

    model3 = Concatenate()([model1, model2])
    model3 = LeakyReLU(alpha=0.1)(model3)
    model3 = Dense(NUM_CLASSES, activation="softmax")(model3)
    model = Model([main_input_1, main_input_2], model3)
    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=[categorical_accuracy])
    model.summary()

    return model


def cnnModel(embeddingMatrix1):
    """Constructs the architecture of the model
    Input:
    embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
    model : A basic CNN model
    """

    main_input_1 = Input(shape=(100,), name='main_input_1')
    embeddingLayer1 = Embedding(embeddingMatrix1.shape[0],
                                EMBEDDING_DIM_GLOVE,
                                weights=[embeddingMatrix1],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)(main_input_1)
    embeddingLayer1 = Reshape((100, EMBEDDING_DIM_GLOVE, 1))(embeddingLayer1)
    model1 = Conv2D(NUM_FILTERS, kernel_size=(3, EMBEDDING_DIM_GLOVE), padding='same', activation='relu')(
        embeddingLayer1)
    model1 = Conv2D(NUM_FILTERS, kernel_size=(4, EMBEDDING_DIM_GLOVE), padding='same', activation='relu')(model1)
    model1 = Conv2D(NUM_FILTERS, kernel_size=(5, EMBEDDING_DIM_GLOVE), padding='same', activation='relu')(model1)
    model1 = MaxPooling2D(EMBEDDING_DIM_GLOVE)(model1)
    model1 = Flatten()(model1)
    model1 = Dense(NUM_CLASSES, activation="softmax")(model1)
    model = Model([main_input_1], model1)
    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=[categorical_accuracy])
    model.summary()

    return model


def CNNModel(embeddingMatrix1):
    main_input = Input(shape=(100,), name='main_input')
    embeddingLayer = Embedding(embeddingMatrix1.shape[0],
                               EMBEDDING_DIM_GLOVE,
                               weights=[embeddingMatrix1],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)(main_input)

    conv_blocks = []
    filter_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    for fs in filter_sizes:
        conv = Conv1D(filters=NUM_FILTERS,
                      kernel_size=fs,
                      padding='valid',
                      strides=1,
                      activation='relu',
                      use_bias=True)(embeddingLayer)
        conv = Conv1D(filters=NUM_FILTERS,
                      kernel_size=fs,
                      padding='valid',
                      strides=1,
                      activation='relu',
                      use_bias=True)(conv)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)

    concat1max = Concatenate()(conv_blocks)
    concat1max = Dropout(DROPOUT)(concat1max)

    output_layer = Dense(50, activation='relu')(concat1max)
    output_layer = Dense(4, activation='softmax')(output_layer)

    model = Model(inputs=main_input, outputs=output_layer)
    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=[categorical_accuracy])
    print(model.summary())

    return model


def microf1(ytrue, ypred):
    return sklearn.metrics.f1_score(ytrue, ypred, average='micro', labels=[1, 2, 3])


def all_metrics(ytrue, ypred):
    return sklearn.metrics.precision_recall_fscore_support(ytrue, ypred)


def helper(a):
    for i in range(len(a)):
        if a[i] == 1:
            return i


def main():
    parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
    parser.add_argument('-config', help='Config to read details', required=True)
    args = parser.parse_args()

    with open(args.config) as configfile:
        config = json.load(configfile)

    global trainDataPath, testDataPath, validDataPath, solutionPath, gloveDir, embeddingMatrixGlovePath, ssweDir
    global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM_GLOVE, EMBEDDING_DIM_SSWE
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE, embeddingMatrixSSWEPath, NUM_FILTERS

    trainDataPath = config["train_data_path"]
    testDataPath = config["test_data_path"]
    validDataPath = config["valid_data_path"]
    solutionPath = config["solution_path"]
    gloveDir = config["glove_dir"]
    embeddingMatrixGlovePath = config["embedding_matrix_glove_path"]
    ssweDir = config["sswe_dir"]
    embeddingMatrixSSWEPath = config["embedding_matrix_sswe_path"]

    NUM_FOLDS = config["num_folds"]
    NUM_CLASSES = config["num_classes"]
    MAX_NB_WORDS = config["max_nb_words"]
    MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
    EMBEDDING_DIM_GLOVE = config["embedding_dim_glove"]
    EMBEDDING_DIM_SSWE = config["embedding_dim_sswe"]
    BATCH_SIZE = config["batch_size"]
    LSTM_DIM = config["lstm_dim"]

    DROPOUT = config["dropout"]
    LEARNING_RATE = config["learning_rate"]
    NUM_EPOCHS = config["num_epochs"]
    NUM_FILTERS = config["num_filters"]

    print("Processing training data...")
    new_trainText = []
    tweet_tokenizer = TweetTokenizer()
    trainIndices, trainTexts, labels = preprocessData(trainDataPath, mode="train")
    for i in range(len(trainTexts)):
        tokens = tweet_tokenizer.tokenize(trainTexts[i])
        sent = ' '.join(tokens)
        new_trainText.append(sent)
    print('Size of trainText = ', len(trainTexts))
    print('Size of new_trainText = ', len(new_trainText))

    print("\nProcessing Validation data...")
    new_validText = []
    validIndices, validTexts, validlabels = preprocessData(validDataPath, mode="train")
    for i in range(len(validTexts)):
        tokens = tweet_tokenizer.tokenize(validTexts[i])
        sent = ' '.join(tokens)
        new_validText.append(sent)
    print('Size of validText = ', len(validTexts))
    print('Size of new_validText = ', len(new_validText))

    print("\nProcessing Test data...")
    new_testText = []
    testIndices, testTexts, testlabels = preprocessData(testDataPath, mode="train")
    for i in range(len(testTexts)):
        tokens = tweet_tokenizer.tokenize(testTexts[i])
        sent = ' '.join(tokens)
        new_testText.append(sent)
    print('Size of testText = ', len(testTexts))
    print('Size of new_testText = ', len(new_testText))

    # Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable
    # writeNormalisedData(trainDataPath, trainTexts)

    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(new_trainText)
    trainSequences = tokenizer.texts_to_sequences(new_trainText)
    validSequences = tokenizer.texts_to_sequences(new_validText)
    testSequences = tokenizer.texts_to_sequences(new_testText)

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    print("Populating embedding matrix...")
    embeddingMatrix_glove = getEmbeddingMatrix_glove(wordIndex, True)
    embeddingMatrix_sswe = getEmbeddingMatrix_sswe(wordIndex, True)

    data = pad_sequences(trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    print("Shape of training data tensor: ", data.shape)
    print("Shape of label tensor: ", labels.shape)

    data_valid = pad_sequences(validSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels_valid = to_categorical(np.asarray(validlabels))
    print("Shape of Validation data tensor: ", data_valid.shape)
    print("Shape of validation label tensor: ", labels_valid.shape)

    data_test = pad_sequences(testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels_test = to_categorical(np.asarray(testlabels))
    print("Shape of Validation data tensor: ", data_test.shape)
    print("Shape of validation label tensor: ", labels_test.shape)

    # Randomize Training data
    np.random.shuffle(trainIndices)
    data = data[trainIndices]
    labels = labels[trainIndices]

    # Randomize Validation data
    np.random.shuffle(validIndices)
    data_valid = data_valid[validIndices]
    labels_valid = labels_valid[validIndices]

    print("Retraining model on entire data to create solution file")
    model = CNNModel(embeddingMatrix_glove)

    model.fit([data], labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    predictions = model.predict([data_valid])
    predictions = predictions.argmax(axis=1)

    final_predictions = model.predict([data_test])
    final_predictions = final_predictions.argmax(axis=1)

    f1score_dev = microf1(labels_valid.argmax(axis=1), predictions)

    f1score_test = microf1(labels_test.argmax(axis=1), final_predictions)

    all_metrics_dev = all_metrics(labels_valid.argmax(axis=1), predictions)

    all_metrics_test = all_metrics(labels_test.argmax(axis=1), final_predictions)

    result = "DROPOUT-%.2f, NUM_FILTERS-%d, LEARNING_RATE-%.4f, NUM_EPOCHS-%d, F1SCORE_DEV-%.2f, F1SCORE_TEST-%0.2f\n" % (
    DROPOUT, NUM_FILTERS, LEARNING_RATE, NUM_EPOCHS, f1score_dev, f1score_test)
    print(result)
    print(all_metrics_dev)
    print(all_metrics_test)


if __name__ == '__main__':
    main()
