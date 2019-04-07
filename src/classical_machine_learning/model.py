import io
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import time
import os

gloveDir = "C://Users//hp//emocontext//glove"
ssweDir = "C://Users//hp//emocontext//sswe"
trainDataPath = "C://Users//hp//emocontext//dataset//train.txt"
testDataPath = "C://Users//hp//emocontext//dataset//test.txt"
solutionPath = "C://Users//hp//emocontext//dataset//solution.txt"
featureVectorsPath = "C://Users//hp//emocontext//dataset//features.npy"
featureVectorsPath_2 = "C://Users//hp//emocontext//dataset//features2.npy"

label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}


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

            # conv = ' <eos> '.join(line[1:4])
            conv = ' '.join(line[1:4])

            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)

            indices.append(int(line[0]))
            conversations.append(conv.lower())

    if mode == "train":
        return indices, conversations, labels
    else:
        return indices, conversations


def extract_features(corpus):
    """
    :param corpus: list of conversations (train + test)
    :return: list of feature vectors (a vector for each conversation)
    """
    vectorizer = CountVectorizer()
    count = vectorizer.fit_transform(corpus)

    transformer = TfidfTransformer(smooth_idf=True)
    feature_vectors = transformer.fit_transform(count)

    return feature_vectors


def extract_features_glove(corpus, load=False):
    if load:
        featureVectors = np.load(featureVectorsPath)
        return featureVectors

    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join(gloveDir, 'glove.6B.100d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector

    print('Found %s word vectors.' % len(embeddingsIndex))

    featureVectors = np.zeros((len(corpus), 100))
    tfidf = getTfidf(corpus)

    for s in range(len(corpus)):
        sentence = corpus[s].split(" ")
        sentence2vec = np.zeros(100)
        for word in sentence:
            embeddingVector = embeddingsIndex.get(word)
            # words not found in embedding index will be all-zeros
            if embeddingVector is not None:
                sentence2vec += (tfidf.get(word, 0.0) * embeddingVector)

        featureVectors[s] = sentence2vec

    np.save(featureVectorsPath, featureVectors)

    return featureVectors


def extract_features_sswe(corpus, load=False):
    if load:
        featureVectors = np.load(featureVectorsPath_2)
        return featureVectors

    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join(ssweDir, 'sswe-h.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector

    print('Found %s word vectors.' % len(embeddingsIndex))

    featureVectors = np.zeros((len(corpus), 50))

    for s in range(len(corpus)):
        sentence = corpus[s].split(" ")
        sentence2vec = np.zeros(50)
        for word in sentence:
            embeddingVector = embeddingsIndex.get(word)
            # words not found in embedding index will be all-zeros
            if embeddingVector is not None:
                sentence2vec += embeddingVector

        featureVectors[s] = sentence2vec

    np.save(featureVectorsPath_2, featureVectors)
    return featureVectors


def getTfidf(corpus, load=False):
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(corpus)

    # map words to tf-idf scores
    keys = vectorizer.get_feature_names()
    values = vectorizer.idf_
    tfidf = dict(zip(keys, values))

    return tfidf


def build_model(name="svm"):

    model_1 = svm.SVC(gamma='scale', verbose=True)
    model_2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=7)
    model_3 = LogisticRegression(multi_class="auto", solver="liblinear", random_state=7, verbose=True)
    model = VotingClassifier(estimators=[('lr', model_3), ('gbc', model_2), ('svm', model_1)], voting='hard')

    name2model = {"svm": model_1, "gbc": model_2, "lr": model_3, "ensemble": model}

    return name2model[name]


def main():
    start_time = time.time()

    balanced = False
    indices_train, conversations_train, labels_train = preprocessData(trainDataPath, mode="train")
    if balanced:
        conversations_train = conversations_train[:len(conversations_train)//2]
        labels_train = labels_train[:len(labels_train)//2]
    indices_test, conversations_test = preprocessData(testDataPath, mode="test")

    corpus = conversations_train + conversations_test

    print("Training data size - %d" % len(conversations_train))
    print("Test data size - %d" % len(conversations_test))

    feature_vectors = extract_features_sswe(corpus, load=True)

    x_train = feature_vectors[:len(conversations_train)]
    x_test = feature_vectors[len(conversations_train):]
    print("Number of features - %d" % x_train.shape[1])

    print("Preparing training...")
    y_train = np.array(labels_train)
    model = build_model(name="svm")
    model.fit(x_train, y_train)
    print("Training accuracy - %.2f" % model.score(x_train, y_train))

    print("Preparing testing...")
    predictions = model.predict(x_test)

    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[predictions[lineNum]] + '\n')
    print("Training and testing completed.")

    print("Time taken for training and testing - %.2fs" % (time.time() - start_time))


if __name__ == "__main__":
    main()