from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import numpy as np
from tqdm import tqdm


def load_trainset(embeddings):
    x_train, y_train = [], []
    with open("antonym-synonym set/Antonym_vietnamese.txt", "r", encoding="utf8") as f:
        ant_pairs = f.readlines()
    with open("antonym-synonym set/Synonym_vietnamese.txt", "r", encoding="utf8") as f:
        syn_pairs = f.readlines()

    # -----------------------------------------------------------
    # antonym data, y=0
    for pair in ant_pairs:
        words = pair.split()
        u1 = words[0].strip()
        u2 = words[1].strip()
        if not (u1 in embeddings) or not (u2 in embeddings):
            continue
        v1 = embeddings[u1]
        v2 = embeddings[u2]
        x_train.append(v1 + v2)
        y_train.append(0)

    # -----------------------------------------------------------
    # synomyn data, y=1
    for pair in syn_pairs:
        words = pair.split()
        u1 = words[0].strip()
        try:
            u2 = words[1].strip()  # có dòng chỉ có 1 từ
        except:
            continue
        else:
            u2 = u2

        if not (u1 in embeddings) or not (u2 in embeddings):
            continue
        v1 = embeddings[u1]
        v2 = embeddings[u2]
        x_train.append(v1 + v2)
        y_train.append(1)

    return x_train, y_train


def load_testset(embeddings):
    x_test, y_test = [], []
    with open("datasets/ViCon-400/400_noun_pairs.txt", "r", encoding="utf8") as f:
        noun_pairs = f.readlines()
    with open("datasets/ViCon-400/400_verb_pairs.txt", "r", encoding="utf8") as f:
        verb_pairs = f.readlines()
    with open("datasets/ViCon-400/600_adj_pairs.txt", "r", encoding="utf8") as f:
        adj_pairs = f.readlines()
    testset = noun_pairs[1:] + verb_pairs[1:] + adj_pairs[1:]

    for pair in testset:
        words = pair.split()
        u1 = words[0].strip()
        u2 = words[1].strip()
        if not (u1 in embeddings) or not (u2 in embeddings):
            continue
        v1 = embeddings[u1]
        v2 = embeddings[u2]
        x_test.append(v1 + v2)
        if words[2] == "ANT":
            y_test.append(0)
        else:
            y_test.append(1)

    return x_test, y_test


# load embeddings
def read_embedding(path):
    words = {}
    with open(path, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines[2:]):
        line = line[:-1].strip().split(" ")  # remove "\n" last line

        line.remove("")
        words[line[0]] = np.array(list(map(float, line[1:])))
    return words


def main():
    # load embeddings, trainset, testset
    embeddings = read_embedding("word2vec/W2V_150.txt")
    X_train, y_train = load_trainset(embeddings)
    X_test, y_test = load_testset(embeddings)

    # print(f"shape train dataset: {len(X_train)}")
    # print(f"shape train dataset: {len(X_test)}")
    # logistic regression train
    print("Train with logistic regression ")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test
    pred = model.predict(X_test)
    print("Precision score:", precision_score(y_test, pred))
    print("Recall score:", recall_score(y_test, pred))
    print("F1 score:", f1_score(y_test, pred))
    print("Accuracy:", accuracy_score(y_test, pred))

    print()
    print("Train with MLP")
    clf = MLPClassifier(hidden_layer_sizes=(500)).fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("Precision score:", precision_score(y_test, pred))
    print("Recall score:", recall_score(y_test, pred))
    print("F1 score:", f1_score(y_test, pred))
    print("Accuracy:", accuracy_score(y_test, pred))


    Y_hat = clf.predict(X_test)

    # print(clf.score(X_test, y_test))
    print(classification_report(y_test, Y_hat, target_names=['synonym','antonym'], digits=4))

    # Y_score = clf.predict_proba(X_test)

    # print(np.where(y_test!=Y_hat))
    # y_test = np.array(y_test)
    # print(y_test[np.where(y_test!=Y_hat)[0]])
    # print(Y_hat[np.where(y_test!=Y_hat)[0]])
    # print(Y_score[np.where(y_test!=Y_hat)[0]])
    # for wword in X_test[np.where(y_test!=Y_hat)[0]]:
    #     wlen = wword.shape[0]//2
    #     embed_w1, embed_w2 = wword[:wlen],wword[wlen:]
    #     for key,value in embeddings.items():
    #         if np.all(value==embed_w1): print(key)
    #         if np.all(value==embed_w2): print(key)


if __name__ == "__main__":
    main()
