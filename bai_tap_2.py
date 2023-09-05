import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def read_embedding(path):
    words = {}
    with open(path, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines[2:]):
        line = line[:-1].strip().split(" ")  # remove "\n" last line

        line.remove("")
        words[line[0]] = np.array(list(map(float, line[1:])))
    return words


def read_viSim400(path):

    datasets = []
    with open(path) as f:
        lines = f.readlines()

    for line in tqdm(lines[1:]):
        datasets.append(line[:-1].split("\t"))
    return datasets


def calculate_cos_sim(w1, w2):
    embed1 = words[w1]
    embed2 = words[w2]
    return cosine_similarity([embed1], [embed2])[0][0]


def KNN(w1, k=10):
    assert w1 in list(words.keys())
    KNN = sorted(
        list(words.keys()), key=lambda w2: calculate_cos_sim(w1, w2), reverse=True
    )
    for w2 in KNN[:k]:
        print(w2,f"{calculate_cos_sim(w1,w2):.04f}")
    return KNN[:k]


words = read_embedding("word2vec/W2V_150.txt")
datasets = read_viSim400("datasets/ViSim-400/Visim-400.txt")

# home work 2
# print(KNN("thú_vị"))
# return ['thú_vị', 'lý_thú', 'mới_mẻ', 'hữu_ích', 'điều_thú_vị', 'tuyệt_vời', 'kỳ_lạ', 'hữu_dụng', 'đáng_nhớ', 'lạ_mắt']

print(KNN("nguyên_nhân"))
