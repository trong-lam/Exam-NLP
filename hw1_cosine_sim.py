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


# tính cosin similarity
# xuất hiện từ trong visim-400 không nằm trong W2V_150 (cõi_tục, ...)
def cos_sim():
    list_word1 = []
    list_word2 = []
    cos_sims = []
    for word1, word2, pos, sim1, sim2, std in datasets:
        if word1 in words and word2 in words:
            list_word1.append(word1)
            list_word2.append(word2)

            cos_sim = calculate_cos_sim(word1, word2)
            if cos_sim > 0.8:
                print(f"Hight cos sim: {word1} {word2} {cos_sim}")
            elif cos_sim >= 0.5 and cos_sim <=0.6:
                print(f"Median cos sim: {word1} {word2} {cos_sim}")
            elif cos_sim <= 0.1:
                print(f"Low cos sim: {word1} {word2} {cos_sim}")
            # cover [-1, 1] to [0,4]
            cos_sims.append((cos_sim + 1) * 2)

            # print(word1, word2, sim1, (cos_sim + 1) * 2)
    # export ra file txt
    with open("cosine_similarity_ViSim-400.txt", "w", encoding="utf8") as f:
        for i in range(len(list_word1)):
            f.write(list_word1[i] + " " + list_word2[i] + " " + str(cos_sims[i]) + "\n")


words = read_embedding("word2vec/W2V_150.txt")
datasets = read_viSim400("datasets/ViSim-400/Visim-400.txt")

# Home work 1
cos_sim()
