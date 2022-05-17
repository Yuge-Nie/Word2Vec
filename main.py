# -*- coding: utf-8 -*-
import jieba
import os
import re
from gensim.models import word2vec
import numpy as np
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle as pkl
file_w = 'C:/Users/NYG/PycharmProjects/Word2Vec/seg.txt'

def cut_sentences(content):
    end_flag = ['?', '!', '.', '？', '！', '。', '…', '......', '……', '\n']
    content_len = len(content)
    sentences = []
    tmp_char = ''
    for idx, char in enumerate(content):
        tmp_char += char
        if (idx + 1) == content_len:
            sentences.append(tmp_char)
            break
        if char in end_flag:
            next_idx = idx + 1
            if not content[next_idx] in end_flag:
                sentences.append(tmp_char)
                tmp_char = ''
    return sentences


def word_seg(path):
    files = os.listdir(path)
    for file in files:
        position = path + '\\' + file
        with open(position, "r", encoding="ANSI") as f:
            data = f.read()
            f.close()
        text = cut_sentences(data)
        with open(file_w, "a", encoding="utf-8") as f:
            for sentence in text:
                sentence = re.sub('[^\u4e00-\u9fa5]+', '', sentence)
                f.write(" ".join(jieba.lcut(sentence, use_paddle=True, cut_all=False)) + '\n')


def vec_gen(path):
    train_data = word2vec.LineSentence(path)
    model = word2vec.Word2Vec(train_data,
                              vector_size=100,
                              window=5,
                              workers=4)
    model.wv.vectors = model.wv.vectors / (np.linalg.norm(model.wv.vectors, axis=1).reshape(-1, 1))
    vec_dist = dict(zip(model.wv.index_to_key, model.wv.vectors))
    with open('vec_dist', 'wb') as f:
        pkl.dump(vec_dist, f)


def cluster(data):
    with open('vec_dist', 'rb') as f:
        vec_dist = pkl.load(f)
    vec = []
    for d in data:
        vec.append(vec_dist[d])
    center, label, inertia = k_means(vec, n_clusters=4)
    vec = PCA(n_components=2).fit_transform(vec)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(vec[:, 0], vec[:, 1], c=label)
    for i, w in enumerate(data):
        plt.annotate(text=w, xy=(vec[:, 0][i], vec[:, 1][i]),
                     xytext=(vec[:, 0][i] + 0.01, vec[:, 1][i] + 0.01))
    plt.show()


if __name__ == "__main__":
    # s = word_seg('data')
    # wv = vec_gen('seg.txt')
    cluster(['郭靖', '黄蓉', '杨过', '嘉兴', '襄阳', '张无忌', '令狐冲', '桃花岛', '绝情谷', '少林', '武当', '降龙十八掌', '打狗棒法', '一阳指'])