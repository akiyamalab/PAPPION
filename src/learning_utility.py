import csv
import numpy as np
import os
import glob
import itertools
import sys

def build_tppk_kernel(kernel, split_num, gamma = "auto"):
    def tppk_kernel(feature1, feature2):
        x1, x2 = feature1[:, :split_num], feature1[:, split_num:]
        y1, y2 = feature2[:, :split_num], feature2[:, split_num:]
        return (kernel(x1, y1, gamma = gamma) * kernel(x2, y2, gamma = gamma) + kernel(x1, y2, gamma = gamma) * kernel(x2, y1, gamma = gamma))
    return tppk_kernel

def build_mlpk_kernel(kernel, split_num, gamma = "auto"):
    def mlpk_kernel(feature1, feature2):
        x1, x2 = feature1[:, :split_num], feature1[:, split_num:]
        y1, y2 = feature2[:, :split_num], feature2[:, split_num:]
        return np.square((kernel(x1, y1, gamma = gamma) + kernel(x2, y2, gamma = gamma) - kernel(x1, y2, gamma = gamma) - kernel(x2, y1, gamma = gamma)))
    return mlpk_kernel

def build_pair_tppk_kernel(split_num, gamma_1, gamma_2, kernel):
    def pair_tppk_kernel(feature1, feature2):
        x1, x2 = feature1[:split_num], feature1[split_num:]
        y1, y2 = feature2[:split_num], feature2[split_num:]
        return kernel(x1, y1, gamma = gamma_1) * kernel(x2, y2, gamma = gamma_2)
    return pair_tppk_kernel

def save_matrix(matrix, file_name):
    with open(file_name, "w") as f:
        for row in matrix:
            f.write(",".join(list(map(str, row))) + "\n")

def build_similarity_matrix(protein_list, protein_feature_list, kernel):
    N = len(protein_list)
    similarity_matrix = [[None for __ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            similarity_matrix[i][j] = kernel(protein_feature_list[i], protein_feature_list[j])
            similarity_matrix[j][i] = kernel(protein_feature_list[i], protein_feature_list[j])
    for i in range(N):
        for j in range(N):
            assert similarity_matrix[i][j] is not None,  "feature vector contains None {0} {1}".format(protein_list[j], protein_list[i])
    return similarity_matrix

def build_gram_matrix(protein_list, data_list, similarity_matrix, pair_func):
    protein_index_dct = {}
    for i, p in enumerate(protein_list):
        protein_index_dct[p] = i
    N = len(data_list)
    gram_matrix = [[None for _ in range(N)] for __ in range(N)]
    for i, pair1 in enumerate(data_list):
        x1, x2 = pair1
        i1, i2 = protein_index_dct[x1], protein_index_dct[x2]
        for j, pair2 in enumerate(data_list):
            y1, y2 = pair2
            j1, j2 = protein_index_dct[y1], protein_index_dct[y2]
            s1, s2 = similarity_matrix[i1][j1], similarity_matrix[i2][j2]
            s3, s4 = similarity_matrix[i1][j2], similarity_matrix[i2][j1]
            gram_matrix[i][j] = pair_func(s1, s2, s3, s4)
    for i in range(N):
        for j in range(N):
            assert gram_matrix[i][j] is not None,  "feature vector contains None {0} {1}".format(protein_list[j], protein_list[i])
    return np.array(gram_matrix)

def build_apoptosis_gram_matrix(protein_list, apoptosis_protein_list, data_list, apoptosis_data_list, similarity_matrix, pair_func):
    protein_index_dct = {}
    apoptosis_protein_index_dct = {}
    for i, p in enumerate(protein_list):
        protein_index_dct[p] = i
    for i, p in enumerate(apoptosis_protein_list):
        apoptosis_protein_index_dct[p] = i
    N = len(data_list)
    M = len(apoptosis_data_list)
    gram_matrix = [[None for _ in range(N)] for __ in range(M)]
    print(N, M)
    print(similarity_matrix.shape)
    for i, pair1 in enumerate(apoptosis_data_list):
        x1, x2 = pair1
        i1, i2 = apoptosis_protein_index_dct[x1], apoptosis_protein_index_dct[x2]
        for j, pair2 in enumerate(data_list):
            y1, y2 = pair2
            j1, j2 = protein_index_dct[y1], protein_index_dct[y2]
            s1, s2 = similarity_matrix[i1][j1], similarity_matrix[i2][j2]
            s3, s4 = similarity_matrix[i1][j2], similarity_matrix[i2][j1]
            gram_matrix[i][j] = pair_func(s1, s2, s3, s4)

    for i in range(M):
        for j in range(N):
            assert gram_matrix[i][j] is not None,  "feature vector contains None {0} {1}".format(data_list[j], apoptosis_data_list[i])
    return np.array(gram_matrix)

def tppk_func(s1, s2, s3, s4):
    return s1 * s2 + s3 * s4

def mlpk_func(s1, s2, s3, s4):
    return np.square(s1 + s2 - s3 - s4)

def make_pair_feature(protein_list, data_list, feature_list):
    pair_feature_list = []
    protein_index_dct = {}
    for i, p in enumerate(protein_list):
        protein_index_dct[p] = i
    for p1, p2 in data_list:
        assert p1 in protein_index_dct, "protein_list not contain {0}".format(p1)
        assert p2 in protein_index_dct, "protein_list not contain {0}".format(p2)
        i1 = protein_index_dct[p1]
        i2 = protein_index_dct[p2]
        pair_feature_list.append(np.r_[feature_list[i1], feature_list[i2]])

    return pair_feature_list

"""
def convert_matrix(matrix):
    U, s, V = np.linalg.svd(matrix, full_matrices = False)
    #return np.dot(np.dot(U, np.diag(s)),V)
    s = np.array(list(map(lambda x: 1 + x if x > 0 else 0, list(s))))
    return np.dot(U, np.dot(np.diag(s),V))
"""

def print_data(data_list, label_list):
    protein_set = set()
    print("Number of Protein pair: " + str(len(data_list)), flush = True)
    print("Number of positive: " + str(len(list(filter(lambda x: x == 1, label_list)))), flush = True)
    print("Number of negative: " + str(len(list(filter(lambda x: x == 0, label_list)))), flush = True)

def print_result(mcc_list, f_list, recall_list, precision_list):
    print("Avarage of MCC: " + "{:.3}".format(np.mean(mcc_list)) + " (" + "{:.3}".format(np.std(mcc_list)) + ")", flush = True)
    print("Avarage of F: " + "{:.3}".format(np.mean(f_list)) + " (" + "{:.3}".format(np.std(f_list)) + ")", flush = True)
    print("Average of Racell: " + "{:.3}".format(np.mean(recall_list)) + " (" + "{:.3}".format(np.std(recall_list)) + ")", flush = True)
    print("Average of Precision: " + "{:.3}".format(np.mean(precision_list)) + " (" + "{:.3}".format(np.std(precision_list)) + ")", flush = True)

    print("MCC list: " + ",".join(list(map(lambda x: "{:.3}".format(x), mcc_list))), flush = True)
    print("F list: " + ",".join(list(map(lambda x: "{:.3}".format(x), f_list))), flush = True)
    print("Recall list: " + ",".join(list(map(lambda x: "{:.3}".format(x), recall_list))), flush = True)
    print("Precision list: " + ",".join(list(map(lambda x: "{:.3}".format(x), precision_list))), flush = True)

def print_cutoff(cutoff_list):
    print("Average of cutoff: " + str(np.mean(cutoff_list)), flush = True)
    print("cutoff list: " + ",".join(list(map(str, cutoff_list))), flush = True)

def print_info(y_pred_1, y_pred_2, y_label):
    print("----- print summary of 2 clf prediction", flush = True)

    print("TP of 2 clf: " + str(len(list(filter(lambda x: x == 1, [1 if x == 1 and y == 1 and z == 1 else 0 for x, y, z in zip(y_pred_1, y_pred_2, y_label)])))), flush = True)
    print("TP of clf1: " + str(len(list(filter(lambda x: x == 1, [1 if x == 1 and z == 1 else 0 for x, z in zip(y_pred_1, y_label)])))), flush = True)
    print("TP of clf2: " + str(len(list(filter(lambda x: x == 1, [1 if x == 1 and z == 1 else 0 for x, z in zip(y_pred_2, y_label)])))), flush = True)

    print("FP of 2 clf: " + str(len(list(filter(lambda x: x == 1, [1 if x == 1 and y == 1 and z == 0 else 0 for x, y, z in zip(y_pred_1, y_pred_2, y_label)])))), flush = True)
    print("FP of clf1: " + str(len(list(filter(lambda x: x == 1, [1 if x == 1 and z == 0 else 0 for x, z in zip(y_pred_1, y_label)])))), flush = True)
    print("FP of clf2: " + str(len(list(filter(lambda x: x == 1, [1 if x == 1 and z == 0 else 0 for x, z in zip(y_pred_2, y_label)])))), flush = True)

def output_metrix(file_name, mcc_list, f_list, recall_list, precision_list, best_gamma, best_c):
    with open(file_name, "w") as f:
        f.write("#AUC,F,recall,precision\n")
        s = "{:.3}".format(np.mean(mcc_list))
        s += "," + "{:.3}".format(np.mean(f_list))
        s += "," + "{:.3}".format(np.mean(recall_list))
        s += "," + "{:.3}".format(np.mean(precision_list))
        f.write(s + "\n")
        s = "{:.3}".format(np.std(mcc_list))
        s += "," + "{:.3}".format(np.std(f_list))
        s += "," + "{:.3}".format(np.std(recall_list))
        s += "," + "{:.3}".format(np.std(precision_list))
        f.write(s + "\n")
        s = str(best_gamma) + "," + str(best_c)
        f.write(s + "\n")
        f.write(",".join(list(map(lambda x: "{:.3}".format(x), mcc_list))) + "\n")
        f.write(",".join(list(map(lambda x: "{:.3}".format(x), f_list))) + "\n")
        f.write(",".join(list(map(lambda x: "{:.3}".format(x), recall_list))) + "\n")
        f.write(",".join(list(map(lambda x: "{:.3}".format(x), precision_list))) + "\n")
        

