import learning_utility
import dill
import argparse
import csv
import os
import numpy as np
from sklearn.svm import SVC

def load_protein_pair(input_file):
    protein_pair = []
    with open(input_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "\n":
                continue
            assert len(row) == 2, "invalid input file, no 2 column"
            protein_pair.append((row[0], row[1].strip()))
    return protein_pair

def load_vector(base_dir, protein_pair):
    protein_to_vec = {}
    for p1, p2 in protein_pair:
        print(base_dir)
        print(p1, p2)
        vec_path_1 = base_dir + os.path.splitext(os.path.basename(p1))[0] + ".npy"
        if p1 not in protein_to_vec:
            protein_to_vec[p1] = np.load(vec_path_1)
        vec_path_2 = base_dir + os.path.splitext(os.path.basename(p2))[0] + ".npy"
        if p2 not in protein_to_vec:
            protein_to_vec[p2] = np.load(vec_path_2)
    return protein_to_vec

def make_pair_vec(protein_to_vec, protein_pair_list):
    vec_list = [None for _ in range(len(protein_pair_list))]
    c = 0
    for p1, p2 in protein_pair_list:
        vec_list[c] = np.r_[protein_to_vec[p1], protein_to_vec[p2]]
        c += 1
    return vec_list

def load_model(file_name):
    model = None
    with open(file_name, "rb") as f:
        model = dill.load(f)
    assert model is not None, "Fail to load model" + file_name
    return model

def predict_single_model(input_file, vec_dir, output_file):
    protein_pair_list = load_protein_pair(input_file)
    protein_to_vec = load_vector(vec_dir, protein_pair_list)
    protein_pair_vec = make_pair_vec(protein_to_vec, protein_pair_list)

    clf = load_model("models/model_1_550.pickle")
    predict_list = clf.predict_proba(protein_pair_vec)
    predict_list = list(map(lambda x: x[1], predict_list))
    
    with open(output_file, "w") as f:
        for pair, proba in zip(protein_pair_list, predict_list):
            f.write(",".join([pair[0], pair[1], str(proba)]) + "\n")

def predict_stacking_model(input_file, vec_dir, output_file):
    pass

def main():
    # parse argment
    parser = argparse.ArgumentParser(description="PPI prediction program based on PAPPION vector")
    parser.add_argument("input_file", help="csv file for PDB file path of target protein pair")
    parser.add_argument("vec_dir", help="PAPPION vector directory (output directory of make_vactor.py)")
    parser.add_argument("-o", "--out", default="result.csv", help="output file name, default=result.csv")
    #parser.add_argument("-o", "--out")
    parser.add_argument("--stacking", action="store_true", help="use stacking model")
    args = parser.parse_args() 

    if args.stacking:
        predict_stacking_model(args.input_file, args.vec_dir, args.out)
    else:
        predict_single_model(args.input_file, args.vec_dir, args.out)
    
    #mammoth_path = os.environ["MAMMOTH_PATH"]

if __name__ == "__main__":
    main()

