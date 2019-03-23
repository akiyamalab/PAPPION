# coding: utf-8

import glob
import dill
import argparse
import csv
import os
import numpy as np
from joblib import Parallel, delayed
import subprocess
from sklearn.preprocessing import StandardScaler

INF = 20


def load_scaler(file_name):
    model = None
    with open(file_name, "rb") as f:
        model = dill.load(f)
    assert model is not None, "Fail to load Scaler"
    return model

def run_mammoth(pdb_file, protein_list, out_dir, mammoth_path, sc):
    rec_name = os.path.basename(pdb_file).split(".")[0]
    tmp_out = out_dir + rec_name + "_.txt"
    out_file = out_dir + rec_name + ".npy"
    result_vector = []

    for i, lig_file in enumerate(protein_list):
        lig_name = os.path.basename(lig_file).split(".")[0]
        subprocess.call([mammoth_path, "-e", pdb_file, "-p", lig_file, "-o", tmp_out])
        s = 0
        e_val = 0
        if not os.path.exists(tmp_out):
            print("Empty Out: " + rec_name + "," + lig_name + "\n", flush = True)
            return None
        with open(tmp_out, "r") as f:
            for line in f:
                if line.strip().find("-ln(E)") != -1:
                    array = list(filter(lambda x: x != "", line.strip().split(" ")))
                    if array[0] == "Infinity":
                        s = INF
                    else: 
                        s = float(array[3]) if float(array[3]) < INF else INF
        subprocess.call(["rm", tmp_out])
        result_vector.append(s)
    result_vector = np.array([result_vector])
    result_vector = sc.transform(result_vector)
    #print(result_vector[0])
    np.save(out_file, result_vector[0])
    print("Finish calculating vector " + pdb_file, flush = True)
    return result_vector

def make_single_vector(input_file, output_dir, cpu_num, mammoth_path):
    protein_pair_list = load_protein_pair(input_file)
    protein_set = set()
    for p1, p2 in protein_pair_list:
        protein_set.add(p1)
        protein_set.add(p2)
    protein_list = list(protein_set)
    rec_list = glob.glob("rep_str/P_1_550/*.pdb")
    sc = load_scaler("models/scaler_1_550.pickle")
    r = Parallel(n_jobs = cpu_num)([delayed(run_mammoth)(p, rec_list, output_dir, mammoth_path, sc) for p in protein_list])

    for i, vec in enumerate(r):
        assert vec is not None, "cannot create vector " + protein_list[i]

def make_stacking_vector(input_file, output_dir, cpu_num, mammoth_path):
    pass

def main():
    # parse argment
    parser = argparse.ArgumentParser(description="making PAPPION vector script")
    parser.add_argument("input_file", help="csv file for PDB file path of target protein pair")
    parser.add_argument("output_dir", help="output directory")
    parser.add_argument("--stacking", action="store_true", help="use stacking model")
    parser.add_argument("--cpunum", default="1", type=int, help="number of CPU, default=1")
    args = parser.parse_args()

    if "MAMMOTH_PATH" not in os.environ:
        print("Please set MAMMOTH_PATH", flush = True)
        sys.exit()
    mammoth_path = os.environ["MAMMOTH_PATH"]
    
    if args.stacking:
        make_stacking_vector(args.input_file, args.output_dir, args.cpunum, mammoth_path)
    else:
        make_single_vector(args.input_file, args.output_dir, args.cpunum, mammoth_path)

if __name__ == "__main__":
    main()
