from benchmark import Benchmark

"""
Benchmark of Machine Learning methodologies on bioactivity data based on the Lenselink, et al. 
(2017) protocol.


This file contains the settings to reproduce the Lenselink, et al. (2017) benchmarking.

Lenselink, E. B., ten Dijke, N., Bongers, B., Papadatos, G., van Vlijmen, H. W. T., Kowalczyk, 
W., IJzerman, A. P., van Westen, G. J. P. (2017). Beyond the hype: Deep neural networks 
outperform established methods using a ChEMBL bioactivity benchmark set. Journal of 
Cheminformatics, 9(45).


For benchmarking using different variables, edit this file to your desired variables.
"""
YOUR_CWD_PATH = ""
YOUR_DATASET_PATH = ""
YOUR_DESC_PATH = "" 

benchmark = Benchmark(cwd = YOUR_CWD_PATH,
                      n_jobs = 1,
                      save_split = False,
                      debug_slice = None)

benchmark.setData(data_path = YOUR_DATASET_PATH, 
                  pred_col = "BIOACT_PCHEMBL_VALUE",
                  smiles_col = "Canonical_Smiles",
                  tgt_col = "TGT_CHEMBL_ID", 
                  smiles_standardizer = None)


runs = {"run1": {"split":"rand", "split_param": {"test_fraction": 0.3}, "algorithm":"NB", "alg_param": {"alpha":1.0, "force_alpha":True}, "th":5.0},
        "run2": {"split":"rand", "split_param": {"test_fraction": 0.3}, "algorithm":"NB", "alg_param": {"alpha":1.0, "force_alpha":True}, "th":6.5},
        "run3": {"split":"rand", "split_param": {"test_fraction": 0.3}, "algorithm":"RF", "alg_param": {"n_estimators":1000, "max_depth":None, "max_features":0.3}, "th":6.5},
        "run4": {"split":"rand", "split_param": {"test_fraction": 0.3}, "algorithm":"SVM", "alg_param": {"kernel":"rbf", "gamma":"auto", "max_iter": -1, "probability":True}, "th":6.5},
        "run5": {"split":"rand", "split_param": {"test_fraction": 0.3}, "algorithm":"LR", "alg_param": {"solver":"sag", "max_iter":100}, "th":6.5},
        "run6": {"split":"temp", "split_param": {"timesplit": 2012, "timeprop": "DOC_YEAR"}, "algorithm":"NB", "alg_param": {"alpha":1.0, "force_alpha":True}, "th":5.0},
        "run7": {"split":"temp", "split_param": {"timesplit": 2012, "timeprop": "DOC_YEAR"}, "algorithm":"NB", "alg_param": {"alpha":1.0, "force_alpha":True}, "th":6.5},
        "run8": {"split":"temp", "split_param": {"timesplit": 2012, "timeprop": "DOC_YEAR"}, "algorithm":"RF", "alg_param": {"n_estimators":1000, "max_depth":None, "max_features":0.3}, "th":6.5},
        "run9": {"split":"temp", "split_param": {"timesplit": 2012, "timeprop": "DOC_YEAR"}, "algorithm":"SVM", "alg_param": {"kernel":"rbf", "gamma":"auto", "max_iter": -1, "probability":True}, "th":6.5},
        "run10": {"split":"temp", "split_param": {"timesplit": 2012, "timeprop": "DOC_YEAR"}, "algorithm":"LR", "alg_param": {"solver":"sag", "max_iter":100}, "th":6.5}}

for run, variables in runs.items():  
    benchmark.setBenchmark(model = "QSAR", 
                            th = variables["th"], 
                            split = variables["split"], 
                            split_param = variables["split_param"], 
                            algorithm = variables["algorithm"], 
                            alg_param = variables["alg_param"], 
                            desc = ["custom"],
                            desc_standardizer = "MinMaxScaler", 
                            desc_path = YOUR_DESC_PATH, 
                            name = None, 
                            check = "strict", 
                            check_param = {"check_n": 30, "check_Ntrain": 5, "check_ntest": 5}, 
                            reps = 1)          

    benchmark.run()