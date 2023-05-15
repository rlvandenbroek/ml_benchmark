import os, random, sys, time
import numpy as np
import pandas as pd

from qsprpred.data.data import QSPRDataset
from qsprpred.data.utils.datasplitters import randomsplit, temporalsplit
from qsprpred.data.utils.descriptorcalculator import DescriptorsCalculator
from qsprpred.data.utils.descriptorsets import CustomCompoundDescriptor
from qsprpred.models.tasks import TargetTasks
from qsprpred.models.models import QSPRsklearn

from rdkit.ML.Scoring.Scoring import CalcAUC, CalcBEDROC

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


class Benchmark():
    """
    Benchmark of Machine Learning methodologies on bioactivity data based on the Lenselink, et al. 
    (2017) protocol.

    Lenselink, E. B., ten Dijke, N., Bongers, B., Papadatos, G., van Vlijmen, H. W. T., Kowalczyk, 
    W., IJzerman, A. P., van Westen, G. J. P. (2017). Beyond the hype: Deep neural networks 
    outperform established methods using a ChEMBL bioactivity benchmark set. Journal of 
    Cheminformatics, 9(45).
    """
    def __init__(self, cwd=os.getcwd, n_jobs=1, save_split=False, debug_slice=None):
        """      
        Args: 
            cwd (str) : PATH to working directory
                Default: os.getcwd()
            n_jobs (int) : Set number of CPUs
                Default: 1
            save_split (bool, optional) : Save test and training sets with descriptors
            debug_slice (int, optional) : Create slice of targets for debugging runs
        """
        self.cwd = cwd
        self.n_jobs = n_jobs
        self.save_split = save_split
        self.debug_slice = debug_slice

        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)
        os.chdir(self.cwd)   

        self.setDataCheck = False
        self.setBenchmarkCheck = False

    def setData(self, data_path, pred_col, smiles_col, tgt_col=None, smiles_standardizer=None):
        """
        Retrieve dataset and specify dataset-specific variables.

        Args:
            data_path (str) : PATH to bioactivity dataset
            pred_col (str) : Column name of property to be predicted
            smiles_col (str) : Column name containing canonical SMILES
            tgt_col (str, optional) : Column name of target identifiers for QSAR models
                Required if setBenchmark(model="QSAR")
            smiles_standardizer (str) : Select smiles standardizer:
                Options: None
        """
        self.data_path = data_path
        self.pred_col = pred_col
        self.smiles_col = smiles_col
        self.tgt_col = tgt_col
        self.smiles_standardizer = smiles_standardizer

        if not os.path.exists(self.data_path):
            sys.exit(f"Error: data_loc ({self.data_path}) does not exist")
        self.data = pd.read_csv(self.data_path, sep=",")
             
        if self.tgt_col:
          self.tgt_total = self.data[self.tgt_col].nunique()

        self.class_col = self.pred_col + "_class" 
        self.label_col = self.class_col + "_Label"
        self.probN_col = self.class_col + "_ProbabilityClass_1"

        self.setDataCheck = True

    def setBenchmark(self, model, th, split, split_param, algorithm, alg_param, desc,  
                     desc_standardizer=None, desc_path=None, name=None, check="strict", 
                     check_param={"check_n":30,"check_Ntrain":5,"check_ntest":5}, reps=1):
        """" 
        Set all Machine Learning variables for benchmarking.

        Args:
            model (str) : Select ML method:
                Options: "QSAR"
            th (float) : Set active/inactive threshold
            split (str) : Select datasplit method:
                Options: "rand" - Random datasplit
                         "temp" - Temporal datasplit
            split_param (dict) : Set datasplit parameters as used in QSPRPred.
                if split == "rand":
                    {test_fraction (float): fraction of total dataset to testset}
                if split == "temp":
                    {timesplit(float): time point after which sample to test set,
                     timeprop (str): column name of column in df that contains the timepoints}
            algorithm (str) : Select ML algorithm:
                Options: "LR" - Logistic Regression
                         "NB" - Naive Bayes      
                         "RF" - Random Forest
                         "SVM" - Support Vector Machine
                         "random" - Random predictions (Baseline metrics)
            alg_param (dict) : Set algorithm parameters. Each algorithm has their own parameters,
                               so look at their code to find the parameters. Example:
                if algorithm == "RF":
                    {"n_estimators":1000,
                    "max_depth":None,
                    "max_features":0.3}
            desc (list) : Select (multiple) descriptors:
                Options: "custom"
            desc_standardizer (str) : Select descriptor standardizer:
                Options: None -> Might not work for NB
                        "MinMaxScaler"
            desc_path (str, optional) : PATH to descriptor set 
                Required if setBenchmark(desc = "custom")
            check (str) : Set strictness of checks
                Options: "base" - Bare minimun requirements to run models
                         "strict" - Require set number of datapoints/actives/inactives to train model
            check_param (dict) : Set check parameters 
                check_n (int) : Set minimum number of datapoints
                check_Ntrain (int) : Set minumun number of actives and inactives in training set
                check_ntest (int) : Set minumum number of datapoints in test set             
                Default: {"check_n":30,
                          "check_Ntrain":5,
                          "check_ntest":5}   
            name (str, optional) : Base name for saving results
                Default: self.split + "_" + self.algorithm + "_" + str(self.th)
            reps (int) : Set number of replicates
                Default: 1
        """       
        self.desc = desc
        self.desc_standardizer = desc_standardizer
        self.desc_path = desc_path
        self.model = model
        self.th = th
        self.split = split
        self.split_param = split_param
        self.algorithm = algorithm
        self.alg_param = alg_param
        self.check = check
        self.check_param = check_param       
        self.name = name
        self.reps = reps

        if self.desc == "custom":
            if not os.path.exists(self.desc_path):
                sys.exit(f"Error: desc_loc ({self.desc_path}) does not exist")

        if not self.name:
            self.name = self.split + "_" + self.algorithm + "_" + str(self.th) 

        self.setAlgorithm()
        self.setFeatureCalculator()
        self.setFeatureStandardizer()
        self.setSplitter()

        self.setBenchmarkCheck = True

    def run(self):                        
        """
        Execute benchmarking script for the selected model.
        IMPORTANT: setData() and setBenchmark() must have been executed prior to run()
        """
        if self.setDataCheck == False:
            sys.exit("setData() has not been executed prior to run()")
        if self.setBenchmarkCheck == False:
            sys.exit("setBenchmark() has not been executed prior to run()")
        
        # Create replicates
        for rep in range(self.reps):
            self.rep_name = self.name + "_" + str(rep)
    
            # Modelling
            start_time = time.time() 
            if self.model == "QSAR":
                metrics_Global, metrics_PerTarget, skip_count = self.modelQSAR()
            else:
                sys.exit(f'Error: {self.model} not in model options ["QSAR"]')
            runtime = time.time() - start_time

            # Calculate general model-wide metrics
            metrics_merged = pd.concat([metrics_Global, metrics_PerTarget])
            metrics_Mean = metrics_merged[["BEDROC", "AUC", "MCC", "Precision", "Recall"]].mean()

            results_global = pd.DataFrame({"NAME": self.rep_name,
                                           "RUNTIME": runtime, 
                                           "N_TGT": self.tgt_total,
                                           "N_SKIP": skip_count,
                                           "MEAN_BEDROC": metrics_Mean["BEDROC"], 
                                           "MEAN_AUC": metrics_Mean["AUC"], 
                                           "MEAN_MCC": metrics_Mean["MCC"],
                                           "MEAN_Precision": metrics_Mean["Precision"],        
                                           "MEAN_Recall": metrics_Mean["Recall"],                                    
                                           "GLOBAL_BEDROC": metrics_Global["BEDROC"], 
                                           "GLOBAL_AUC": metrics_Global["AUC"], 
                                           "GLOBAL_MCC": metrics_Global["MCC"],
                                           "GLOBAL_Precision": metrics_Global["Precision"],
                                           "GLOBAL_Recall": metrics_Global["Recall"],
                                           "AVG_BEDROC": metrics_PerTarget["BEDROC"], 
                                           "AVG_AUC": metrics_PerTarget["AUC"], 
                                           "AVG_MCC": metrics_PerTarget["MCC"],
                                           "AVG_Precision": metrics_PerTarget["Precision"],                                    
                                           "AVG_Recall": metrics_PerTarget["Recall"]}, index=[0])
            print(f"\n{results_global}")    

            if not os.path.exists("./results/summary.csv"):
                results_global.to_csv("./results/summary.csv", index=False, header=True)
            else:
                results_global.to_csv("./results/summary.csv", mode="a", index=False, header=False)

        return results_global

    def modelQSAR(self):
        """
        Perform QSAR modelling for each target within the dataset.
        """    
        results_pool = pred_pool = metrics_Global = pd.DataFrame()
        tgt_count = skip_count = 0

        # Create models for each target
        for tgt, df_tgt in self.data.groupby(self.tgt_col):     
            dataset_tgt = None
            tgt_count += 1

            if self.debug_slice:
                if tgt_count > self.debug_slice:
                    continue

            model_name = tgt + "_" + self.rep_name
            os.makedirs("./"+self.rep_name+"/qspr/models/"+model_name, exist_ok=True)

            print(f"\n{model_name} ({tgt_count}/{self.tgt_total}) - {len(df_tgt)} data points")

            # Setup dataset using QSPRPred
            skip, error_msg = self.checkData(df_tgt)            
            if not skip:
                dataset_tgt = QSPRDataset(name = model_name,
                                          store_dir = self.rep_name,
                                          df = df_tgt,
                                          target_props = [{"name":self.pred_col, 
                                                           "task":TargetTasks.SINGLECLASS,
                                                           "th":[self.th]}],
                                          smilescol = self.smiles_col,                              
                                          overwrite = True,
                                          n_jobs = self.n_jobs)

                # Create test/training set with descriptors
                dataset_tgt = self.prepareDatasetQSPRPred(dataset_tgt)
                skip, error_msg = self.checkSplit(dataset_tgt)         
                if self.save_split == True:
                    self.saveSplit(dataset_tgt, tgt)             

                # Create model and make predictions
                if not skip:
                    if self.algorithm in ["LR", "NB", "RF", "SVM"]:
                        pred_tgt = self.modelQSPRsklearn(dataset_tgt, model_name)
                    if self.algorithm in ["random"]:
                        pred_tgt = self.modelRandom(df_tgt, model_name)

            # Create random predictions if target is skipped
            if skip:
                skip_count += 1
                pred_tgt = self.modelRandom(df_tgt, model_name)

            # Calculate model results (counts and metrics)
            dataset_counts = self.calculateCounts(dataset_tgt, skip)
            metrics = self.calculateMetrics(pred_tgt)

            results_tgt = pd.DataFrame({"TGT": tgt,  
                                        "DATA": len(df_tgt),
                                        "TRAIN": dataset_counts["TRAIN"], 
                                        "ACT_TRAIN": dataset_counts["ACT_TRAIN"],
                                        "TEST": dataset_counts["TEST"],
                                        "ACT_TEST": dataset_counts["ACT_TEST"],
                                        "BEDROC": metrics["BEDROC"], 
                                        "AUC": metrics["AUC"], 
                                        "MCC": metrics["MCC"], 
                                        "Precision": metrics["Precision"],
                                        "Recall": metrics["Recall"],
                                        "SKIP": error_msg}, index=[0])   
            print(results_tgt)

            # Save model predictions and results into global pool
            pred_pool = pd.concat([pred_pool, pred_tgt])
            results_pool = pd.concat([results_pool, results_tgt], ignore_index=True)

        # Save pooled results
        os.makedirs("./results", exist_ok=True) 
        results_pool.to_csv("./results/tgt_"+self.rep_name+".csv", index=False) 

        # Calculate QSAR metrics
        metrics_Global = pd.DataFrame(self.calculateMetrics(pred_pool), index=[0])
        metrics_PerTarget = results_pool[["BEDROC", "AUC", "MCC", "Precision", "Recall"]].mean()

        return metrics_Global, metrics_PerTarget, skip_count

    def checkData(self, df):
        """
        Check if proper bioactivity data for each target.

        Checks (base):
            Contains active class at threshold - NoActives
            Contains inactive class at threshold - NoInactives
        Checks (strict):
            At least check_param["check_n"] datapoints - InsufficientDatapoints
            At least check_param["check_Ntrain"] actives - InsuficientActives
            At least check_param["check_Ntrain"] inactives - InsuficientInactives
        """
        if df[self.pred_col].max() < self.th:
            error_msg = "NoActives"
        elif df[self.pred_col].min() >= self.th:
            error_msg = "NoInactives"   
        else:
            error_msg = None

        if self.check == "strict": 
            if len(df) < self.check_param["check_n"]:
                error_msg = "InsufficientDatapoints"
            if len(df[df[self.pred_col] >= self.th]) < self.check_param["check_Ntrain"]:
                error_msg = "InsufficientActives"
            if len(df[df[self.pred_col] < self.th]) < self.check_param["check_Ntrain"]:
                error_msg = "InsufficientInactives"            

        # Additional checks if split is temporal
        if self.split == "temp" and error_msg == None:
            error_msg = self.checkDataTemp(df)

        # Skip if a check fails
        if error_msg == None:
            skip = False
        else: 
            skip = True

        return skip, error_msg

    def checkDataTemp(self, df):
        """
        Check if proper bioactivity data for each target for temporal split.

        Checks (base):
            Contains training set with temporal split - NoTraining
            Contains test set with temporal split - NoTest
        """       
        if df[self.split_param["timeprop"]].min() > self.split_param["timesplit"]:         
            error_msg = "NoTraining"
        elif df[self.split_param["timeprop"]].max() <= self.split_param["timesplit"]:
            error_msg = "NoTest"
        else:
            error_msg = None

        return error_msg

    def checkSplit(self, dataset):
        """
        Check if proper split for each target. 
        Checks (base):
            Active and inactive class in training set - SingleClassTrain 
        Checks (strict):
            At least check_param["check_Ntrain"] active datapoints in training set - InsufficientActivesTrain
            At least check_param["check_Ntrain"] inactive datapoints in training set - InsufficientInactivesTrain
            At least check_param["check_ntest"] datapoints in test set - InsufficientTest
        """
        if dataset.y[self.class_col].nunique() == 1:
            error_msg = "SingleClassTrain"
        else: 
            error_msg = None

        if self.check == "strict":
            if len(dataset.y[dataset.y[self.class_col] == True]) < self.check_param["check_Ntrain"]:
                error_msg = "InsufficientActivesTrain"
            elif len(dataset.y[dataset.y[self.class_col] == False]) < self.check_param["check_Ntrain"]:
                error_msg = "InsufficientInactivesTrain"
            elif len(dataset.y_ind) < self.check_param["check_ntest"]:
                print(dataset.y_ind)
                error_msg = "InsufficientTest"
        
        # Skip if a check fails
        if error_msg == None:
            skip = False
        else: 
            skip = True

        return skip, error_msg

    def calculateCounts(self, dataset, skip):
        if skip == False:   
            dp_train = len(dataset.y)
            dp_act_train = (dataset.y[self.class_col] == True).sum()
            dp_test = len(dataset.y_ind)
            dp_act_test = (dataset.y_ind[self.class_col] == True).sum()
        else:
            dp_train = dp_act_train = dp_test = dp_act_test = None

        dataset_counts = {"TRAIN": dp_train,
                          "ACT_TRAIN": dp_act_train,
                          "TEST": dp_test,
                          "ACT_TEST": dp_act_test}

        return dataset_counts

    def calculateMetrics(self, pred):
        mcc, precision, recall = self.calculateMetricsSKLearn(pred) 
        auc, bedroc = self.calculateMetricsRDKit(pred, 20)

        metrics = {"BEDROC": bedroc,
                   "AUC": auc,
                   "MCC": mcc,
                   "Precision": precision,
                   "Recall": recall}

        return metrics

    def calculateMetricsRDKit(self, pred, alpha):
        pred_rd = pred[[self.label_col, self.probN_col]]
        pred_rd = pred_rd.sort_values(by=[self.probN_col], ascending=False)

        auc = CalcAUC(scores = pred_rd.values, col=0)
        bedroc = CalcBEDROC(scores = pred_rd.values, col=0, alpha=alpha)

        return auc, bedroc

    def calculateMetricsSKLearn(self, pred):    
        y_true = pred[self.label_col]
        y_pred = np.where(pred[self.probN_col] >= 0.5, 1, 0)
            
        mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
        precision = precision_score(y_true=y_true, y_pred=y_pred)
        recall = recall_score(y_true=y_true, y_pred=y_pred)

        return mcc, precision, recall

    def modelQSPRsklearn(self, dataset, model_name):
        model = QSPRsklearn(base_dir = self.rep_name, 
                            name = model_name,
                            data = dataset, 
                            alg = self.alg, 
                            parameters = self.alg_param)

        model.evaluate(cross_validation=False)
        model.fit()

        pred_tgt = pd.read_csv("./"+self.rep_name+"/qspr/models/"+model_name+"/"+model_name+".ind.tsv", sep="\t")

        return pred_tgt     

    def modelRandom(self, df_tgt, model_name):
        pred_tgt = pd.DataFrame()

        pred_tgt[self.label_col] = df_tgt[self.pred_col] >= self.th
        pred_tgt[self.probN_col] = np.random.rand(len(pred_tgt))

        pred_tgt.to_csv("./"+self.rep_name+"/qspr/models/"+model_name+"/"+model_name+".ind.tsv", sep="\t", index=False)

        return pred_tgt

    def prepareDatasetQSPRPred(self, dataset):
        dataset.prepareDataset(smiles_standardizer = self.smiles_standardizer,
                               split = self.setSplitter(),
                               feature_calculator = self.feature_calculator,
                               feature_standardizer = self.feature_standardizer)
        
        # During random split, redo selection to ensure training set contains enough active and inactive datapoints
        if self.split == "rand":
            while len(dataset.y[dataset.y[self.class_col] == True]) < self.check_param["check_Ntrain"] or \
                len(dataset.y[dataset.y[self.class_col] == False]) < self.check_param["check_Ntrain"]: 
                random.seed(random.random())
                dataset.prepareDataset(smiles_standardizer = self.smiles_standardizer,
                                       split = self.setSplitter(),
                                       feature_calculator = self.feature_calculator,
                                       feature_standardizer = self.feature_standardizer)

        return dataset

    def saveSplit(self, dataset, tgt):
        os.makedirs("./splits", exist_ok=True)

        train_df = pd.merge(dataset.y, dataset.X, on="QSPRID")
        train_df.to_csv("./splits/"+tgt+"_"+self.split+"_train.csv", index=False)
        test_df = pd.merge(dataset.y_ind, dataset.X_ind, on="QSPRID")
        test_df.to_csv("./splits/"+tgt+"_"+self.split+"_test.csv", index=False) 

        return

    def setAlgorithm(self):
        if self.algorithm == "LR":
            self.alg = LogisticRegression
        elif self.algorithm == "NB":
            self.alg = MultinomialNB
        elif self.algorithm == "RF":
            self.alg = RandomForestClassifier
        elif self.algorithm == "SVM":
            self.alg = SVC
        elif self.algorithm == "random":
            pass
        else:
            sys.exit(f'Error: {self.algorithm} not in algorithm options ["LR", "NB", "RF", "SVM", "random"]')

        return

    def setFeatureCalculator(self):
        descsets = []
        
        for descriptor in self.desc:
            if descriptor == "custom":
                descsets.append(CustomCompoundDescriptor(fname = self.desc_path))
            else:
                sys.exit(f'Error: {descriptor} not in descriptor options ["custom"]')
        
        self.feature_calculator = DescriptorsCalculator(descsets=descsets)

        return

    def setFeatureStandardizer(self):
        if self.desc_standardizer == None:
            self.feature_standardizer = None
        elif self.desc_standardizer == "MinMaxScaler":
            self.feature_standardizer = MinMaxScaler()
        else: 
            sys.exit(f'Error: {self.desc_standardizer} not in standardizer options [None, "MinMaxScaler"]')
        
        return

    def setSplitter(self):
        if self.split == "rand":
            self.splitter = randomsplit(test_fraction=self.split_param["test_fraction"])
        elif self.split == "temp":
            self.splitter = temporalsplit(timesplit=self.split_param["timesplit"],
                                          timeprop=self.split_param["timeprop"])  
        else:
            sys.exit(f'Error: {self.split} not in split options ["rand", "temp"]')

        return self.splitter

if __name__ == "__main__":
    benchmark = Benchmark(cwd = "/home/remco/projects/ml_benchmark/benchmark/test",
                          n_jobs = 1,
                          save_split = False,
                          debug_slice = 10)
    
    benchmark.setData(data_path = "/home/remco/projects/qlattice/dataset/lenselink2017_dataset.csv", 
                      pred_col = "BIOACT_PCHEMBL_VALUE",
                      smiles_col = "Canonical_Smiles",
                      tgt_col = "TGT_CHEMBL_ID", 
                      smiles_standardizer = None)


    ref_runs = {"ref1": {"split":"rand", "split_param": {"test_fraction": 0.3}, "algorithm":"random", "alg_param": {}, "th":5.0},
                "ref2": {"split":"rand", "split_param": {"test_fraction": 0.3}, "algorithm":"random", "alg_param": {}, "th":6.5}, 
                "ref3": {"split":"temp", "split_param": {"timesplit": 2012, "timeprop": "DOC_YEAR"}, "algorithm":"random", "alg_param": {}, "th":5.0},
                "ref4": {"split":"temp", "split_param": {"timesplit": 2012, "timeprop": "DOC_YEAR"}, "algorithm":"random", "alg_param": {}, "th":6.5}}

    for run, variables in ref_runs.items():  
        benchmark.setBenchmark(model = "QSAR", 
                               th = variables["th"], 
                               split = variables["split"], 
                               split_param = variables["split_param"], 
                               algorithm = variables["algorithm"], 
                               alg_param = variables["alg_param"], 
                               desc = ["custom"],
                               desc_standardizer = "MinMaxScaler", 
                               desc_path = "/home/remco/projects/qlattice/dataset/lenselink2017_cmp_desc.json", 
                               name = None, 
                               check = "strict", 
                               check_param = {"check_n": 30, "check_Ntrain": 5, "check_ntest": 5}, 
                               reps = 1)  

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
                               desc_path = "/home/remco/projects/qlattice/dataset/lenselink2017_cmp_desc.json", 
                               name = None, 
                               check = "strict", 
                               check_param = {"check_n": 30, "check_Ntrain": 5, "check_ntest": 5}, 
                               reps = 1)          

        benchmark.run()