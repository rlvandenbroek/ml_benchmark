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



def benchmarkMachineLearning(data_path,
                             pred_col,
                             smiles_col,
                             desc,
                             model, 
                             th,
                             split, 
                             split_param,
                             algorithm, 
                             alg_param,
                             cwd=os.getcwd(), 
                             tgt_col=None,
                             desc_standardizer=None,
                             desc_path=None,
                             name=None,
                             reps=1, 
                             n_jobs=1,
                             save_split=False, 
                             debug_slice=None):
    """
    Benchmark bioactivity data using the Lenselink, et al. (2017) protocol.

    Args: 
        cwd (str) : PATH to working directory
            Default: os.getcwd()

        data_path (str) : PATH to bioactivity dataset
        pred_col (str) : Column name of property to be predicted
        smiles_col (str) : Column name containing canonical SMILES
        tgt_col (str, optional) : Column name of target identifiers for QSAR models

        desc (list) : Select (multiple) descriptors:
            Options: 'custom'
        desc_standardizer (str) : Select descriptor standardizer:
            Options: None -> Might not work for NB
                     'MinMaxScaler'
        desc_path (str, optional) : PATH to descriptor set if desc = custom

        model (str) : Select ML method:
            Options: 'QSPR'

        th (float) : Set active/inactive threshold

        split (str) : Select datasplit method:
            Options: 'rand' - Random datasplit
                     'temp' - Temporal datasplit
        split_param (dict) : Set datasplit parameters as used in QSPRPred.
            if split == 'rand':
                {test_fraction (float): fraction of total dataset to testset}
            if split == 'temp':
                {timesplit(float): time point after which sample to test set,
                 timeprop (str): column name of column in df that contains the timepoints}
            
        algorithm (str) : Select ML algorithm:
            Options: 'LR' - Logistic Regression
                     'NB' - Naive Bayes      
                     'RF' - Random Forest
                     'SVM' - Support Vector Machine
        alg_param (dict) : Set algorithm parameters. Example:
            if algorithm == 'RF':
                {'n_estimators':1000,
                 'max_depth':None,
                 'max_features':0.3}

        name (str) : Base name to save results
            Default: split + '_' + algorithm + '_' + str(th)
                    
        reps (int) : Set number of replicates
            Default: 1
        n_jobs (int) : Set number of CPUs
            Default: 1

        save_split (bool, optional) : Save test and training sets with descriptors
        debug_slice (int, optional) : Create slice of targets for debugging runs
    """
    # Check if paths exist
    if not os.path.exists(cwd):
        os.makedirs(cwd)
    os.chdir(cwd)    

    if not os.path.exists(data_path):
        sys.exit(f"Error: data_loc ({data_path}) does not exist")
    if desc == 'custom':
        if not os.path.exists(desc_path):
            sys.exit(f"Error: desc_loc ({desc_path}) does not exist")

    # Retrieve dataset
    dataframe = pd.read_csv(data_path, sep=',')

    # Set name
    if not name:
        name = split + '_' + algorithm + '_' + str(th)  
    
    # Create replicates
    for rep in range(reps):
        run_name = name + '_' + str(rep)
 
        # Modelling
        start_time = time.time() 
        if model == 'QSAR':
            metrics_Global, metrics_PerTarget, skip_count = modelQSAR(df = dataframe, 
                                                                      pred_col = pred_col,
                                                                      smiles_col = smiles_col,
                                                                      tgt_col = tgt_col,
                                                                      desc = desc,
                                                                      desc_standardizer = desc_standardizer,
                                                                      desc_path = desc_path,
                                                                      split = split, 
                                                                      split_param = split_param,
                                                                      algorithm = algorithm,
                                                                      alg_param = alg_param,
                                                                      th = th,
                                                                      name = run_name,
                                                                      n_jobs = n_jobs,
                                                                      save_split = save_split,
                                                                      debug_slice = debug_slice)
        else:
            sys.exit(f"Error: {model} not in model options ['QSAR']")
        runtime = time.time() - start_time

        # Calculate general model-wide metrics
        metrics_merged = pd.concat([metrics_Global, metrics_PerTarget])
        metrics_Mean = metrics_merged[['BEDROC', 'AUC', 'MCC', 'Precision', 'Recall']].mean()

        results_global = pd.DataFrame({'NAME': run_name,
                                       'RUNTIME': runtime, 
                                       'N_TGT': dataframe[tgt_col].nunique(),
                                       'N_SKIP': skip_count,
                                       'MEAN_BEDROC': metrics_Mean['BEDROC'], 
                                       'MEAN_AUC': metrics_Mean['AUC'], 
                                       'MEAN_MCC': metrics_Mean['MCC'],
                                       'MEAN_Precision': metrics_Mean['Precision'],        
                                       'MEAN_Recall': metrics_Mean['Recall'],                                    
                                       'GLOBAL_BEDROC': metrics_Global['BEDROC'], 
                                       'GLOBAL_AUC': metrics_Global['AUC'], 
                                       'GLOBAL_MCC': metrics_Global['MCC'],
                                       'GLOBAL_Precision': metrics_Global['Precision'],
                                       'GLOBAL_Recall': metrics_Global['Recall'],
                                       'AVG_BEDROC': metrics_PerTarget['BEDROC'], 
                                       'AVG_AUC': metrics_PerTarget['AUC'], 
                                       'AVG_MCC': metrics_PerTarget['MCC'],
                                       'AVG_Precision': metrics_PerTarget['Precision'],                                    
                                       'AVG_Recall': metrics_PerTarget['Recall']}, index=[0])
        print(results_global)    

        if not os.path.exists('./results/summary.csv'):
            results_global.to_csv('./results/summary.csv', index=False, header=True)
        else:
            results_global.to_csv('./results/summary.csv', mode='a', index=False, header=False)

    return results_global

def modelQSAR(df, pred_col, smiles_col, tgt_col, desc, split, split_param, algorithm, alg_param, th, name, n_jobs, desc_standardizer=None, desc_path=None, save_split=False, debug_slice=None):
    """
    Preform QSAR classification modelling

    Args: 
        df (pd.DataFrame) : Bioactivity dataset
        pred_col (str) : Column name of property to be predicted
        smiles_col (str) : Column name containing canonical SMILES
        tgt_col (str, optional) : Column name of target identifiers for QSAR models

        desc (str) : Select descriptors:
            Options: 'custom'
        desc_standardizer (str) : Select descriptor standardizer:
            Options: None -> Might not work for NB
                     'MinMaxScaler'
            Default: None
        desc_path (str, optional) : PATH to descriptor set if desc = custom

        split (str) : Select datasplit method:
            Options: 'rand' - Random datasplit
                     'temp' - Temporal datasplit
        split_param (dict) : Set datasplit parameters as used in QSPRPred.
            if split == 'rand':
                {test_fraction (float): fraction of total dataset to testset}
            if split == 'temp':
                {timesplit(float): time point after which sample to test set,
                 timeprop (str): column name of column in df that contains the timepoints}

        algorithm (str) : Select ML algorithm:
            Options: 'LR' - Logistic Regression
                     'NB' - Naive Bayes      
                     'RF' - Random Forest
                     'SVM' - Support Vector Machine
        alg_param (dict) : Set algorithm parameters. Example:
            if algorithm == 'RF':
                {'n_estimators':1000,
                 'max_depth':None,
                 'max_features':0.3}

        th (float) : Set active/inactive threshold

        name (str) : Base name to save results
            Default: split + '_' + algorithm + '_' + str(th)
                    
        n_jobs (int) : Set number of CPUs
            Default: 1

        save_split (bool, optional) : Save test and training sets with descriptors
        debug_slice (int, optional) : Create slice of targets for debugging runs
    """    
    dataset = None
    class_col = pred_col + '_class' 
    label_col = class_col + '_Label'
    probAct_col = class_col + '_ProbabilityClass_1'

    results_pool = pred_pool = metrics_Global = pd.DataFrame()
    tgt_count = skip_count = 0

    tgt_total = df[tgt_col].nunique()

    # Split data per target
    for tgt, df_tgt in df.groupby(tgt_col):     
        tgt_count += 1

        if debug_slice:
            if tgt_count > debug_slice:
                continue

        model_name = tgt + '_' + name
        os.makedirs('./'+name+'/qspr/models/'+model_name, exist_ok=True)

        print(f'\n{model_name} ({tgt_count}/{tgt_total}) - {len(df_tgt)} data points')

        # Check data
        skip, error_msg = checkData(df_tgt, pred_col, th, split, split_param)

        # Setup dataset using QSPRPred
        if not skip:
            dataset = QSPRDataset(name = model_name,
                                  store_dir = name,
                                  df = df_tgt,
                                  target_props = [{'name':pred_col, 
                                                   'task':TargetTasks.SINGLECLASS,
                                                   'th':[th]}],
                                  smilescol = smiles_col,                              
                                  overwrite = True,
                                  n_jobs = n_jobs)
    
            feature_calculator = setFeatureCalculator(desc, desc_path)
            feature_standardizer = setFeatureStandardizer(desc_standardizer)

            # Create test/training set with descriptors
            dataset = prepareDatasetQSPRPred(dataset, setSplitter(split, split_param), feature_calculator, feature_standardizer)

            # During random split, redo selection to ensure training set contains enough active and inactive datapoints
            if split == 'rand':
                while len(dataset.y[dataset.y[class_col] == True]) < 5 and \
                      len(dataset.y[dataset.y[class_col] == False]) < 5: 
                    random.seed(random.random())
                    dataset = prepareDatasetQSPRPred(dataset, setSplitter(split), feature_calculator, feature_standardizer)

            # Check split
            skip, error_msg = checkSplit(dataset, class_col)         

            # Save split
            if save_split == True:
                saveSplit(dataset, split, tgt)             

            # Create model and make predictions
            if not skip:
                if algorithm in ['LR', 'NB', 'RF', 'SVM']:
                    pred_tgt = modelQSPRsklearn(dataset, algorithm, alg_param, name, model_name)
                else:
                    sys.exit(f"Error: {algorithm} not in algorithm options ['LR', 'NB', 'RF', 'SVM']")

        # Create random predictions if target is skipped
        if skip:
            skip_count += 1
            pred_tgt = modelRandom(df_tgt, pred_col, label_col, probAct_col, th, name, model_name)

        # Calculate model results (counts and metrics)
        dataset_counts = calculateCounts(dataset, class_col, skip)
        metrics = calculateMetrics(pred_tgt, label_col, probAct_col)

        results_tgt = pd.DataFrame({'TGT': tgt,  
                                    'DATA': len(df_tgt),
                                    'TRAIN': dataset_counts['TRAIN'], 
                                    'ACT_TRAIN': dataset_counts['ACT_TRAIN'],
                                    'TEST': dataset_counts['TEST'],
                                    'ACT_TEST': dataset_counts['ACT_TEST'],
                                    'BEDROC': metrics['BEDROC'], 
                                    'AUC': metrics['AUC'], 
                                    'MCC': metrics['MCC'], 
                                    'Precision': metrics['Precision'],
                                    'Recall': metrics['Recall'],
                                    'SKIP': error_msg}, index=[0])   
        print(results_tgt)

        # Save model predictions and results into global pool
        pred_pool = pd.concat([pred_pool, pred_tgt])
        results_pool = pd.concat([results_pool, results_tgt], ignore_index=True)

    # Save pooled results
    os.makedirs('./results', exist_ok=True) 
    results_pool.to_csv('./results/tgt_'+name+'.csv', index=False) 

    # Calculate QSAR metrics
    metrics_Global = pd.DataFrame(calculateMetrics(pred_pool, label_col, probAct_col), index=[0])
    metrics_PerTarget = results_pool[['BEDROC', 'AUC', 'MCC', 'Precision', 'Recall']].mean()

    return metrics_Global, metrics_PerTarget, skip_count

def checkData(df, pred_col, th, split, split_param):
    """
    Check if proper bioactivity data for each target.
    Checks:
        Contains active class at threshold - NoActives
        Contains inactive class at threshold - NoInactives

    The following checks are not mentioned in BTH, but were verbally confirmed by BB:
        At least 30 datapoints - InsufficientDatapoints
        Has at least 5 actives at threshold - InsufficientActives
        Has at least 5 inactives at threshold - InsufficientInactives
    """
    if df[pred_col].max() < th:
        error_msg = 'NoActives'
    elif df[pred_col].min() >= th:
        error_msg = 'NoInactives'

    elif len(df) < 30:
        error_msg = 'InsufficientDatapoints'       
    elif len(df[df[pred_col] < th]) < 5:
        error_msg = 'InsufficientInactives'
    elif len(df[df[pred_col] >= th]) < 5:
        error_msg = 'InsufficientActives'
       
    else:
        error_msg = None

    # Additional checks if split is temporal
    if split == 'temp' and error_msg == None:
        error_msg = checkDataTemp(df, split_param)

    # Skip if a check fails
    if error_msg == None:
        skip = False
    else: 
        skip = True

    return skip, error_msg

def checkDataTemp(df, split_param):
    """
    Check if proper bioactivity data for each target for temporal split.
    Checks:
        At least 1 datapoint in training set - NoTraining
        At least 1 datapoint in test set - NoTest
    """
    if df[split_param['timeprop']].min() > split_param['timesplit']:         
        error_msg = 'NoTraining'
    elif df[split_param['timeprop']].max() <= split_param['timesplit']:
        error_msg = 'NoTest'
    else:
        error_msg = None

    return error_msg

def checkSplit(dataset, class_col):
    """
    Check if proper split for each target. 
    Checks (general)
        2 classes in training set - singleClassTrain   
    """
    if dataset.y[class_col].nunique() == 1:
        error_msg = 'singleClassTrain'
    else: 
        error_msg = None
    
    # Skip if a check fails
    if error_msg == None:
        skip = False
    else: 
        skip = True

    return skip, error_msg

def calculateCounts(dataset, class_col, skip):
    if skip == None:   
        dp_train = len(dataset.y)
        dp_act_train = (dataset.y[class_col] == True).sum()
        dp_test = len(dataset.y_ind)
        dp_act_test = (dataset.y_ind[class_col] == True).sum()
    else:
        dp_train = dp_act_train = dp_test = dp_act_test = None

    dataset_counts = {'TRAIN': dp_train,
                      'ACT_TRAIN': dp_act_train,
                      'TEST': dp_test,
                      'ACT_TEST': dp_act_test}

    return dataset_counts

def calculateMetrics(pred, label_col, probAct_col):
    mcc, precision, recall = calculateMetricsSKLearn(pred, label_col, probAct_col) 
    auc, bedroc = calculateMetricsRDKit(pred, 20, label_col, probAct_col)

    metrics = {'BEDROC': bedroc,
               'AUC': auc,
               'MCC': mcc,
               'Precision': precision,
               'Recall': recall}

    return metrics

def calculateMetricsRDKit(pred, alpha, label_col, probAct_col):
    pred_rd = pred[[label_col, probAct_col]]
    pred_rd = pred_rd.sort_values(by=[probAct_col], ascending=False)

    auc = CalcAUC(scores = pred_rd.values, col=0)
    bedroc = CalcBEDROC(scores = pred_rd.values, col=0, alpha=alpha)

    return auc, bedroc

def calculateMetricsSKLearn(pred, label_col, probAct_col):    
    y_true = pred[label_col]
    y_pred = np.where(pred[probAct_col] >= 0.5, 1, 0)
           
    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)

    return mcc, precision, recall

def calculateStatistics(values):
    mean = sum(values) / len(values)


    return mean

def modelQSPRsklearn(dataset, algorithm, alg_param, name, model_name):
    alg = setAlgorithm(algorithm)
    model = QSPRsklearn(base_dir = name, 
                        name = model_name,
                        data = dataset, 
                        alg = alg, 
                        parameters = alg_param)

    model.evaluate(cross_validation=False)
    model.fit()

    pred_tgt = pd.read_csv('./'+name+'/qspr/models/'+model_name+'/'+model_name+'.ind.tsv', sep='\t')

    return pred_tgt     

def modelRandom(df_tgt, pred_col, label_col, probAct_col, th, name, model_name):
    pred_tgt = pd.DataFrame()

    pred_tgt[label_col] = df_tgt[pred_col] >= th
    pred_tgt[probAct_col] = np.random.rand(len(pred_tgt))

    pred_tgt.to_csv('./'+name+'/qspr/models/'+model_name+'/'+model_name+'.ind.tsv', sep='\t', index=False)

    return pred_tgt

def prepareDatasetQSPRPred(dataset, splitter, feature_calculator, feature_standardizer):
    dataset.prepareDataset(smiles_standardizer=None,
                           split=splitter,
                           feature_calculator=feature_calculator,
                           feature_standardizer=feature_standardizer)
    
    return dataset

def saveSplit(dataset, split, tgt):
    os.makedirs('./splits', exist_ok=True)

    train_df = pd.merge(dataset.y, dataset.X, on='QSPRID')
    train_df.to_csv('./splits/'+tgt+'_'+split+'_train.csv', index=False)
    test_df = pd.merge(dataset.y_ind, dataset.X_ind, on='QSPRID')
    test_df.to_csv('./splits/'+tgt+'_'+split+'_test.csv', index=False) 

def setAlgorithm(algorithm):
    if algorithm == 'LR':
        alg = LogisticRegression
    if algorithm == 'NB':
        alg = MultinomialNB
    if algorithm == 'RF':
        alg = RandomForestClassifier
    if algorithm == 'SVM':
        alg = SVC

    return alg

def setFeatureCalculator(desc, desc_path):
    descsets = []
    
    for descriptor in desc:
        if descriptor == 'custom':
            descsets.append(CustomCompoundDescriptor(fname=desc_path))
        else:
            sys.exit(f"Error: {descriptor} not in descriptor options ['custom']")
    
    feature_calculator = DescriptorsCalculator(descsets=descsets)

    return feature_calculator

def setFeatureStandardizer(desc_standardizer):
    if desc_standardizer == None:
        feature_standardizer = None
    elif desc_standardizer == 'MinMaxScaler':
        feature_standardizer = MinMaxScaler()
    else: 
        sys.exit(f"Error: {desc_standardizer} not in standardizer options [None, 'MinMaxScaler']")
     
    return feature_standardizer

def setSplitter(split, split_param):
    if split == 'rand':
        splitter = randomsplit(test_fraction=split_param['test_fraction'])
    elif split == 'temp':
        splitter = temporalsplit(timesplit=split_param['timesplit'],
                                 timeprop=split_param['timeprop'])  
    else:
        sys.exit(f"Error: {split} not in split options ['rand', 'temp']")

    return splitter


if __name__ == "__main__":
    runs = {'run1': {'split':'rand', 'split_param': {'test_fraction': 0.3}, 'algorithm':'NB', 'alg_param': {'alpha':1.0, 'force_alpha':True}, 'th':5.0},
            'run2': {'split':'rand', 'split_param': {'test_fraction': 0.3}, 'algorithm':'NB', 'alg_param': {'alpha':1.0, 'force_alpha':True}, 'th':6.5},
            'run3': {'split':'rand', 'split_param': {'test_fraction': 0.3}, 'algorithm':'RF', 'alg_param': {'n_estimators':1000, 'max_depth':None, 'max_features':0.3}, 'th':6.5},
            'run4': {'split':'rand', 'split_param': {'test_fraction': 0.3}, 'algorithm':'SVM', 'alg_param': {'kernel':'rbf', 'gamma':'auto', 'max_iter': -1, 'probability':True}, 'th':6.5},
            'run5': {'split':'rand', 'split_param': {'test_fraction': 0.3}, 'algorithm':'LR', 'alg_param': {'solver':'sag', 'max_iter':100}, 'th':6.5},
            'run6': {'split':'temp', 'split_param': {'timesplit': 2012, 'timeprop': 'DOC_YEAR'}, 'algorithm':'NB', 'alg_param': {'alpha':1.0, 'force_alpha':True}, 'th':5.0},
            'run7': {'split':'temp', 'split_param': {'timesplit': 2012, 'timeprop': 'DOC_YEAR'}, 'algorithm':'NB', 'alg_param': {'alpha':1.0, 'force_alpha':True}, 'th':6.5},
            'run8': {'split':'temp', 'split_param': {'timesplit': 2012, 'timeprop': 'DOC_YEAR'}, 'algorithm':'RF', 'alg_param': {'n_estimators':1000, 'max_depth':None, 'max_features':0.3}, 'th':6.5},
            'run9': {'split':'temp', 'split_param': {'timesplit': 2012, 'timeprop': 'DOC_YEAR'}, 'algorithm':'SVM', 'alg_param': {'kernel':'rbf', 'gamma':'auto', 'max_iter': -1, 'probability':True}, 'th':6.5},
            'run10': {'split':'temp', 'split_param': {'timesplit': 2012, 'timeprop': 'DOC_YEAR'}, 'algorithm':'LR', 'alg_param': {'solver':'sag', 'max_iter':100}, 'th':6.5}}

    for run, variables in runs.items():        
        benchmarkMachineLearning(cwd = '/home/remco/projects/ml_benchmark/benchmark/test',
                                 data_path = '/home/remco/projects/qlattice/dataset/lenselink2017_dataset.csv',
                                 pred_col = 'BIOACT_PCHEMBL_VALUE',
                                 smiles_col = 'Canonical_Smiles',
                                 tgt_col = 'TGT_CHEMBL_ID',
                                 desc = ['custom'],
                                 desc_standardizer = 'MinMaxScaler',
                                 desc_path = '/home/remco/projects/qlattice/dataset/lenselink2017_cmp_desc.json',
                                 model= 'QSAR',  
                                 split = variables['split'],   
                                 split_param = variables['split_param'],  
                                 algorithm = variables['algorithm'],   
                                 alg_param = variables['alg_param'],         
                                 th = variables['th'],
                                 reps = 1,
                                 name = None,
                                 n_jobs = 1,
                                 save_split = False,
                                 debug_slice = 10)              
