import math, os, random, sys, time
import numpy as np
import pandas as pd

from qsprpred.data.data import QSPRDataset
from qsprpred.data.utils.datasplitters import randomsplit, temporalsplit
from qsprpred.data.utils.descriptorcalculator import DescriptorsCalculator
from qsprpred.data.utils.descriptorsets import CustomCompoundDescriptor, FingerprintSet
from qsprpred.models.tasks import TargetTasks
from qsprpred.models.models import QSPRsklearn

from rdkit.ML.Scoring.Scoring import CalcAUC, CalcBEDROC

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC



def benchmarkLenselink(cwd, data_loc, desc_loc, model, split, algorithm, th, name=None, save_split=False, debug_slice=None):
    """
    Benchmark bioactivity data using the Lenselink, et al. (2017) protocol.

    Args: 
        name (str, optional) : Base name to save results
                        Default: name=split+'_'+algorithm+'_'+model
        cwd (str) : PATH to working directory
        data_loc (str) : PATH to bioactivity dataset
        desc_loc (str) : PATH to descriptor set
        model (str) : Select ML model:
                         'QSPR'
        split (str) : Select datasplit:
                        'rand' - Random datasplit
                        'temp' - Temporal datasplit
        algorithm (str) : Select algorithm:
                            'LR' - Logistic Regression
                            'NB' - Naive Bayes    
                            'RF' - Random Forest
                            'SVM' - Support Vector Machine 
        th (float) : Set active/inactive threshold
        save_split (bool, optional) : Save test and training sets with descriptors
        debug_slice (int, optional) : Create slice of X targets
    """
    if not os.path.exists(cwd):
        os.makedirs(cwd)
    os.chdir(cwd)    

    if not os.path.exists(data_loc):
        sys.exit(f"Error: data_loc ({data_loc}) does not exist")
    if not os.path.exists(desc_loc):
        sys.exit(f"Error: desc_loc ({desc_loc}) does not exist")

    if not name:
        name = split+'_'+algorithm+'_'+str(th)
 
    # Retrieve dataset
    dataframe = pd.read_csv(data_loc, sep=',')

    # Modelling
    start_time = time.time() 
    if model == 'QSAR':
        pred_pool, results_compl, skip_count = qsprClassification(name = name,
                                                                  df = dataframe, 
                                                                  th = th,
                                                                  split = split,  
                                                                  desc_loc = desc_loc,                                                          
                                                                  algorithm = algorithm,
                                                                  save_split = save_split,
                                                                  debug_slice = debug_slice
                                                                  )
    else:
        sys.exit(f"Error: {model} not in model options ['QSAR']")

    runtime = time.time() - start_time

    # Calculate final metrics
    avg = results_compl[['BEDROC', 'AUC', 'MCC', 'Precision', 'Recall']].mean()
    avg_bedroc, avg_auc, avg_mcc, avg_precision, avg_recall = avg['BEDROC'], avg['AUC'], avg['MCC'], avg['Precision'], avg['Recall']
    global_mcc, global_precision, global_recall, global_auc, global_bedroc = calculateMetrics(pred=pred_pool)

    mean_bedroc, sem_bedroc = calculateStatistics([avg_bedroc, global_bedroc])
    mean_auc, sem_auc = calculateStatistics([avg_auc, global_auc])
    mean_mcc, sem_mcc = calculateStatistics([avg_mcc, global_mcc])
    mean_precision, sem_precision = calculateStatistics([avg_precision, global_precision])
    mean_recall, sem_recall = calculateStatistics([avg_recall, global_recall])

    results_global = pd.DataFrame({'NAME': name,
                                   'RUNTIME': runtime, 
                                   'N_TGT': dataframe['TGT_CHEMBL_ID'].nunique(),
                                   'N_SKIP': skip_count,
                                   'MEAN_BEDROC': mean_bedroc, 
                                   'MEAN_AUC': mean_auc, 
                                   'MEAN_MCC': mean_mcc,
                                   'MEAN_Precision': mean_precision,        
                                   'MEAN_Recall': mean_recall,
                                   'SEM_BEDROC': sem_bedroc, 
                                   'SEM_AUC': sem_auc, 
                                   'SEM_MCC': sem_mcc,
                                   'SEM_Precision': sem_precision,
                                   'SEM_Recall': sem_recall,                                    
                                   'GLOBAL_BEDROC': global_bedroc, 
                                   'GLOBAL_AUC': global_auc, 
                                   'GLOBAL_MCC': global_mcc,
                                   'GLOBAL_Precision': global_precision,
                                   'GLOBAL_Recall': global_recall,
                                   'AVG_BEDROC': avg_bedroc, 
                                   'AVG_AUC': avg_auc, 
                                   'AVG_MCC': avg_mcc,
                                   'AVG_Precision': avg_precision,                                    
                                   'AVG_Recall': avg_recall                                    
                                  }, index=[0])
    print(results_global)    

    os.makedirs('./results', exist_ok=True) 
    results_compl.to_csv('./results/tgt_'+name+'.csv', index=False)

    if not os.path.exists('./results/summary.csv'):
        results_global.to_csv('./results/summary.csv', index=False, header=True)
    else:
        results_global.to_csv('./results/summary.csv', mode='a', index=False, header=False)

def qsprClassification(name, df, th, split, desc_loc, algorithm, save_split=False, debug_slice=None):
    """
    Preform QSAR classification modelling for a whole dataset

    Args: 
        name (str) : Base name for QSAR models
        df (pd.DataFrame) : Bioactivity dataset
        th (float) : Set active/inactive threshold
        split (str) : Selected datasplit    
        desc_loc (str) : PATH to descriptor set
        algorithm (str) : Selected algorithm
        save_split (bool, optional) : Save test and training sets with descriptors
        debug_slice (int, optional) : Create slice of X targets
    """    
    results_compl = pred_pool = pd.DataFrame()
    
    tgt_count = skip_count = 0
    tgt_total = df['TGT_CHEMBL_ID'].nunique()

    for tgt, df_tgt in df.groupby('TGT_CHEMBL_ID'):     
        tgt_count += 1
        if debug_slice:
            if tgt_count > debug_slice:
                continue
        
        model_name = tgt +'_'+ name
        os.makedirs('./'+name+'/qspr/models/'+model_name, exist_ok=True)

        print(f'\n{model_name} ({tgt_count}/{tgt_total}) - {len(df_tgt)} data points')

        pred_tgt = pd.DataFrame()

        # Check data
        skip, message = checkData(df_tgt, th, split)

        # Setup QSPRPred dataset and make predictions
        if not skip:
            dataset = QSPRDataset(name = model_name,
                                  store_dir = name,
                                  df = df_tgt,
                                  target_props = [{'name':"BIOACT_PCHEMBL_VALUE", 
                                                   'task':TargetTasks.SINGLECLASS,
                                                   'th':[th]}],
                                  smilescol = 'Canonical_Smiles',                              
                                  overwrite = True)
    
            feature_calculator = setFeatureCalculator(desc_loc)
            feature_standardizer = setFeatureStandardizer(algorithm)

            dataset = prepareDatasetQSPRPred(dataset, setSplitter(split), feature_calculator, feature_standardizer)
            if split == 'rand':
                while dataset.y['BIOACT_PCHEMBL_VALUE_class'].nunique() == 1:
                    random.seed(random.random())
                    dataset = prepareDatasetQSPRPred(dataset, setSplitter(split), feature_calculator, feature_standardizer)

            # Check split
            skip, message = checkSplit(dataset.y)
            dp_train = len(dataset.y)
            dp_act_train = (dataset.y['BIOACT_PCHEMBL_VALUE_class'] == True).sum()
            dp_test = len(dataset.y_ind)
            dp_act_test = (dataset.y_ind['BIOACT_PCHEMBL_VALUE_class'] == True).sum()           

            # Save split
            if save_split == True:
                train_df = pd.merge(dataset.y, dataset.X, on='QSPRID')
                test_df = pd.merge(dataset.y_ind, dataset.X_ind, on='QSPRID')
                
                os.makedirs('./splits', exist_ok=True) 
                train_df.to_csv('./splits/'+tgt+'_'+split+'_train.csv', index=False)
                test_df.to_csv('./splits/'+tgt+'_'+split+'_test.csv', index=False)                

            # Create model and make predictions
            if not skip:
                if algorithm in ['LR', 'NB', 'RF', 'SVM']:
                    alg, param = setAlgorithm(algorithm)
                    model = QSPRsklearn(base_dir = name, 
                                        name = model_name,
                                        data = dataset, 
                                        alg = alg, 
                                        parameters=param)

                    model.evaluate(cross_validation=False)
                    model.fit()

        # Calculate metrics from predictions
        if not skip:
            if algorithm in ['LR', 'NB', 'RF', 'SVM']:
                pred_tgt = pd.read_csv('./'+name+'/qspr/models/'+model_name+'/'+model_name+'.ind.tsv', sep='\t')             
        else:
            # Create random target prediction
            skip_count += 1
            dp_train = dp_test = dp_act_train = dp_act_test = None
            pred_tgt['BIOACT_PCHEMBL_VALUE_class_Label'] = df_tgt['BIOACT_PCHEMBL_VALUE'] >= th
            pred_tgt['BIOACT_PCHEMBL_VALUE_class_ProbabilityClass_1'] = np.random.rand(len(pred_tgt))

        if skip:
            pred_tgt.to_csv('./'+name+'/qspr/models/'+model_name+'/'+model_name+'.ind.tsv', sep='\t', index=False)

        pred_pool = pd.concat([pred_pool, pred_tgt])

        mcc, precision, recall, auc, bedroc = calculateMetrics(pred_tgt)

        results_tgt = pd.DataFrame({'TGT': tgt,  
                                    'DATA': len(df_tgt),
                                    'TRAIN': dp_train, 
                                    'ACT_TRAIN': dp_act_train,
                                    'TEST': dp_test,
                                    'ACT_TEST': dp_act_test,
                                    'BEDROC': bedroc, 
                                    'AUC': auc, 
                                    'MCC': mcc, 
                                    'Precision': precision,
                                    'Recall': recall,
                                    'SKIP': message}, 
                                    index=[0])   
        print(results_tgt)
        results_compl = pd.concat([results_compl, results_tgt], ignore_index=True)   

    return pred_pool, results_compl, skip_count

def checkData(df, th, split):
    """
    Check if proper bioactivity data for each target.
    Checks (general):
        At least 30 datapoints - Insufficient Datapoints (ID)
        At least 1 active data point at threshold - No Actives (NA)
        At least 1 inactive data point at threshold - No Inactives (NI)
    Checks (temporal split)
        At least 1 datapoint in training set - No Train (NTr)
        At least 1 datapoint in test set - No Test (NTe)
    """
    skip = True
    message = None

    if len(df) < 30:
        message = "ID"    
    elif df['BIOACT_PCHEMBL_VALUE'].max() < th:
        message = 'NA'
    elif df['BIOACT_PCHEMBL_VALUE'].min() >= th:
        message = 'NI'
    elif split == 'temp' and df['DOC_YEAR'].min() >= 2013:         
        message = 'NTr'
    elif split == 'temp' and df['DOC_YEAR'].max() < 2013:
        message = 'NTe'
    else:
        skip = False
    
    return skip, message

def checkSplit(df):
    """
    Check if proper split for each target. 
    Checks (general)
        2 classes in training set - Single Class in Train (SC)   
    """
    skip = True
    message = None
    
    if df['BIOACT_PCHEMBL_VALUE_class'].nunique() == 1:
        message = 'SC'
    elif len(df[df['BIOACT_PCHEMBL_VALUE_class'] == True]) < 5:
        message = 'IA'
    elif len(df[df['BIOACT_PCHEMBL_VALUE_class'] == False]) < 5:
        message = 'II'
    else: 
        skip = False
    
    return skip, message

def calculateMetrics(pred):
    mcc, precision, recall = calculateMetricsSKLearn(pred) 
    auc, bedroc = calculateMetricsRDKit(pred, 20)

    return mcc, precision, recall, auc, bedroc

def calculateMetricsRDKit(pred, alpha):
    pred_rd = pred[['BIOACT_PCHEMBL_VALUE_class_Label', 'BIOACT_PCHEMBL_VALUE_class_ProbabilityClass_1']]
    pred_rd = pred_rd.sort_values(by=['BIOACT_PCHEMBL_VALUE_class_ProbabilityClass_1'], ascending=False)

    auc = CalcAUC(scores = pred_rd.values, col=0)
    bedroc = CalcBEDROC(scores = pred_rd.values, col=0, alpha=alpha)

    return auc, bedroc

def calculateMetricsSKLearn(pred):    
    y_true = pred['BIOACT_PCHEMBL_VALUE_class_Label']
    y_pred = np.where(pred['BIOACT_PCHEMBL_VALUE_class_ProbabilityClass_1'] >= 0.5, 1, 0)
           
    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)

    return mcc, precision, recall

def calculateStatistics(values):
    mean = sum(values) / len(values)
    variance = sum([(value - mean) ** 2 for value in values]) / (len(values) - 1)
    sem = math.sqrt(variance / len(values))

    return mean, sem

def prepareDatasetQSPRPred(dataset, splitter, feature_calculator, feature_standardizer):
    dataset.prepareDataset(smiles_standardizer=None,
                           split=splitter,
                           feature_calculator=feature_calculator,
                           feature_standardizer=feature_standardizer)
    
    return dataset

def setAlgorithm(algorithm):
    if algorithm == 'LR':
        alg = LogisticRegression
        param = {'solver':'sag',
                      'max_iter':100}
    elif algorithm == 'NB':
        alg = MultinomialNB
        param = None
    elif algorithm == 'RF':
        alg = RandomForestClassifier
        param = {'n_estimators':1000,
                        'max_depth':None,
                        'max_features':0.3}
    elif algorithm == 'SVM':
        alg = SVC
        param = {'kernel':'rbf',
                        'gamma':'auto',
                        'C':1,
                        'probability':True}
    else:
        sys.exit(f"Error: {algorithm} not in algorithm options ['LR', 'NB', 'RF', 'SVM']")

    return alg, param

def setFeatureCalculator(desc_loc, morganFP):
    descsets = [CustomCompoundDescriptor(fname=desc_loc)]
    descsets.append(FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=256))
    
    feature_calculator = DescriptorsCalculator(descsets=descsets)

    return feature_calculator

def setFeatureStandardizer(algorithm):
    if algorithm == 'NB':
        feature_standardizer = MinMaxScaler()
    else:
        feature_standardizer = None
    
    return feature_standardizer

def setSplitter(split):
    if split == 'rand':
        splitter = randomsplit(test_fraction=0.3)
    elif split == 'temp':
        splitter = temporalsplit(timesplit=2012,
                              timeprop='DOC_YEAR')  
    else:
        sys.exit(f"Error: {split} not in split options ['rand', 'temp']")

    return splitter


if __name__ == "__main__":
    N_CPU = 12
    
    replicates = ['1']
    for run_id in replicates:
        runs = {'run1': {'split':'rand', 'algorithm':'NB', 'th':5.0},
                'run2': {'split':'rand', 'algorithm':'NB', 'th':6.5},
                'run3': {'split':'rand', 'algorithm':'RF', 'th':6.5},
                'run4': {'split':'rand', 'algorithm':'SVM', 'th':6.5},
                'run5': {'split':'rand', 'algorithm':'LR', 'th':6.5},
                'run6': {'split':'temp', 'algorithm':'NB', 'th':5.0},
                'run7': {'split':'temp', 'algorithm':'NB', 'th':6.5},
                'run8': {'split':'temp', 'algorithm':'RF', 'th':6.5},
                'run9': {'split':'temp', 'algorithm':'SVM', 'th':6.5},
                'run10': {'split':'temp', 'algorithm':'LR', 'th':6.5}}

        for run, variables in runs.items():
            
            name = variables['split'] + '_' + variables['algorithm'] + '_' + str(variables['th'])
            name += '_'+run_id
            
            print(f'\n----- Now running: {name} -----')

            benchmarkLenselink(cwd = '/home/remco/projects/ml_benchmark/benchmark/test',
                            data_loc = '/home/remco/projects/qlattice/dataset/lenselink2017_dataset.csv',
                            desc_loc = '/home/remco/projects/qlattice/dataset/lenselink2017_cmp_desc.json',
                            name = name,
                            model= 'QSAR',  
                            split = variables['split'],     
                            algorithm = variables['algorithm'],            
                            th = variables['th'],
                            save_split = False,
                            debug_slice = None)              
