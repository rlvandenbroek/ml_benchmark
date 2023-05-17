import json, gzip, os
import pandas as pd

from rdkit import Chem


def datasetLenselink(cwd, data_dir):
    """
    Process dataset using the Lenselink, et al. (2017) protocol for benchmarking.

    Args: 
        cwd (str) : PATH to working directory
        data_dir (str) : PATH to folder containing bioactivity dataset
    """
    if not os.path.exists(cwd):
        os.makedirs(cwd)
    os.chdir(cwd)    
    
    src1_path = data_dir + '/curated_set_with_publication_year.sd.gz'
    src2_path = data_dir + '/compound_additional_physchem_features.txt.gz'
    src3_path = data_dir + '/compound_features_256.txt.gz'

    data_path = cwd + '/lenselink2017_dataset.csv'
    desc_path = cwd + '/lenselink2017_cmp_desc.json'
    
    # Retrieve dataset (relevant columns)
    data = desc = []

    with gzip.open(src1_path, 'rb') as src1:
        mf_src = Chem.ForwardSDMolSupplier(src1)
        for mol in mf_src: 
            # Create compound-target dataset
            data.append({
                'TGT_CHEMBL_ID': mol.GetProp('TGT_CHEMBL_ID'),            
                'Canonical_Smiles': Chem.MolToSmiles(mol),
                'BIOACT_PCHEMBL_VALUE': mol.GetProp('BIOACT_PCHEMBL_VALUE'),
                'DOC_YEAR': mol.GetProp('DOC_YEAR')})
            
            # Create compound descriptor set (PhysChem)
            desc.append({
                'CMP_CHEMBL_ID': mol.GetProp('CMP_CHEMBL_ID'),
                'Canonical_Smiles': Chem.MolToSmiles(mol),
                'CMP_ALOGP': mol.GetProp('CMP_ALOGP'),
                'CMP_FULL_MWT': mol.GetProp('CMP_FULL_MWT'),
                'CMP_HBD': mol.GetProp('CMP_HBD'),
                'CMP_HBA': mol.GetProp('CMP_HBA'),
                'CMP_RTB': mol.GetProp('CMP_RTB')})

    # Process compound-target dataset        
    df_data = pd.DataFrame(data)
    df_data = df_data.drop_duplicates(subset=['TGT_CHEMBL_ID','Canonical_Smiles'])

    print(f'Datapoints: {len(df_data)}')
    print(f"Targets: {df_data['TGT_CHEMBL_ID'].nunique()}")
    print(f"Compounds: {df_data['Canonical_Smiles'].nunique()}")

    # Expand compound descriptor set (Fractional PSA)      
    df_desc = pd.DataFrame(desc)

    src2 = pd.read_csv(src2_path, compression='gzip', delimiter="\t")
    df_psa = src2[['CMP_CHEMBL_ID', ' CMP_MOLECULAR_POLAR_SURFACEAREA_FRAC']]
    df_psa = df_psa.rename(columns={' CMP_MOLECULAR_POLAR_SURFACEAREA_FRAC': 'CMP_FPSA'})
    df_desc = pd.merge(df_desc, df_psa, on=['CMP_CHEMBL_ID'])

    # Expand compound descriptor set (MorganFP)  
    src3 = pd.read_csv(src3_path, compression='gzip', delimiter="\t")
    df_mfp = src3.rename(columns={'Compound_ID': 'CMP_CHEMBL_ID'})
    df_desc = pd.merge(df_desc, df_mfp, on=['CMP_CHEMBL_ID'])

    # Process compound descriptor set
    df_desc = df_desc.drop_duplicates(subset=['Canonical_Smiles'])
    df_desc = df_desc.drop(["CMP_CHEMBL_ID", "TGT_CHEMBL_ID", "BIOACT_PCHEMBL_VALUE", "DOC_YEAR"], axis=1)
    df_desc = df_desc.set_index(['Canonical_Smiles'], drop=True)

    print(df_desc.columns)
    
    # Save dataset and compound descriptor files
    df_data.to_csv(data_path, index=False)
    
    dict_desc = df_desc.to_dict('index')
    with open(desc_path, 'w') as desc_file:
        json.dump(dict_desc, desc_file)



if __name__ == "__main__":
    datasetLenselink(cwd = '/home/remco/projects/ml_benchmark/dataset',
                     data_dir = '/home/remco/datasets/Lenselink_2017/dataset')     
    