import gzip
import json
import pandas as pd

from rdkit import Chem


def DatasetLenselink(cwd, data_loc):
    """
    Process dataset using the Lenselink, et al. (2017) protocol.

    Args: 
        cwd (str) : PATH to working directory
        data_loc (str) : PATH to folder containing bioactivity dataset
        physchem (bool) : Add physicochemical descriptors the dataset
    """
    src1_path = data_loc + '/curated_set_with_publication_year.sd.gz'
    src2_path = data_loc + '/compound_additional_physchem_features.txt.gz'

    base_path = cwd + '/lenselink2017_dataset.csv'
    desc_path = cwd + '/lenselink2017_cmp_desc.json'
    
    # Retrieve dataset (relevant columns)
    desc = []
    base = []

    with gzip.open(src1_path, 'rb') as src1:
        mf_src = Chem.ForwardSDMolSupplier(src1)
        for mol in mf_src: 
            base.append({
                'TGT_CHEMBL_ID': mol.GetProp('TGT_CHEMBL_ID'),            
                'Canonical_Smiles': Chem.MolToSmiles(mol),
                'BIOACT_PCHEMBL_VALUE': mol.GetProp('BIOACT_PCHEMBL_VALUE'),
                'DOC_YEAR': mol.GetProp('DOC_YEAR')})
            
            # Create base for compound descriptors
            desc.append({
                'CMP_CHEMBL_ID': mol.GetProp('CMP_CHEMBL_ID'),
                'Canonical_Smiles': Chem.MolToSmiles(mol),
                'CMP_ALOGP': mol.GetProp('CMP_ALOGP'),
                'CMP_FULL_MWT': mol.GetProp('CMP_FULL_MWT'),
                'CMP_HBD': mol.GetProp('CMP_HBD'),
                'CMP_HBA': mol.GetProp('CMP_HBA'),
                'CMP_RTB': mol.GetProp('CMP_RTB')})
            
    df_base = pd.DataFrame(base)
    df_base = df_base.drop_duplicates(subset=['TGT_CHEMBL_ID','Canonical_Smiles'])

    print(f'Datapoints: {len(df_base)}')
    print(f"Targets: {df_base['TGT_CHEMBL_ID'].nunique()}")
    print(f"Compounds: {df_base['Canonical_Smiles'].nunique()}")

    # Create dataframe with compound descriptors      
    df_desc = pd.DataFrame(desc)

    src2 = pd.read_csv(src2_path, compression='gzip', delimiter="\t")
    df_psa = src2[['CMP_CHEMBL_ID', ' CMP_MOLECULAR_POLAR_SURFACEAREA_FRAC']]
    df_psa = df_psa.rename(columns={' CMP_MOLECULAR_POLAR_SURFACEAREA_FRAC': 'CMP_FPSA'})

    df_desc = pd.merge(df_desc, df_psa, on=['CMP_CHEMBL_ID'])
    df_desc = df_desc.drop_duplicates(subset=['Canonical_Smiles']).drop(columns=['CMP_CHEMBL_ID'])
    df_desc = df_desc.set_index(['Canonical_Smiles'], drop=True)

    print(f"Compounds with descriptors: {len(df_desc)}")
    
    # Save base and descriptor files
    df_base.to_csv(base_path, index=False)
    
    dict_desc = df_desc.to_dict('index')
    with open(desc_path, 'w') as desc_file:
        json.dump(dict_desc, desc_file)



if __name__ == "__main__":
    N_CPU = 12
    
    DatasetLenselink(cwd = '/home/remco/projects/qlattice/dataset',
                        data_loc = '/home/remco/datasets/Lenselink_2017/dataset')     
    