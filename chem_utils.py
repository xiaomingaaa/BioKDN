from propy import PyPro
from propy.GetProteinFromUniprot import GetProteinSequence
import numpy as np
import rdkit
from rdkit.Chem import AllChem
from rdkit import Chem,DataStructs



def get_protein_des(seq):
    print(seq)
    DesObject = PyPro.GetProDes(seq)
    dpc = DesObject.GetDPComp() ## 20
    aac = DesObject.GetAAComp() ## 400
    # tpc = DesObject.GetTPComp() ## 8000
    print(np.array(list(aac.values())))
    # x = np.stack([np.array(list(aac.values()), np.array(dpc.values()), np.array(tpc.values())])
    return np.concatenate([list(aac.values()), list(dpc.values())])

def get_smiles_ECPF(smiles):
    """传入smiles编码文件列表"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        fps = AllChem.GetMorganFingerprintAsBitVect(mol,3,2048)
        fps = np.array(fps)
    except:
        fps = None

    return fps