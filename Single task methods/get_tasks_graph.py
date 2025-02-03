import pandas as pd
import os, csv
import numpy as np
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures

import torch
import dgl
from dgl.convert import graph
from statistics import mean
from sklearn.model_selection import train_test_split

chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

def mol_dict():
    return {'n_node': [],
            'n_edge': [],
            'node_attr': [],
            'edge_attr': [],
            'src': [],
            'dst': []}

def get_graph_data(rsmi_list, class_list, filename):

    def add_mol(mol_dict, mol):

        def _DA(mol):
    
            D_list, A_list = [], []
            for feat in chem_feature_factory.GetFeaturesForMol(mol):
                if feat.GetFamily() == 'Donor': D_list.append(feat.GetAtomIds()[0])
                if feat.GetFamily() == 'Acceptor': A_list.append(feat.GetAtomIds()[0])
            
            return D_list, A_list

        def _chirality(atom):
            
            if atom.HasProp('Chirality'):
                c_list = [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] 
            else:
                c_list = [0, 0]

            return c_list
            
        def _stereochemistry(bond):
            
            if bond.HasProp('Stereochemistry'):
                s_list = [(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')] 
            else:
                s_list = [0, 0]

            return s_list     
            

        n_node = mol.GetNumAtoms()
        n_edge = mol.GetNumBonds() * 2
        
        D_list, A_list = _DA(mol)  
        atom_fea1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
        atom_fea2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-1]
        atom_fea5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors = True)) for a in mol.GetAtoms()]][:,:-1]
        atom_fea6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
        atom_fea8 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
        atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
        atom_fea10 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype = bool)
        
        node_attr = np.hstack([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10])
    
        mol_dict['n_node'].append(n_node)
        mol_dict['n_edge'].append(n_edge)
        mol_dict['node_attr'].append(node_attr)
    
        if n_edge > 0:

            bond_fea1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
            bond_fea2 = np.array([[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()], dtype = bool)
            bond_fea3 = np.array([_stereochemistry(b) for b in mol.GetBonds()], dtype = bool)
            
            edge_attr = np.hstack([bond_fea1, bond_fea2, bond_fea3])
            edge_attr = np.vstack([edge_attr, edge_attr])
            
            bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype=int)
            src = np.hstack([bond_loc[:,0], bond_loc[:,1]])
            dst = np.hstack([bond_loc[:,1], bond_loc[:,0]])
        
            mol_dict['edge_attr'].append(edge_attr)
            mol_dict['src'].append(src)
            mol_dict['dst'].append(dst)
        
        return mol_dict

    def add_dummy(mol_dict):

        n_node = 1
        n_edge = 0
        node_attr = np.zeros((1, len(atom_list) + len(charge_list) + len(degree_list) + len(hybridization_list) + len(hydrogen_list) + len(valence_list) + len(ringsize_list) + 1))
    
        mol_dict['n_node'].append(n_node)
        mol_dict['n_edge'].append(n_edge)
        mol_dict['node_attr'].append(node_attr)
        
        return mol_dict
  
    def dict_list_to_numpy(mol_dict):
    
        mol_dict['n_node'] = np.array(mol_dict['n_node']).astype(int)
        mol_dict['n_edge'] = np.array(mol_dict['n_edge']).astype(int)
        mol_dict['node_attr'] = np.vstack(mol_dict['node_attr']).astype(bool)
        if np.sum(mol_dict['n_edge']) > 0:
            mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr']).astype(bool)
            mol_dict['src'] = np.hstack(mol_dict['src']).astype(int)
            mol_dict['dst'] = np.hstack(mol_dict['dst']).astype(int)
        else:
            mol_dict['edge_attr'] = np.empty((0, len(bond_list) + 2)).astype(bool)
            mol_dict['src'] = np.empty(0).astype(int)
            mol_dict['dst'] = np.empty(0).astype(int)

        return mol_dict
    
    atom_list = ['C','N','O','F','P','S','Cl','Br','K','I','Si','Fe','B','Na','H','Rh','Cr']
    
    charge_list = [1, 2, -1, 0]
    degree_list = [1, 2, 3, 4, 6, 0]
    valence_list = [1, 2, 3, 4, 5, 6, 0]
    hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S']
    hydrogen_list = [1, 2, 3, 4, 0]
    ringsize_list = [3, 4, 5, 6, 7, 8]
    bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
    
    rmol_dict = mol_dict()         
    reaction_dict = {'ee_class': [], 'rsmi': []}     
    
    print('--- generating graph data for %s' %filename)
    print('--- n_reactions: %d' %(len(rsmi_list))) 
                 
    for i in range(len(rsmi_list)):
    
        rsmi = rsmi_list[i].replace('~', '-')
        ee_class = class_list[i]

        rmol = Chem.MolFromSmiles(rsmi)
        rs = Chem.FindPotentialStereo(rmol)
        for element in rs:
            if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified': rmol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
            elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified': rmol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))

        rmol = Chem.RemoveHs(rmol)
        rmol_dict = add_mol(rmol_dict, rmol) 
        
        # ee and SMILES
        reaction_dict['ee_class'].append(ee_class)
        reaction_dict['rsmi'].append(rsmi)
    
        # monitoring
        if (i+1) % 1000 == 0: print('--- %d/%d processed' %(i+1, len(rsmi_list))) 
        
    # datatype to numpy
    rmol_dict = dict_list_to_numpy(rmol_dict)   
    reaction_dict['ee_class'] = np.array(reaction_dict['ee_class'])

    # save file
    np.savez_compressed(filename, data = [rmol_dict, reaction_dict]) 
    return rmol_dict
   
class GraphDataset():

    def __init__(self, id, task, id1):
        self._task = task
        self._id = id
        self._id1 = id1
        self.load()

    def load(self):

        [rmol_dict, reaction_dict] = np.load('/homes/ss2971/Documents/AHO/AHO_Graph/%s_tasks_%s_cluster_%s.npz' %(self._task, self._id, self._id1), allow_pickle=True)['data']
    
        self.rmol_n_node = rmol_dict['n_node']
        self.rmol_n_edge = rmol_dict['n_edge'] 
        self.rmol_node_attr = rmol_dict['node_attr'] 
        self.rmol_edge_attr = rmol_dict['edge_attr'] 
        self.rmol_src = rmol_dict['src']
        self.rmol_dst = rmol_dict['dst'] 
        
        self.label = reaction_dict['ee_class']
        self.rsmi = reaction_dict['rsmi']

        self.rmol_n_csum = np.concatenate([[0], np.cumsum(self.rmol_n_node)])
        self.rmol_e_csum = np.concatenate([[0], np.cumsum(self.rmol_n_edge)])
        
    def __getitem__(self, idx):

        g1 = graph((self.rmol_src[self.rmol_e_csum[idx]:self.rmol_e_csum[idx+1]],
                     self.rmol_dst[self.rmol_e_csum[idx]:self.rmol_e_csum[idx+1]]
                     ), num_nodes = self.rmol_n_node[idx])
              
        g1.ndata['attr'] = torch.from_numpy(self.rmol_node_attr[self.rmol_n_csum[idx]:self.rmol_n_csum[idx+1]]).float()
        g1.edata['edge_attr'] = torch.from_numpy(self.rmol_edge_attr[self.rmol_e_csum[idx]:self.rmol_e_csum[idx+1]]).float()
        
        label = self.label[idx]
        
        return g1, label
            
    def __len__(self):
        return self.label.shape[0]
        
# df = pd.read_csv(f"/homes/ss2971/Documents/AHO/AHO_Graph/test_tasks_graph_wo_cluster.csv")
# df = df.to_numpy()
# np.save("/homes/ss2971/Documents/AHO/AHO_Graph/test_task_graph_cluster_wo", df)

def load_dataset(task, id1):
    load_dict = np.load('/homes/ss2971/Documents/AHO/AHO_Graph/%s_task_graph_cluster_%s.npy' %(task, id1), allow_pickle=True)
    # load_dict = np.load('/homes/ss2971/Documents/AHO/AHO_Graph/%s_task_graph_cluster_%s_rnd.npy' %(task, id1), allow_pickle=True)
    
    reactant_smiles = load_dict[:, 0]
    ligand_smiles = load_dict[:, 1]
    solvent_smiles = load_dict[:, 2]
    ee_class = load_dict[:, 14]
    ee = load_dict[:, 13]

    filename1 = '/homes/ss2971/Documents/AHO/AHO_Graph/%s_tasks_react_cluster_%s.npz'%(task, id1)
    filename2 = '/homes/ss2971/Documents/AHO/AHO_Graph/%s_tasks_lig_cluster_%s.npz'%(task, id1)
    filename3 = '/homes/ss2971/Documents/AHO/AHO_Graph/%s_tasks_solv_cluster_%s.npz'%(task, id1)
    reactant_data = get_graph_data(reactant_smiles, ee_class, filename1)
    ligand_data = get_graph_data(ligand_smiles, ee_class, filename2)
    solvent_data = get_graph_data(solvent_smiles, ee_class, filename3)
    
    data_react = GraphDataset(id='react', task=task, id1=id1)
    data_lig = GraphDataset(id='lig', task=task, id1=id1)
    data_solv = GraphDataset(id='solv', task=task, id1=id1)
    
    batchdata_react = list(map(list, zip(*data_react))) 
    reactant_graph = batchdata_react[0]
    batchdata_lig = list(map(list, zip(*data_lig)))  
    ligand_graph = batchdata_lig[0]
    batchdata_solv = list(map(list, zip(*data_solv)))  
    solvent_graph = batchdata_solv[0]
    
    dict = {'reactant_graph':reactant_graph, 'ligand_graph':ligand_graph, 'solvent_graph':solvent_graph, 'pressure':load_dict[:, 5], 'temperature':load_dict[:, 6], 'S/C':load_dict[:, 7], 'metal_1':load_dict[:, 8], 'metal_2':load_dict[:, 9], 'metal_3':load_dict[:, 10], 'add_1':load_dict[:, 11], 'add_2':load_dict[:, 12], 'ee_class':load_dict[:, 14], 'cluster':load_dict[:, 15]}
 
    data = pd.DataFrame(dict)
    data.to_pickle(f"/homes/ss2971/Documents/AHO/AHO_Graph/{task}_tasks_graph_cluster_{id1}.pkl")
    
    return None

# load_dataset('test', 'oob')

def load_tasks_graph():

    # train tasks
    data_train = pd.read_pickle(f"/homes/ss2971/Documents/AHO/AHO_Graph/train_tasks_graph_cluster_wo.pkl")
    dataset_train_tasks = {
        'cluster': ['C1','C2','C3','C4','C5','C6','C7','C8']
    }
    
    train_tasks = dataset_train_tasks.get('cluster')
    train_dfs = dict.fromkeys(train_tasks)
    data_train.set_index("cluster", inplace = True)

    for task in train_tasks:
        df = data_train.loc[task]
        df.columns = ['reactant_graph','ligand_graph','solvent_graph','pressure','temperature','S/C','metal_1','metal_2','metal_3','add_1','add_2','ee_class']
        train_dfs[task] = df
    
    # validation tasks
    data_val = pd.read_pickle(f"/homes/ss2971/Documents/AHO/AHO_Graph/val_tasks_graph_cluster_wo.pkl")
    
    dataset_val_tasks = {
        'cluster': ['C1']
    }

    val_tasks = dataset_val_tasks.get('cluster')
    val_dfs = dict.fromkeys(val_tasks)
    data_val.set_index("cluster", inplace = True)

    for task in val_tasks:
        df = data_val.loc[task]
        df.columns = ['reactant_graph','ligand_graph','solvent_graph','pressure','temperature','S/C','metal_1','metal_2','metal_3','add_1','add_2','ee_class']
        val_dfs[task] = df
    
    # test tasks
    data_test = pd.read_pickle(f"/homes/ss2971/Documents/AHO/AHO_Graph/test_tasks_graph_cluster_wo.pkl")
    dataset_test_tasks = {
        'cluster': ['C1','C2','C3','C4','C5','C6','C7','C8']
    }

    test_tasks = dataset_test_tasks.get('cluster')
    test_dfs = dict.fromkeys(test_tasks)
    data_test.set_index("cluster", inplace = True)

    for task in test_tasks:
        df = data_test.loc[task]
        df.columns = ['reactant_graph','ligand_graph','solvent_graph','pressure','temperature','S/C','metal_1','metal_2','metal_3','add_1','add_2','ee_class']
        test_dfs[task] = df

    return train_tasks, train_dfs, val_tasks, val_dfs, test_tasks, test_dfs
