import pandas as pd
import numpy as np

def load_tasks():

    # train tasks
    data_train = pd.read_csv(f"/homes/ss2971/Documents/AHO/AHO_FP/train_task_identity.csv")
    dataset_train_tasks = {
        'cluster': ['C1','C2','C3','C4','C5','C6','C7','C8']
    }
   
    train_tasks = dataset_train_tasks.get('cluster')
    train_dfs = dict.fromkeys(train_tasks)
    data_train.set_index("cluster", inplace = True)

    for task in train_tasks:
        df = data_train.loc[task]
        train_dfs[task] = df
    
    # validation tasks
    data_val = pd.read_csv(f"/homes/ss2971/Documents/AHO/AHO_FP/val_task_identity.csv")
    dataset_val_tasks = {
        'cluster': ['C1']
    }

    val_tasks = dataset_val_tasks.get('cluster')
    val_dfs = dict.fromkeys(val_tasks)
    data_val.set_index("cluster", inplace = True)

    for task in val_tasks:
        df = data_val.loc[task]
        val_dfs[task] = df
    
    # test tasks
    data_test = pd.read_csv(f"/homes/ss2971/Documents/AHO/AHO_FP/test_task_identity.csv")
    dataset_test_tasks = {
        'cluster': ['C1','C2','C3','C4','C5','C6','C7','C8']
    }
   
    test_tasks = dataset_test_tasks.get('cluster')
    test_dfs = dict.fromkeys(test_tasks)
    data_test.set_index("cluster", inplace = True)

    for task in test_tasks:
        df = data_test.loc[task]
        test_dfs[task] = df

    return train_tasks, train_dfs, val_tasks, val_dfs, test_tasks, test_dfs
