import numpy as np
import pandas as pd
import surprise
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import KFold
from surprise.prediction_algorithms.matrix_factorization import NMF, SVD

def readData():
    df = pd.read_csv(
        './ml-latest-small/ratings.csv',
        delimiter=',',
        names = ['uid', 'mid', 'r', 't'],
        header=0
    )   
    return df

def getRatingMatrix(df,num_m):
    R = np.zeros([np.max(df['uid']),num_m])
    for i in range(len(df['uid'])):
        R[df['uid'][i]-1,df['mid'][i]-1] = df['r'][i]
    return R


    
def train_knn(data):
    rmse = []
    mae = []
    sim_options = {'name': 'pearson'}
    for k in range(2, 102, 2):
        print ("using k = %d" % k)
        knn = KNNWithMeans(k = k, sim_options = sim_options)
        temp = cross_validate(knn, data, measures=['RMSE', 'MAE'], cv=10)
        rmse.append(np.mean(temp['test_rmse']))
        mae.append(np.mean(temp['test_mae']))
    print ("k-fold validation finished!")
    return (rmse, mae)

def trim(testSet,R):
    C = np.copy(R)
    C[C > 0] = 1
    C = np.sum(C,axis=0)
    var = np.var(R,axis=0)
    (p_testset, u_testset,hv_testset) = ([],[],[])
    for (u,m,r) in testSet:
        if C[int(m)] > 2:
            p_testset.append((u,m,r))
        if C[int(m)] <= 2:
            u_testset.append((u,m,r))
        if C[int(m)] >= 5 and var[int(m)] >= 2:
            hv_testset.append((u,m,r))
    return (p_testset, u_testset,hv_testset)
               
def train_trim_knn(data,R):
    kfold = KFold(n_splits = 10)
    sim_options = {'name': 'pearson'}
    rmse_list = [[],[],[]]
    for k in range(2, 102, 2):
        print ("using k = %d" % k)
        p_rmse = []
        u_rmse = []
        hv_rmse = []
        knn = KNNWithMeans(k = k, sim_options = sim_options)
        for trainset,testset in kfold.split(data):
            knn.fit(trainset)
            (p_testset, u_testset,hv_testset) = trim(testset,R)
            
            p_pred = knn.test(p_testset)
            u_pred = knn.test(u_testset)
            hv_pred = knn.test(hv_testset)
            
            p_rmse.append(accuracy.rmse(p_pred))
            u_rmse.append(accuracy.rmse(u_pred))
            hv_rmse.append(accuracy.rmse(hv_pred))
        rmse_list[0].append(np.mean(p_rmse))
        rmse_list[1].append(np.mean(u_rmse))
        rmse_list[2].append(np.mean(hv_rmse))
    print ("KNN with trim is finished!!")
    return rmse_list

def train_nmf(data):
    rmse = []
    mae = []
    sim_options = {'name': 'pearson'}
    for k in range(2, 52, 2):
        print ("using k = %d" % k)
        nmf = NMF(n_factors = k)
        temp = cross_validate(nmf, data, measures=['RMSE', 'MAE'], cv=10)
        rmse.append(np.mean(temp['test_rmse']))
        mae.append(np.mean(temp['test_mae']))
    print ("k-fold validation finished!")
    return (rmse, mae)

def train_trim_nmf(data,R):
    kfold = KFold(n_splits = 10)
    rmse_list = [[],[],[]]
    for k in range(2, 52, 2):
        print ("using k = %d" % k)
        p_rmse = []
        u_rmse = []
        hv_rmse = []
        nmf = NMF(n_factors = k)
        for trainset,testset in kfold.split(data):
            nmf.fit(trainset)
            (p_testset, u_testset,hv_testset) = trim(testset,R)
            
            p_pred = nmf.test(p_testset)
            u_pred = nmf.test(u_testset)
            hv_pred = nmf.test(hv_testset)
            
            p_rmse.append(accuracy.rmse(p_pred))
            u_rmse.append(accuracy.rmse(u_pred))
            hv_rmse.append(accuracy.rmse(hv_pred))
        rmse_list[0].append(np.mean(p_rmse))
        rmse_list[1].append(np.mean(u_rmse))
        rmse_list[2].append(np.mean(hv_rmse))
    print ("NMF with trim is finished!!")
    return rmse_list

def train_svd(data):
    rmse = []
    mae = []
    sim_options = {'name': 'pearson'}
    for k in range(2, 52, 2):
        print ("using k = %d" % k)
        nmf = SVD(n_factors = k)
        temp = cross_validate(nmf, data, measures=['RMSE', 'MAE'], cv=10)
        rmse.append(np.mean(temp['test_rmse']))
        mae.append(np.mean(temp['test_mae']))
    print ("k-fold validation finished!")
    return (rmse, mae)

def train_trim_svd(data,R):
    kfold = KFold(n_splits = 10)
    rmse_list = [[],[],[]]
    for k in range(2, 52, 2):
        print ("using k = %d" % k)
        p_rmse = []
        u_rmse = []
        hv_rmse = []
        svd = SVD(n_factors = k)
        for trainset,testset in kfold.split(data):
            svd.fit(trainset)
            (p_testset, u_testset,hv_testset) = trim(testset,R)
            
            p_pred = svd.test(p_testset)
            u_pred = svd.test(u_testset)
            hv_pred = svd.test(hv_testset)
            
            p_rmse.append(accuracy.rmse(p_pred))
            u_rmse.append(accuracy.rmse(u_pred))
            hv_rmse.append(accuracy.rmse(hv_pred))
        rmse_list[0].append(np.mean(p_rmse))
        rmse_list[1].append(np.mean(u_rmse))
        rmse_list[2].append(np.mean(hv_rmse))
    print ("SVD with trim is finished!!")
    return rmse_list

def train_naive(data,R):
    kfold = KFold(n_splits = 10)
    ur_mean = np.mean(R,axis = 1)
    rmse = []
    for _,testset in kfold.split(data):
        r_pred = []
        r = []
        for item in testset:
            r_pred.append(ur_mean[int(item[0])-1])
            r.append(item[2])
        rmse.append((np.mean((np.array(r_pred) - np.array(r))**2))**0.5)
    return np.mean(rmse)

def train_trim_naive(data,R):
    kfold = KFold(n_splits = 10)
    ur_mean = np.mean(R,axis = 1)
    rmse = [[],[],[]]
    for _,testset in kfold.split(data):
        r_pred = []
        r = []
        trimmeds = trim(testset,R)
        for i in range(len(trimmeds)):
            trimmed = trimmeds[i]
            r_pred = []
            r = []
            for item in trimmed:
                r_pred.append(ur_mean[int(item[0])-1])
                r.append(item[2])
            rmse[i].append((np.mean((np.array(r_pred) - np.array(r))**2))**0.5)
    return (np.mean(rmse[0]),np.mean(rmse[1]),np.mean(rmse[2]))

def calculate_precision_recall(classifiers,threshold,data):
    kf = KFold(n_splits = 10)
    
    precisions = [[],[],[]]
    recalls = [[],[],[]]   
    for t in range(1,26):
        precision_list = []
        recall_list = []
        for i in range(3):
            classifier = classifiers[i]
            if i == 1:
                print("doing nmf")
            elif i == 2:
                print("doing svd")
            for trainset, testset in kf.split(data):
                pass
            classifier.fit(trainset)
            prediction = classifier.test(testset)
            
            S = dict()
            # user: 88         item: 337        r_ui = 3.50   est = 3.74   {'actual_k': 24, 'was_impossible': False}
            for (uid, mid, r, r_pred, _) in prediction:
                if uid in S:
                    S[uid].append((mid,r,r_pred))
                else:
                    S[uid] = [(mid,r,r_pred)]
                    
            count,p_sum,r_sum = (0,0,0)
            for uid in S:
                if len(S[uid]) >= t:
                    pred_r = S[uid]
                    G = set([x[0] for x in filter(lambda x: x[1] >= threshold, pred_r)])
                    if len(G) != 0:
                        pred_r = sorted(pred_r, key = lambda x : -int(x[2]))
                        S2 = set([x[0] for x in pred_r[: t]])
                        inter = G & S2
                        precision = float(len(inter)) / len(S2)
                        recall = float(len(inter)) / len(G)
                        count += 1
                        p_sum += precision
                        r_sum += recall
            precisions[i].append(p_sum/count)
            recalls[i].append(r_sum/count)
        
            
        
    return precisions, recalls
                    
               
            
            
            
    
    

            
        
    
    
    
    

            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
        
    
    
   