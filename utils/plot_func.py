
import numpy as np
from matplotlib import pyplot as plt
# from abstraction.reduction import PCA_R

def show_hist(data, bin=40,range = (0,1), alpha=0.7, title='results'):
    print(title, 'Average:', np.average(data))
    plt.hist(data, bins=bin,range = range, facecolor="blue", edgecolor="black", alpha=alpha)
    plt.xlabel("interval")
    plt.ylabel("num")
    plt.title(title)
    plt.show()
    
def plot_successRate( results ,accumulated=True, slideLen = 100, title='results'):
    accRes = np.zeros_like(results,dtype=np.float32)
    accRes[0] = results[0]
    for i,item in enumerate(results):
        if i==0:
            accRes[i] = results[i]
        else:
            accRes[i] = accRes[i-1]+results[i]
    x = np.array(np.arange(len(results)))+1
    
    if accumulated:
        plt.plot(x,accRes/x)
    else:
        successRate = np.zeros_like(results,dtype=np.float32)
        for idx,_ in enumerate(results):
            if idx<slideLen:
                successRate[idx] = accRes[idx]/(idx+1.0)
            else:
                successRate[idx] = (accRes[idx]-accRes[idx-slideLen])/(slideLen*1.0)
        plt.plot(x,successRate)
    plt.title(title)
    plt.show()

from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
# sns.set()
# import matplotlib 
# # matplotlib.use('Agg')
# from matplotlib import pyplot as plt
# %matplotlib inline
def plot_scatter(good_states, bad_states, pcaModel, dim):
    fig = plt.figure(figsize=(8,8))
    good_data = pcaModel.transform(good_states)
    bad_data = pcaModel.transform(bad_states)
    if dim==2:
        plt.scatter(good_data[:,0],good_data[:,1],label="good")
        plt.scatter(bad_data[:,0],bad_data[:,1],label="bad")
    elif dim==3:
        ax1 = plt.axes(projection='3d')
        ax1.scatter3D(good_data[:,0],good_data[:,1],good_data[:,2], label="good")
        ax1.scatter3D(bad_data[:,0],bad_data[:,1],bad_data[:,2], label="bad")
    else: assert False,"cannot plot scatters for dim={}".format(dim)
    plt.legend(loc=1)
    plt.show()


import seaborn as sns
sns.set()

def plot_2_distribution(id_dis,id_label,ood_dis,ood_label,title="results",range = (0,1), kde=True):
    sns.set(color_codes=True)
    sns.distplot(id_dis,label = id_label,bins=40,kde=kde,hist_kws={'range':range,'edgecolor':"black",'alpha':0.5})
    sns.distplot(ood_dis,label = ood_label,bins=40,kde=kde,hist_kws={'range':range,'edgecolor':"black",'alpha':0.5})
    plt.legend()
    plt.title(title)
    plt.show() 