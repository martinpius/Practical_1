import pandas as pd 
import numpy as np 
from typing import List, Tuple, Dict 
import matplotlib.pyplot as plt 

data: Dict[str, List] = {
    'id': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
    "surv1": [6,6,6,7,10,13,16,22,23,6,9,10,11,17,19,20,25,32,32,34,35],
    "delta1": [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    "surv2": [1,2,3,4,5,8,8,11,12,15,17,12,23,1,2,4,5,8,8,11,12],
    "delta2": [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]}


df = pd.DataFrame(data = data)
print("\n----------------------------")
print(df) # Display the dataset: 
#In case of large dataset [print(df.shape), df.head(10)] can be used

def compute_summary_stats(
        df: pd.DataFrame = df) -> Tuple[float, float, float,
                                        float, float, float, float]:

    '''
    @Martin:
    ---------------
    
    This method compute the summary descriptive stats for the given 
    survival data:

    -----------------
    Arguments:
    df: pd.DataFrame => Pandas dataframe. Can be any data format ie csv etc
    -----------------
    Returns
    -------

    -avg_1: average survival time for group 1 = sum(surv_1)/N
    -avg_2: average survival time for group 2 = sum(surv_2)/N
    avg_h1: average hazard for group 1 = # events / sum(surv_1)
    avg_h2: average hazard for group 2 = # events / sum(surv_2)
    hr: hazard ratio = avg_h1 / avg_h2

    '''
    
    # compute average survival time for both groups

    avg_1, avg_2 = df.surv1.mean(), df.surv2.mean()

    # compute average hazard for both groups

    avg_h1 = df[df["delta1"]==1]["surv1"].count()/df["surv1"].sum()
    avg_h2 = df[df["delta2"]==1]["surv2"].count()/df["surv2"].sum()
    
    # compute the hazards ratio:

    hr = float(avg_h1 / avg_h2)

    # compute the percentage of censored subjects:

    fs1 =  df[df["delta1"] == 0]['surv1'].count() /len(df)*100
    fs2 =  df[df["delta2"] == 0]['surv2'].count() /len(df)*100

    return (avg_1, avg_2, avg_h1, avg_h2, hr, fs1, fs2)

# call the above function to execute
res = compute_summary_stats()

avg_1, avg_2, avg_h1, avg_h2, hr, fs1, fs2 = res # unpacking the results

# print out the results

print("\n----------------------------")
print(f"\n>>>> The average survival for group 1: {avg_1:.3f}\
      \n>>>> The average survival for group 2: {avg_2:.3f}")
print("\n----------------------------")

print(f" >>>> Average hazard for group1: {avg_h1:.4f},\
      \n >>>> Average hazard for group2: {avg_h2:.4f}")
print("\n----------------------------")
print(f">>>> The hazard ratio: {avg_h1 / avg_h2:.4f}\n")
 

def surv_distributions_plot(df: pd.DataFrame) -> None:

    '''
    @ Martin:
    This method plot the histogram of raw survival times for both groups

    argument:
    -----------
    df: pd.DataFrame, can also be dataset in any other format like csv
    '''

    plt.figure(figsize = (10,12))
    df['surv1'].hist(label = "group1 survival time")
    df['surv2'].hist(label = "group 2 survival time")
    plt.legend(loc = "best")
    plt.xlabel("survival time in 'years'")
    plt.ylabel("counts")
    plt.title("Raw survival times distribution")
    plt.grid(False)
    plt.show()

def events_distribution_plot(df: pd.DataFrame) -> None:

    '''
     @ Martin:
    This method plot the histogram of raw event times for both groups

    argument:
    -----------
    df: pd.DataFrame, can also be dataset in any other format like csv
    '''
    
    plt.figure(figsize = (10,12))
    df[df["delta1"] == 1]['surv1'].hist(label = "group 1 events distribution")
    df[df["delta2"] == 1]['surv2'].hist(label = "group 2 events distribution")
    plt.legend(loc = "best")
    plt.xlabel("Event times")
    plt.ylabel("counts")
    plt.title("Distribution of event times for the two groups")
    plt.grid(False)
    plt.show()
    
def censore_distribution_plot(df: pd.DataFrame) -> None:

    '''
     @ Martin:
    This method plot the histogram of censored subjects in both groups

    argument:
    -----------
    df: pd.DataFrame, can also be dataset in any other format like csv
    '''
    plt.figure(figsize = (10,12))
    df[df["delta1"]==0]['surv1'].hist(label = "censored group1")
    df[df["delta2"]==0]['surv2'].hist(label = "censored group2")
    plt.legend(loc = "best")
    plt.xlabel("Censoreship times")
    plt.ylabel("counts")
    plt.title("Distribution of censored subjects in both groups")
    plt.grid(False)  
    plt.show()


def naive_estimator(t: float, df: pd.DataFrame) -> List[float]:
    """
    @Martin
    -----------
    Return naive estimate for S(t), the probability
    of surviving past time t. Given by number
    of cases who survived past time t divided by the
    number of cases who weren't censored before time t.
    
    Arguments:
        t (float): Specified time stamp
        df: pd.DataFrame: Survival data.

    Returns:
    ----------
        S_t (float): List of probabilities: estimator for survival function evaluated at t.
    """   
    X = sum(df['surv1'] > t)
    M = sum( (df['surv1'] > t) | (df['delta1'] == 1) )
    S_t = X / M
    
    return S_t  

# run here to excute:

# max_time = df.surv1.max() # grab the max survival time
# x = range(0, max_time+1) # set the time coordinates
# y = np.zeros(len(x)) # to pack the survival probs
# for i, t in enumerate(x): # iterate over all time steps
#     y[i] = naive_estimator(t, df) # estimate the survival prob @ every time point

# # ploting the naive curve  
# plt.plot(x, y)
# plt.title("Naive Survival Estimate")
# plt.xlabel("Time")
# plt.ylabel("Estimated cumulative survival rate")
# plt.show()

def KM(df: pd.DataFrame, group: str = "g1") -> Tuple[List[float], List[float]]:

    """
    @ Martin
    -------------

    This method compute KM estimate evaluated at every distinct
    time (event or censored) recorded in the dataset.

    We use the product limit theory/formular: S_hat = Product(1 - mi/ri)

    where mi: number of events at time t_i, and ri is the risk set at that time

    Arguments:
    -----------
    df: pd.DataFrame ==> Survival data
    group: str ==> group "g1" for group 1, "g2" for group 2

    Return
    ----------
    S: List of survival probabilities ==> KM estimates
    events: List of survival times

    """
    event_times = [0] # In the begining we initialize survival function to 1 (at time 0, p = 1) 
    p = 1.0
    S = [p]
    
    # get collection of unique observed event times
    obst1 = df.surv1.unique()
    obst2 = df.surv2.unique()
  
    # sort event times in ascending order of magnitude 
    obst1 = sorted(obst1)
    obst2 = sorted(obst2)
    
    if group == "g1":
        # iterate through event times
        for t in obst1:      
    
            # compute n_t, number of people who survive at least to time t
            n_t = len(df[df.surv1 >= t])
    
            # compute d_t, number of people who die at time t
            d_t = len(df[(df.surv1 == t) & (df.delta1 == 1)])
    
            # update p
            p = p*(1 - float(d_t)/n_t)
    
            # update S and event_times 
            event_times.append(t)
            S.append(p)
    else:
        for t in obst2:      
    
            # compute n_t, number of people who survive at least to time t
            n_t = len(df[df.surv2 >= t])
    
            # compute d_t, number of people who die at time t
            d_t = len(df[(df.surv2 == t) & (df.delta2 == 1)])
    
            # update p
            p = p*(1 - float(d_t)/n_t)
    
            # update S and event_times 
            event_times.append(t)
            S.append(p)

  
    return (event_times, S)

# Run and plot the estimated KM
# max_time1 = df.surv1.max() # for group 1
# max_time2 = df.surv2.max() # for group 2

# x1 = range(0, max_time1 + 1)
# y1 = np.zeros(len(x1)) 

# x2 = range(0, max_time2 + 1)
# y2 = np.zeros(len(x2))

# for i, t in enumerate(x1):
#     y1[i] = naive_estimator(t, df)

# for i, t in enumerate(x2):
#     y2[i] = naive_estimator(t, df)

# plt.figure(figsize = (10, 12))    
# plt.plot(x1, y1, label="Naive estimator group 1")
# plt.plot(x2, y2, label = "Naive estimator group 2")
# plt.legend(loc = "best")
# plt.xlabel("Survival time")
# plt.ylabel("Survival probability")
#plt.title("Naive estimator for for the two groups")
# plt.show()

res1 = KM(df, group = "g1")
x1, y1 = res1
res2 = KM(df, group = "g2")
x2, y2 = res2
print(f"\n------------------------")
print(f">>>> KM surv probs group 1: {y1}\n>>>> KM surv probs group 2: {y2}")
print(f"\n------------------------")

plt.figure(figsize = (10, 12))
plt.step(x1, y1, label = "group 1")
plt.step(x2, y2, label = "group 2")
plt.xlabel("Survival time")
plt.ylabel("Survival probability estimate (KM)")
plt.legend()
plt.title("Kaplan Mier Plot for the two groups")
plt.show()





