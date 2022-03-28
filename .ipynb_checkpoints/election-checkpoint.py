import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats.distributions import chi2
from os import listdir
from tqdm import tqdm_notebook as tqdm
import matplotlib as mpl

new_rc_params = {'text.usetex': False,
    "svg.fonttype": 'none'
    }
mpl.rcParams.update(new_rc_params)

States = sorted(['Alaska', 'Alabama', 'Arkansas', 'Arizona', 'California', 
          'Colorado', 'Connecticut', 'District of Columbia', 'Delaware', 'Florida', 
          'Georgia', 'Hawaii', 'Iowa', 'Idaho', 'Illinois', 'Indiana', 'Kansas', 
          'Kentucky', 'Louisiana', 'Massachusetts', 'Maryland', 'Maine', 'Michigan', 'Minnesota', 
          'Missouri', 'Mississippi', 'Montana', 'North Carolina', 'North Dakota', 'Nebraska', 
          'New Hampshire', 'New Jersey', 'New Mexico', 'Nevada', 'New York', 'Ohio', 'Oklahoma', 
          'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 
          'Tennessee', 'Texas', 'Utah', 'Virginia', 'Vermont', 'Washington', 
          'Wisconsin', 'West Virginia', 'Wyoming'])

Voteshare = [9, 3, 11, 6, 55, 9, 7, 3, 3, 29, 16, 4, 4, 20, 11, 6, 6, 
            8, 8, 4, 10, 11, 16, 10, 6, 10, 3, 5, 6, 4, 14, 5, 29, 15, 
            3, 18, 7, 7, 20, 4, 9, 3, 11, 38, 6, 3, 13, 12, 5, 10, 3]

def draw(probs, States = States, returnvotes = False):
    votes = []
    for i in range(len(States)):
        prob = probs[i]
        p = [1 - prob, prob]
        vote = np.random.choice([0, 1], replace = True, p = p)
        votes.append(vote)
    votes *= np.asarray(Voteshare)
    if returnvotes == True:
        return np.sum(votes), votes
    return np.sum(votes)

def montecarlo(runs, probs):
    out = []
    for i in tqdm(range(runs)):
        run = draw(probs)
        out.append(run)
    return out
