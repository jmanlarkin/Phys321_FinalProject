import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats.distributions import chi2
from scipy.stats import mode
from os import listdir
from tqdm import tqdm_notebook as tqdm
import matplotlib as mpl

votinghistory_full = np.loadtxt('1976-2020-president.csv', delimiter=',', skiprows=1, usecols=(0, 1, 10, 11, 14), 
                                dtype=[('year', int), ('state', 'U25'), ('votes', int), ('total_votes', int), 
                                       ('party', 'U20')])

states = np.unique(votinghistory_full['state'])

votes_2016 = votinghistory_full[(votinghistory_full['year'] == 2016) & 
                                    (votinghistory_full['party'] == 'DEMOCRAT')]['total_votes']

#Number of Electoral College votes allocated to each state
college_weights = [9, 3, 11, 6, 55, 9, 7, 3, 3, 29, 16, 4, 4, 20, 11, 6, 6, 
                   8, 8, 4, 10, 11, 16, 10, 6, 10, 3, 5, 6, 4, 14, 5, 29, 15, 
                   3, 18, 7, 7, 20, 4, 9, 3, 11, 38, 6, 3, 13, 12, 5, 10, 3]

#Process for creating a monte carlo sample
def draw(mean, cov, states = states):
    probs = np.random.multivariate_normal(mean, cov)
    college_win = []
    
    #from drawn multivariate probabilites, flips weighted coin to determine which states are won
    for i in range(len(states)):
        prob = probs[i]
        p = [1 - prob, prob]
        if (prob <= 0):
            college_win.append(0)
            continue
        if (prob >= 1):
            college_win.append(1)
            continue
        vote_boolean = np.random.choice([0, 1], replace = True, p = p)
        college_win.append(vote_boolean) 
        
    #Popular Vote Results
    votes = (probs * votes_2016).astype(int)
    popular_vote = np.sum(votes)
    popular_voteshare = popular_vote / np.sum(votes_2016)
    
    #Electoral College Results
    college_win *= np.asarray(college_weights)
    college_votes = np.sum(college_win)
    
    return [college_votes, popular_voteshare]

#Run montecarlo sampling with above method for given number of runs
#tqdm wrapped around iterable range allows for convenient progress bar
def montecarlo(runs, mean, cov):
    electoral_college_votes = []
    popular_votes = []
    for i in tqdm(range(runs)):
        college_votes, popular_voteshare = draw(mean, cov)
        electoral_college_votes.append(college_votes)
        popular_votes.append(popular_voteshare)
    
    electoral_college_votes = np.asarray(electoral_college_votes)
    popular_votes = np.asarray(popular_votes)
    
    #Electoral College
    electoral_win = len(electoral_college_votes[electoral_college_votes > 270]) / runs
    electoral_tie = len(electoral_college_votes[electoral_college_votes == 270]) / runs
    electoral_mean = np.mean(electoral_college_votes)
    
    #Popular Vote
    popular_win = len(popular_votes[popular_votes > 0.5]) / runs
    popular_landslide_win = len(popular_votes[popular_votes >= 0.55]) / runs
    popular_landslide_loss = len(popular_votes[popular_votes <= 0.45]) / runs
    popular_mean = np.mean(popular_votes)
    
    return [electoral_college_votes, electoral_win, electoral_tie, electoral_mean, popular_votes, 
            popular_win, popular_landslide_win, popular_landslide_loss, popular_mean]