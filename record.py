import itertools
import subprocess
import numpy as np
from record import * 
import networkx as nx
import matplotlib.pyplot as plt
import re
import scipy.stats
import pandas as pd
import os

import seaborn as sns

def arr2str(arr, delim=''): return ''.join(list(arr))

# def is_similar(s, ref, a='ACTG'):
#     '''
#     Returns True if one change (insertion, deletion, or substitution) results in
#     s being equal to the reference. False otherwise. 
# 
#     Params
#     ------
#     s : str
#     ref : str
#         String against which to compare s. 
#     a : str
#         The alphabet from which to draw possible substitutions. 
#     '''
#     s = np.array(list(s))
#     ref = np.array(list(ref))
#     l = len(ref)
# 
#     for i in range(len(s)):
#         # Deletion
#         if np.all(np.delete(s, i) == ref):
#             return True
#         val = s[i] # Store the original value. 
#         for c in a:
#             # Insertion
#             if np.all(np.insert(s, i, [c])[:l] == ref):
#                 return True
#             # Substitution
#             s[i] = c
#             if np.all(s[:l] == ref):
#                 return True
#             s[i] = val # Restore original value. 
#     return False

# TODO: Maybe allow some degree of error when checking for the palindromic
# sequence.

def get_umi2prot(records):
    '''
    Returns a dictionary mapping each molecule (by UMI) to its corresponding
    protein. 
    '''
    umi2prot = {}
    for r in records:
        umi2prot[r.umi1] = r.prot1
        umi2prot[r.umi2] = r.prot2
    return umi2prot

marker2protein = {
        'GGCTAC':'b-catenin', 
        'ACAGTG':'Smad4',
        'CTTGTA':'pSmad2',
        'CGATGT':'E-cadherin',
        'TGACCA':'Smad2/3'}
# The protein represented by each marker.
proteins =  ['b-catenin', 'Smad4', 'pSmad2', 'E-cadherin', 'Smad2/3']

# Specific regions on primer design. 
s = 'CTATAGTGAGTCGTATTA'
t = 'TGGCGCCA' # Palindromic sequence. 

# NOTE: Maybe there is a cleaner way to do this...
def find_palindrome(seq, t=t):
    '''
    Scan the sequence and find the index where the palindromic sequence starts.
    If the palindromic sequence is not found, then throw an error. 
    '''
    i = 0
    while i < len(seq) - len(t):
        if seq[i:i + len(t)] == t:
            return i
        i += 1
    raise Exception(f'Palindromic sequence {t} not found.')
    # return None 


class Record:
    def __init__(self,
            # Are the R1 and R2 labels the same?
            # label=None
            r1_seq=None, 
            r2_seq=None,
            r1_q=None,
            r2_q=None):
        '''
        Initializes a record object. It is initiated using reads from either end
        of the record (antiparallel). 
        
        Params
        ------
        r1_seq : str
            The DNA record sequence read in from the R1 file.
        r2_seq : str
            The DNA record sequence read in from the R2 file. 
        r1_q : str
            The ASCII-encoded quality score read in from the R1 file. 
        r2_q : str
            The ASCII-encoded quality score read in from the R2 file. 
        '''
        # Make sure all necessary argument are given to create a Record object
        # from two separate reads. 
        try:
            self.r1_seq = r1_seq[:-2] # Trim the newline character.
            self.r1_q = r1_q[:-2]
            self.r2_seq = r2_seq[:-2] # Trim the newline character.
            self.r2_q = r2_q[:-2]
        except TypeError: # Raised if one of the parameters is None.
            raise ValueError('Insufficient arguments were given.')
        
        self.r1_seq = r1_seq
        self.r2_seq = r2_seq

        r1_seq_arr = np.array(list(r1_seq)) # Represent sequence as a Numpy array. 
        r2_seq_arr = np.array(list(r2_seq)) # Represent sequence as a Numpy array. 
            
        # Instead of using the key, try finding the palindromic sequence and
        # looking to the left and right of it. 
        r1_t0 = find_palindrome(r1_seq, t=t)
        r2_t0 = find_palindrome(r2_seq, t=t)

        self.marker1 = arr2str(r1_seq_arr[r1_t0 - 6:r1_t0])
        self.marker2 = arr2str(r2_seq_arr[r2_t0 - 6:r2_t0])
        
        try:
            self.prot1 = marker2protein[self.marker1]
            self.prot2 = marker2protein[self.marker2]
        except KeyError: # Not in the dictionary!
            raise Exception('Unable to detect valid protein marker.')
        
        self.umi1 = arr2str(r1_seq_arr[:25])
        self.umi2 = arr2str(r2_seq_arr[:25])

    def __str__(self): 
        '''
        String representation of a Record.
        '''
        return f' LABEL: {self.label}\nSEQ: {self.seq}'

def get_adjacency_matrix(records): #, allow_err=True):
    '''
    Convert a list of records to an adjacency matrix of interactions.

237     return df
    Params
    ------
    records : lst
        List of Record objects. 
    mode : str
        One of 'prot' and 'umi'. Dictates whether a matrix for single
        molecule interactions or overall protein interactions is created. 
    allow_err : bool
        Whether or not to allow some error between UMIs. 
    '''
    # Get all the UMIs in some fixed order.
    umis = np.unique([r.umi1 for r in records] + [r.umi2 for r in records])
    # Initialize a DataFrame filled with zeros. 
    df = pd.DataFrame(np.zeros((len(umis), len(umis))), columns=umis, index=umis)
    for r in records:
        u, v = r.umi1, r.umi2
#         if allow_err:
#             if is_similar(u + r.r1_seq[25], v):
#                 u = v
#             elif is_similar(v + r.r2_seq[25], u):
#                 v = u
        # Get the indices with which to access the cells in the matrix. 
        df.at[u, v] += 1
        df.at[v, u] += 1
    
    # Remove any rows or columns with all zeros. 
    df = df.loc[(df!=0).any(axis=1)]
    df = df.loc[:, (df!=0).any(axis=0)]
    return df


def get_protein_interaction_data(a, u2p=None):
    '''
    Takes an adjacency matrix as input, and returns a new DataFrame containing
    the number of interactions of each molecule with other molecules of each
    protein type. 
    
    Params
    ------
    a : pd.DataFrame
        An adjacency matrix, the output of the get_adjacency_matrix function. 
    u2p : dict
        A dictionary mapping each UMI to a protein. This is the output of the
        get_u2prot function. 
    '''
    assert u2p is not None
    
    df = pd.DataFrame(columns=['b-catenin', 'Smad4', 'pSmad2', 'E-cadherin', 'Smad2/3'])
    for umi in a.columns:
        # Get a list of the adjacent molecules.
        umis = a.index[np.where(a[umi].values > 0)]
        umis = np.append(umis, umi) # Add the current UMI. 
        # Get a list of all unique interacting proteins. 
        prots = np.unique([u2p[u] for u in umis])
        # NOTE: There is an issue with this method... it does not record the
        # UMIs or measure the quantity of the interactions. 
        new_row = {p:0 for p in ['b-catenin', 'Smad4', 'pSmad2', 'E-cadherin', 'Smad2/3']}
        for p in prots:
            new_row[p] = 1
        df = df.append(new_row, ignore_index=True)
    
    return df

def filter_protein_interaction_data(df, u2p=None, n=3, save=None):
    '''
    Takes a list of records as input, and finds all instances where exactly n
    molecules belonging two different proteins interact (i.e. exist together in
    a record. Returns the information as a pandas DataFrame. 

    Params
    ------
    df : pd.DataFrame
        Should be the output of the get_protein_interaction_data function. 
    '''
    assert u2p is not None

    protein_counts = df.values.sum(axis=1)
    # Use the obtained indices to filter the dataframe.
    df = df.iloc[np.where(protein_counts == n)]
        
    # Need to slightly modify the format of the data output. 
    if not (save is None):
        df.to_csv(save)

    return df
 

# NOTE: Probably would want to add functionality to start at different locations
# in the file. 
def load_records(r1_file, r2_file, num=10000):
    '''
    Load in record data as Record objects from the specified files. Returns a
    list of Record objects. 

    Params
    ------
    r1_file : str
        The name of the file containing the first set of reads. 
    r2_file : str
        The name of the file containing the second set of reads. The reads in
        this file correspond line-by-line to the reads in the r1_file, but they
        are the reverse complements. 
    num : int, None
        Number of records to load from the file. If None, then try to load in
        all records. 
    '''

    records = []
    # File is too big, need to get the number of lines with a bash command. 
    r1_nlines = subprocess.check_output(['wc', '-l', r1_file]).decode('utf-8')
    r2_nlines = subprocess.check_output(['wc', '-l', r2_file]).decode('utf-8')
    r1_nlines = int(r1_nlines.split()[0]) # Actual number is first thing returned. 
    r2_nlines = int(r2_nlines.split()[0]) 
    # Make sure files are the same size. 
    if r1_nlines != r2_nlines:
        raise ValueError('Input files must contain data for the same number of records.')
    nlines = r1_nlines # Should be the same number of lines. 

    if num:
        num = min(num, int(nlines / 3)) # Don't try to grab more than maximum. 
    else:
        num = int(nlines / 3)
    
    print(f'Loading {num} records.')
    with open(r1_file, 'r') as f1:
        with open(r2_file, 'r') as f2:
            for i in range(num):
                # Don't bother paying attention to the labels right now.
                _, s1 = f1.readline(), f1.readline()
                _, s2 = f2.readline(), f2.readline()
                f1.readline() # This is just a + according to FASTQ format. 
                f2.readline() # This is just a + according to FASTQ format. 
                q1 = f1.readline()
                q2 = f2.readline()
                
                try: # If loading the record fails, just give up. 
                    records.append(Record(r1_q=q1, r2_q=q2, r1_seq=s1, r2_seq=s2))
                except:
                    pass
    print(f'{len(records)} records successfully loaded.')
    return records

# Plotting functions
# -------------------------------------------------------------------------------

# def filter_umis(records, threshold=0):
#     '''
#     Returns a list of meaningful UMIs. 
# 
#     Params
#     ------
#     records : list
#         A list of Record objects.
#     threshold : int
#         None by default. If specified, then grab all UMIs which participate in a
#         number of interactions which exceeds the threshold. 
#     '''
#     index, matrix = records2matrix(records, mode='umi')
#     index = np.array(index) # Make sure this is a numpy array. 
#     # Gets the total number of interactions for each molecule. 
#     totals = np.sum(matrix, axis=1)
#     
#     # First, just get the UMIs which meet the threshold. 
#     filter_idxs = totals > threshold
#     umis = index[filter_idxs]
#     data = matrix[filter_idxs, :]
#     
#     if np.max(data) == 0:
#         raise RuntimeWarning('There does not seem to be any detected interactions.')
#     return umis, data
# 

def plot_protein_interaction_hist(df,
        experiments=None,
        target=None, 
        figsize=(40, 40),
        save=None):
    '''
    Creats a histogram using the given data.

    Params
    ------
    df : pd.DataFrame or list of pd.DataFrame
        DataFrame should the the output of get_protein_interaction_data. 
    experiments : list
        Optional. Should be a list of experiment labels for each specified
        DataFrame. 
    figsize : (int, int)
        Figure size. 
    target : str
        A UMI, if you want to look at the data for a single molecule. 
    save : str, None
        None by default. If specified, used as the filename or path under which
        to save the plot.
    '''
    if type(df) == list: # In this case, records is a list of lists. 
        dfs = df
    else:
        dfs = [df]
    if experiments is None:
        experiments = np.arange(len(dfs))

    new_dfs = []
    # Build new datasets to accumulate total interactions. 
    for i, df in enumerate(dfs):
        if target: # If a target UMI is specified, only plot relevant data
            df = df.iloc[target]

        new_df = {}
        new_df['proteins'] = df.columns
        new_df['occurrences'] = df.values.sum(axis=0)
        new_df['exp'] = experiments[i]
        new_dfs.append(pd.DataFrame(new_df))

    df = pd.concat(new_dfs)
    sns.barplot(data=df, x='proteins', y='occurrences', hue='exp')
 
    if not (save is None):
        plt.savefig(save)
    
    plt.show()

def bin_by_count(df):
    '''
    For use in the plot_count_hist function. 
    '''
    new_df = {'counts':[], 'occurrences':[]}
    all_counts = df.values.sum(axis=1)
    for x in range(int(np.max(all_counts)) + 1):
        new_df['counts'].append(x)
        new_df['occurrences'].append(np.count_nonzero(all_counts == x))
    if len(new_df) == 0:
        raise RuntimeError('No interactions detected in given dataset.')
    return pd.DataFrame(new_df)

# Might be worth breaking this plotting code into two steps (a binning step
# prior to the plotting). 
def plot_count_hist(df,
        experiments=None,
        target=None, 
        figsize=(40, 40),
        save=None):
    '''
    Creats a histogram using the given data.

    Params
    ------
    df : pd.DataFrame or list of pd.DataFrame
        DataFrame should the the output of get_adjaceny_matrix 
    experiments : list
        Optional. Should be a list of experiment labels for each specified
        DataFrame. 
    figsize : (int, int)
        Figure size. 
    target : str
        A UMI, if you want to look at the data for a single molecule. 
    save : str, None
        None by default. If specified, used as the filename or path under which
        to save the plot.
    '''
    if type(df) == list: # In this case, records is a list of lists. 
        dfs = df
    else:
        dfs = [df]
    if experiments is None:
        experiments = np.arnge(len(dfs))
    
    new_dfs = []
    # Need to combine the two datasets. Make sure to label each sample. 
    for i, df in enumerate(dfs):
        if target: # If a target UMI is specified, only plot relevant data
            df = df.iloc[df.index.str.fullmatch(target)]
            if len(df) == 0:
                raise RuntimeError('Target UMI not found.')
        new_df = bin_by_count(df)
        new_df['exp'] = experiments[i]
        new_dfs.append(new_df)

    df = pd.concat(new_dfs) # .groupby('counts').sum()
    sns.barplot(data=df, x='counts', y='occurrences', hue='exp')
    
    if save is not None:
        plt.savefig(save)
    
    plt.show()


def get_random_color():
    '''
    Returns a random color in the form of a hexadecimal code. 
    '''
    digits = list('0123456789abcdef')
    num = np.random.choice(digits, size=6)
    return f'#{arr2str(num)}'

def get_random_pos(r, c, a, spread=5):
    '''
    '''
    cx0, cy0 = c
    cx1, cy1 = cx0 + np.cos(a)*r, cy0 + np.sin(a)*r
    
    # Random radius for the new point from the subgroup center. 
    r1 = spread * np.random.uniform()
    # Random angle for the new point from the subgroup center. 
    a1 = np.random.uniform(low=0, high=2*np.pi)
    x, y = cx1 + np.cos(a1)*r1, cy1 + np.sin(a1)*r1
    try: # What the fuck
        return (x[0], y[0])
    except:
        return (x, y)


# Turns out I need to plot it manually because networkx plotting utilities suck
# ass. 
# TODO: Add some kind of filter with the UMI mode so that we are only plotting
# interactions of molecules which are meaningful. 
def plot_interaction_network(df, u2p=None,
        labels=True,
        figsize=(50, 50), 
        spread=5, 
        r=20, 
        c=(0, 0), 
        colors=None, 
        **kwargs):
    '''
    Plots the interactions between proteins or molecules, depending on the
    specified mode. Each node represents one side of an interaction measurement
    (record). 

    Params
    ------
    df : pd.DataFrame
        The output of get_adjacency_matrix.
    u2p : dict
        A dictionary mapping each UMI to a protein. 
    mode : str
        One of 'prot' or 'umi'. Types of interactions to visualize.
    figsize : (float, float)
        Figure size. 
    spread : float
        The amount of spacing between points in the same cluster group. 
    '''
    if u2p is None:
        raise ValueError('UMI-to-protein dictionary must be specified.')

    umis = list(df.index)
    proteins = np.array(['b-catenin', 'Smad4', 'pSmad2', 'E-cadherin', 'Smad2/3'])
    n = len(proteins)

    # Generate a random color for each group. 
    if colors is None:
        colors = np.array([get_random_color() for p in proteins])

    # Divide the circle into sectors according to the number of groups. 
    G = nx.Graph()
    angles = np.linspace(0, 2*np.pi, n + 1)[:-1]
    
    pos, colors_by_node = [], []
    count = 0
    for u, v in [(u, v) for u in umis for v in umis]:
        # Add each node with their custom positions.
        if df.at[u, v] > 0: # If there is an interaction. 
            pos.append(get_random_pos(r, c, angles[proteins == u2p[u]], spread=spread))
            colors_by_node.append(colors[proteins == u2p[u]][0])
            G.add_node(count)

            pos.append(get_random_pos(r, c, angles[proteins == u2p[v]], spread=spread))
            G.add_node(count + 1)
            colors_by_node.append(colors[proteins == u2p[v]][0])
        
            G.add_edge(count, count + 1) # Add an edge for the record. 
            count += 2 # Update the count. 
    
    plt.figure(figsize=figsize)
    nx.draw(G, pos=pos, 
            node_color=colors_by_node, 
            node_size=kwargs.get('node_size', 1),
            edge_color=kwargs.get('edge_color', 'gray'),
            width=kwargs.get('width', 0.5))
    
    label_dist = 5 # Distance of label from cluster.
    for i in range(n):
        label_pos = get_random_pos(spread + r + label_dist, c, angles[i], spread=0)
        plt.annotate(proteins[i], label_pos)
        
    plt.xlim([c[0] - (r + 2*spread + label_dist), c[0] + (r + 2*spread + label_dist)])
    plt.ylim([c[1] - (r + 2*spread + label_dist), c[1] + (r + 2*spread + label_dist)])
    plt.show()


