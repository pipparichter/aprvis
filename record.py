import itertools
import subprocess
import numpy as np
from record import * 
import networkx as nx
import matplotlib.pyplot as plt
import re
import scipy.stats

# TODO: Maybe allow some degree of error when checking for the palindromic
# sequence.

# The protein represented by each marker.
# NOTE: This map doesn't seem to be working?
marker2protein = {'CGATGT':'b-catenin', 'TGACCA':'Smad4', 'ACATCG':'pSmad2',
        'GGCTAC':'E-cadherin', 'CTTGTA':'Smad2/3'}

# H, D : Single-molecule UMI
# X : Marker barcode
# N : Primer UMI
# GGCGCCA : Palindrome
key = np.array(list('HHHHHHHHHHHHHHHHHHHHHHHHHXXXXXXTGGCGCCAYYYYYYDDDDDDDDDDDDDDDDDDDDDDDDDACTATAGTGAGTCGTATTANNNNNNNN'))
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

def arr2str(arr, delim=''): return ''.join(list(arr))

class Record:
    def __init__(self, label, seq, q):
        '''
        Initializes a record object. 
        
        Params
        ------
        label : str
            Label of the record per the FASTQ convention. 
        seq : str
            The DNA record sequence. 
        q : str
            The ASCII-encoded quality score. 
        '''
        self.label = label[:-2]
        self.seq = seq[:-2] # Trim the newline character.
        self.q = q[:-2]

        seq_arr = np.array(list(seq)) # Represent sequence as a Numpy array. 
        
        # Instead of using the key, try finding the palindromic sequence and
        # looking to the left and right of it. 
        t0 = find_palindrome(seq, t=t)
        self.marker1 = arr2str(seq_arr[t0 - 6:t0])
        self.marker2 = arr2str(seq_arr[t0 + len(t):t0 + len(t) + 6])
        try:
            self.prot1 = marker2protein[self.marker1]
            self.prot2 = marker2protein[self.marker2]
        except KeyError: # Not in the dictionary!
            raise Exception('Unable to detect valid protein marker.')

        self.umi1 = arr2str(seq_arr[np.where(key == 'H')])
        self.umi2 = arr2str(seq_arr[np.where(key == 'D')])
        self.primer_umi = arr2str(seq_arr[np.where(key == 'N')])

        self.seq_arr = seq_arr

    def __str__(self): 
        '''
        String representation of a Record.
        '''
        return f' LABEL: {self.label}\nSEQ: {self.seq}'

def records2matrix(records, mode='prot'):
    '''
    Convert a list of records to an adjacency matrix of interactions.

    Params
    ------
    records : lst
        List of Record objects. 
    mode : str
        One of 'prot' and 'umi'. Dictates whether a matrix for single
        molecule interactions or overall protein interactions is created. 
    '''
    assert mode in ['prot', 'umi']
    index = np.unique([getattr(r, f'{mode}1') for r in records] + [getattr(r,
        f'{mode}2') for r in records])

    matrix = np.zeros((len(index), len(index)))
    for r in records:
        u, v = getattr(r, f'{mode}1'), getattr(r, f'{mode}2')
        # Get the indices with which to access the cells in the matrix. 
        i1, i2 = np.where(index == u)[0], np.where(index == v)[0]
        matrix[i1, i2] += 1
        matrix[i2, i1] += 1
    
    return index, matrix.astype(int)

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


# NOTE: Probably would want to add functionality to start at different locations
# in the file. 
def load_records(filename, num=10000):
    '''
    Load in record data as Record objects from the specified file. Returns a
    list of Record objects. 

    Params
    ------
    filename : str
        Filepath to where the record data is stored. 
    num : int, None
        Number of records to load from the file. If None, then try to load in
        all records. 
    '''
    records = []
    # File is too big, need to get the number of lines with a bash command. 
    nlines = subprocess.check_output(['wc', '-l', filename]).decode('utf-8')
    nlines = int(nlines.split()[0]) # Actual number is first thing returned. 
    if num:
        num = min(num, int(nlines / 3)) # Don't try to grab more than maximum. 
    else:
        num = int(nlines / 3)
    
    print(f'Loading {num} records from {filename}.')
    with open(filename, 'r') as f:
        for i in range(num):
            l, s = f.readline(), f.readline()
            f.readline() # This is just a + according to FASTQ format. 
            q = f.readline()
            try: # If loading the record fails, just give up. 
                records.append(Record(l, s, q))
            except:
                pass
    print(f'{len(records)} records successfully loaded.')
    return records

# Plotting functions
# -------------------------------------------------------------------------------

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
    
    return (x, y)

def filter_umis(records, 
        num=None, 
        threshold=None, 
        unique_proteins=None):
    '''
    Returns a list of meaningful UMIs. 

    Params
    ------
    records : lst
        A list of Record objects.
    num : int
        None by default. If specified, then grab the top n UMIs which
        participate in the most interactions. 
    count : float
        None by default. If specified, grabs a list of all UMIs with a total
        number of interactions which exceeds the threshold. 
    unique_proteins : int
        None by default. Allows you to filter for UMIs which participate in
        interactions with a specified number of different proteins (i.e. 
        not the same protein twice) are included in the threshold count. 
    '''
    if np.count_nonzero([threshold, unique_proteins, num]) > 1: 
        raise RuntimeWarning('More than one filter method was specified.')
    if not (threshold or unique_proteins or num):
        raise ValueError('At least one filter method needs to be specified.')

    index, matrix = records2matrix(records, mode='umi')
    index = np.array(index) # Make sure this is a numpy array. 
    # Gets the total number of interactions for each molecule. 
    totals = np.sum(matrix, axis=1)
    
    if unique_proteins: # If not None...
        u2p = get_umi2prot(records)
        p = np.array([u2p[u] for u in index])
        f = lambda a : len(np.unique(p[np.where(a > 0)])) >= unique_proteins
        # Get the indices of UMIs which have more than a specified number of
        # interactions with unique proteins. 
        filter_idxs = np.apply_along_axis(f, 1, matrix)
    elif threshold:
        # First, just get the UMIs which meet the threshold. 
        filter_idxs = totals > threshold
    else: # Argsort sorts in ascending order. 
        filter_idxs = np.argsort(totals)[-num:] 
    
    umis = index[filter_idxs]
    data = matrix[filter_idxs, :]
    
    if np.max(data) == 0:
        raise RuntimeWarning('There does not seem to be any detected interactions.')
    return umis, data

# TODO: We want to find the UMIs for the molecules which interact with at least
# two other molecules belonging to a different protein group. 

def plot_hist(records, 
        target=None, 
        figsize=(40, 40), 
        mode=None, 
        normalize=True, 
        errorbars=True):
    '''
    Creats a histogram using the record data, which is basically just a
    histogram of the edges of the interaction plot. 

    Params
    ------
    target : str, None
        If target is None, then no filtering is done; basically just make a
        histogram of edges in the graph. 
    records : lst
        A list of record objects.
    figsize : (float, float)
        Figure size. 
    mode : str, None
        Indicates the type of the target, either a UMI or a protein. Must be
        specified if a target is specified. 
    normalize : bool
        Whether or not to normalize counts (i.e. make it a real probability
        distribution). 
    '''
    umi2prot = get_umi2prot(records)
    umis, matrix = records2matrix(records, mode='umi')

    if target:
        assert mode in ['prot', 'umi']
        if mode == 'prot':    
            index = np.array([umi2prot[umi] for umi in umis])
        elif mode == 'umi':
            index = umis
        matrix = matrix[np.where(index == target)[0], :]
        matrix = matrix[:, np.where(index != target)[0]]
    
    data = np.sum(matrix, axis=1).astype(int) # Sum up all interactions. 
    xvals = np.arange(0, np.max(data) + 1)
    counts = np.array([np.count_nonzero(data == x) for x in xvals])
    
    if normalize: # Make sure to normalize the error. 
        counts = counts / np.max(counts)
    
    plt.figure(figsize=figsize)
    plt.bar(xvals, counts)
    plt.xticks(ticks=xvals, labels=xvals.astype(str))
    
    if not target:
        plt.title('Interaction distribution')
    else:
        if mode == 'prot': # Target is a protein. 
            plt.title(f'Interaction distribution for {target} protein')
        else: # Target is a UMI. 
            plt.title(f'Interaction distribution for molecule {target}')
    plt.show()
    

# Turns out I need to plot it manually because networkx plotting utilities suck
# ass. 
# TODO: Add some kind of filter with the UMI mode so that we are only plotting
# interactions of molecules which are meaningful. 
def plot_interactions(records,
        mode='prot',
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
    records : lst
        A list of Record objects. 
    mode : str
        One of 'prot' or 'umi'. Types of interactions to visualize.
    labels : bool
        Whether or not to include group labels.
    figsize : (float, float)
        Figure size. 
    spread : float
        The amount of spacing between points in the same cluster group. 
    '''

    index, matrix = records2matrix(records, mode=mode)
    n = len(index)
    # Generate a random color for each group. 
    if not colors:
        colors = [get_random_color() for i in range(n)]

    # Divide the circle into sectors according to the number of groups. 
    G = nx.Graph()
    angles = np.linspace(0, 2*np.pi, n + 1)[:-1]
    
    pos, colors_by_node = [], []
    count = 0
    for u, v in [(u, v) for u in range(n) for v in range(n)]:
        for i in range(matrix[u, v]):
            
            # Add each node with their custom positions. 
            pos.append(get_random_pos(r, c, angles[u], spread=spread))
            colors_by_node.append(colors[u])
            G.add_node(count, label=index[u])

            pos.append(get_random_pos(r, c, angles[v], spread=spread))
            G.add_node(count + 1, label=index[v])
            colors_by_node.append(colors[v])
            
            G.add_edge(count, count + 1) # Add an edge for the record. 
            count += 2 # Update the count. 
    
    plt.figure(figsize=figsize)
    nx.draw(G, pos=pos, 
            node_color=colors_by_node, 
            node_size=kwargs.get('node_size', 1),
            edge_color=kwargs.get('edge_color', 'gray'),
            width=kwargs.get('width', 0.5))
    
    label_dist = 5 # Distance of label from cluster.
    if labels:
        # Add labels to the groups on the graph. 
        for i in range(n):
            label_pos = get_random_pos(spread + r + label_dist, c, angles[i], spread=0)
            plt.annotate(index[i], label_pos)
        
    plt.xlim([c[0] - (r + 2*spread + label_dist), c[0] + (r + 2*spread + label_dist)])
    plt.ylim([c[1] - (r + 2*spread + label_dist), c[1] + (r + 2*spread + label_dist)])
    plt.show()


