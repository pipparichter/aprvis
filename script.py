from record import *

e_records = load_records('./data/e_r1_data.fq', './data/e_r2_data.fq')
m_records = load_records('./data/m_r1_data.fq', './data/m_r2_data.fq')


# Build the dictionary mapping UMIs to proteins. 
e_u2p = get_umi2prot(e_records)

# Build the adjacency matrix. 
e_a = get_adjacency_matrix(e_records)

# Get protein interaction data. 
e_pi = get_protein_interaction_data(e_a, u2p=e_u2p)

# Filter and save protein interaction data. The `save` parameter, if specified, writes the resulting matrix
# to a CSV file. 
e_pi_filtered = filter_protein_interaction_data(e_pi, n=1, save='./e_pi_data.csv')



# Build the dictionary mapping UMIs to proteins.
m_u2p = get_umi2prot(m_records)

# Build the adjacency matrix.
m_a = get_adjacency_matrix(m_records)

# Get protein interaction data.
m_pi = get_protein_interaction_data(m_a, u2p=m_u2p)

# Filter and save protein interaction data. The `save` parameter, if specified, writes the resulting matrix
# to a CSV file.
m_pi_filtered = filter_protein_interaction_data(m_pi, n=1, save='./m_pi_data.csv')


# One mor more adjacency matrices can be given as input.
plot_count_hist([e_a, m_a], experiments=['E', 'M'], save='a_plot.png')
# One more more adjacency matrices can be given as input.
plot_protein_interaction_hist([e_pi, m_pi], experiments=['E', 'M'], save='./another_plot.png')
