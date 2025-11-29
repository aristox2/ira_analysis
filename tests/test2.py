import networkx as nx

G = nx.Graph()
G.add_edge('A', 'B')  # Adds an edge between nodes 'A' and 'B'
G.add_edge('B', 'C', weight=0.5) # Adds a weighted edge