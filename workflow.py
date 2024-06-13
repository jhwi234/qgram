from graphviz import Digraph

dot = Digraph()

# Define nodes with shapes and concise labels
dot.node('init', 'Init\nModel', shape='ellipse')
dot.node('context_sensitive', 'Context-Sensitive\nPrediction', shape='ellipse')
dot.node('get_log_probs', 'Compute Log\nProbabilities', shape='ellipse')
dot.node('extract_contexts', 'Extract\nContexts', shape='ellipse')
dot.node('format_sequence', 'Format\nSequence', shape='ellipse')
dot.node('calc_log_prob', 'Calc Log\nProbability', shape='ellipse')
dot.node('select_predictions', 'Select Top\nPredictions', shape='ellipse')
dot.node('predict_single', 'Predict\nLetter', shape='ellipse')

# Define edges with concise labels
dot.edge('init', 'context_sensitive', label='Start')
dot.edge('context_sensitive', 'get_log_probs', label='For each letter')
dot.edge('get_log_probs', 'extract_contexts', label='Loop: q_range')
dot.edge('extract_contexts', 'format_sequence', label='Extract')
dot.edge('format_sequence', 'calc_log_prob', label='Format')
dot.edge('calc_log_prob', 'get_log_probs', label='Calculate')
dot.edge('get_log_probs', 'select_predictions', label='Aggregate')
dot.edge('select_predictions', 'predict_single', label='Select')

# Adjust graph attributes for better symmetry
dot.attr(rankdir='TB', size='8,5')

# Cluster subgraph for symmetrical layout
with dot.subgraph(name='cluster_context_processing') as c:
    c.attr(style='invis')
    c.node('extract_contexts')
    c.node('format_sequence')
    c.node('calc_log_prob')
    c.edge('extract_contexts', 'format_sequence', style='invis')
    c.edge('format_sequence', 'calc_log_prob', style='invis')

# Render the graph to a PNG file in the current directory
dot.render('predictions_workflow_detailed_clean', format='png', view=True)

print("Graphviz render complete. File saved as 'predictions_workflow_detailed_clean.png'.")
print("To view the file, open 'predictions_workflow_detailed_clean.png' using your preferred image viewer.")
