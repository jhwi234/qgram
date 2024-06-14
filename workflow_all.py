from graphviz import Digraph

dot = Digraph()

# Define nodes with concise labels and shapes
dot.node('data_prep', 'Data Preparation\n(Create Word Lists)', shape='ellipse')
dot.node('preprocessing', 'Preprocessing\n(Extract, Lowercase,\nSeparate Hyphens,\nRemove Non-qualifying Strings)', shape='box')
dot.node('create_sets', 'Create Training and Test Sets\n(Shuffle, Split, Deduplicate)', shape='box')
dot.node('no_overlap', 'Ensure No Overlap\n(No Same Word in Both Sets)', shape='box')
dot.node('test_set', 'Create Test Set\n(Replace Letters with Underscores)', shape='box')
dot.node('training_set', 'Create Training Set\n(Format Words for KenLM)', shape='box')
dot.node('n_gram_models', 'N-Gram Models\n(Build and Parse)', shape='ellipse')
dot.node('kn_smoothing', 'Kneser-Ney Smoothing\n(Handle Data Sparsity)', shape='ellipse')
dot.node('predict_method', 'Prediction Method\n(Predict Missing Letters)', shape='ellipse')
dot.node('init_predict', 'Initialize Prediction\n(Model, q_range,\nunique_characters)', shape='ellipse')
dot.node('context_extract', 'Context Extraction\n(Left and Right Contexts)', shape='box')
dot.node('format_seq', 'Format Sequences\n(Combine Contexts and Letter)', shape='box')
dot.node('calc_prob', 'Calculate Log Probability\n(Model, Sequence)', shape='box')
dot.node('agg_probs', 'Aggregate Log Probabilities\n(All Candidate Letters)', shape='box')
dot.node('select_top', 'Select Top Predictions\n(Normalize and Rank)', shape='box')
dot.node('evaluate', 'Evaluate Predictions\n(Predictive Accuracy,\nWord Validity)', shape='ellipse')

# Define edges with concise labels
dot.edge('data_prep', 'preprocessing')
dot.edge('preprocessing', 'create_sets')
dot.edge('create_sets', 'no_overlap')
dot.edge('no_overlap', 'test_set')
dot.edge('no_overlap', 'training_set')
dot.edge('training_set', 'n_gram_models')
dot.edge('n_gram_models', 'kn_smoothing')
dot.edge('test_set', 'predict_method')
dot.edge('predict_method', 'init_predict')
dot.edge('init_predict', 'context_extract')
dot.edge('context_extract', 'format_seq')
dot.edge('format_seq', 'calc_prob')
dot.edge('kn_smoothing', 'calc_prob')
dot.edge('calc_prob', 'agg_probs')
dot.edge('agg_probs', 'select_top')
dot.edge('select_top', 'evaluate')

# Render the graph to a high-resolution PNG file
dot.render('methodology_workflow', format='png', view=True)

print("Graphviz render complete. File saved as 'methodology_workflow.png'.")
print("To view the file, open 'methodology_workflow.png' using your preferred image viewer.")
