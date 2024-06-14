from graphviz import Digraph

dot = Digraph()

# Define nodes with concise labels and shapes
dot.node('test_word', 'Test Word\n<com_uter>', shape='ellipse', style='filled')
dot.node('extract_contexts', 'Extract Contexts\n(Left: com, Right: uter)', shape='ellipse', style='filled')
dot.node('possible_letters', 'Insert Possible Letters\n(a-z)', shape='ellipse', style='filled')
dot.node('calc_log_prob', 'Calculate Log Probability\n(Model, com + [a-z] + uter)', shape='ellipse', style='filled')
dot.node('agg_probs', 'Aggregate and Rank\nLog Probabilities', shape='ellipse', style='filled')
dot.node('select_predictions', 'Select Top 3\nLetters', shape='ellipse', style='filled')
dot.node('predict_single', 'Predict Missing\nLetter\n(p)', shape='ellipse', style='filled')

# Define edges with concise labels
dot.edge('test_word', 'extract_contexts', label='Start Prediction')
dot.edge('extract_contexts', 'possible_letters', label='Generate Candidates')
dot.edge('possible_letters', 'calc_log_prob', label='Query Model')
dot.edge('calc_log_prob', 'agg_probs', label='Aggregate Scores')
dot.edge('agg_probs', 'select_predictions', label='Rank Candidates')
dot.edge('select_predictions', 'predict_single', label='Final Selection')

# Set graph attributes for better layout
dot.attr(rankdir='TB', size='10,7')

# Render the graph to a high-resolution PNG file
dot.render('prediction_method_workflow', format='png', view=True)

print("Graphviz render complete. File saved as 'prediction_method_workflow.png'.")
print("To view the file, open 'prediction_method_workflow.png' using your preferred image viewer.")
