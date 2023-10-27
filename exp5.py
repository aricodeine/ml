# Sample data for weather and play tennis
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]

# Define a function to calculate Gini impurity
def gini_impurity(labels):
    total_samples = len(labels)
    label_counts = {}
    for label in labels:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    impurity = 1.0
    for label in label_counts:
        prob = label_counts[label] / total_samples
        impurity -= prob ** 2
    return impurity

# Define a function to calculate information gain
def information_gain(data, feature_index, threshold):
    left_labels = [sample[-1] for sample in data if sample[feature_index] == threshold]
    right_labels = [sample[-1] for sample in data if sample[feature_index] != threshold]
    
    left_impurity = gini_impurity(left_labels)
    right_impurity = gini_impurity(right_labels)
    
    total_samples = len(data)
    p_left = len(left_labels) / total_samples
    p_right = len(right_labels) / total_samples
    
    gain = gini_impurity([sample[-1] for sample in data]) - (p_left * left_impurity + p_right * right_impurity)
    return gain

# Define a function to find the best split for the dataset
def find_best_split(data):
    num_features = len(data[0]) - 1
    best_feature = None
    best_threshold = None
    best_gain = 0
    
    for feature_index in range(num_features):
        values = set([sample[feature_index] for sample in data])
        for value in values:
            gain = information_gain(data, feature_index, value)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = value
    
    return best_feature, best_threshold

# Define a function to build a decision tree
def build_tree(data, depth=0, max_depth=None):
    labels = [sample[-1] for sample in data]
    if depth == max_depth or len(set(labels)) == 1:
        return max(set(labels), key=labels.count)
    
    if not data:
        return max(set(labels), key=labels.count)
    
    best_feature, best_threshold = find_best_split(data)
    if best_feature is None:
        return max(set(labels), key=labels.count)
    
    sub_tree = {best_feature: {}}
    
    unique_values = set([sample[best_feature] for sample in data])
    for value in unique_values:
        subset = [sample for sample in data if sample[best_feature] == value]
        sub_tree[best_feature][value] = build_tree(subset, depth + 1, max_depth)
    
    return sub_tree

# Build the decision tree
tree = build_tree(data)

# Define a function to make predictions using the decision tree
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    feature, threshold = list(tree.keys())[0], list(tree.values())[0]
    
    if sample[feature] in threshold:
        return predict(threshold[sample[feature]], sample)
    else:
        return max(threshold, key=threshold.get)

# Example: Make predictions for a new sample
new_sample = ['Sunny', 'Hot', 'High', 'Weak']
prediction = predict(tree, new_sample)
print(f"Prediction for {new_sample}: Play Tennis - {prediction}")
