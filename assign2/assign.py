import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from collections import Counter

# Load the dataset
data = pd.read_csv("BankNote_Authentication.csv")

# Problem 1
# 1. Experiment with a fixed train_test split ratio: Use 25% of the samples for training and the rest for testing.
# a. Run this experiment five times and notice the impact of different random splits of the data into training and test sets.
# b. Print the sizes and accuracies of these trees in each experiment.

for i in range(5):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1), data['class'], test_size=0.25, random_state=i)
    
    # Create Decision Tree Classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = clf.score(X_test, y_test)
    tree_size = clf.tree_.node_count
    
    print(f"Experiment {i+1}: Tree size: {tree_size}, Accuracy: {accuracy}")

# 2. Experiment with different range of train_test split ratio: Try (30%-70%), (40%-60%), (50%-50%), (60%-40%) and (70%-30%):
# a. Run the experiment with five different random seeds for each of split ratio.
# b. Calculate mean, maximum and minimum accuracy for each split ratio and print them.
# c. Print the mean, max and min tree size for each split ratio.
# d. Draw two plots: 1) shows mean accuracy against training set size and 2) the mean number of nodes in the final tree against training set size.

split_ratios = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]
accuracies = []
tree_sizes = []

for ratio in split_ratios:
    accuracy_per_ratio = []
    tree_size_per_ratio = []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(data.drop('class', axis=1), data['class'], test_size=ratio[1], random_state=i)
        
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        
        accuracy = clf.score(X_test, y_test)
        tree_size = clf.tree_.node_count
        
        accuracy_per_ratio.append(accuracy)
        tree_size_per_ratio.append(tree_size)
    
    # Calculate mean, max, and min accuracy and tree size
    mean_accuracy = np.mean(accuracy_per_ratio)
    max_accuracy = np.max(accuracy_per_ratio)
    min_accuracy = np.min(accuracy_per_ratio)
    
    mean_tree_size = np.mean(tree_size_per_ratio)
    max_tree_size = np.max(tree_size_per_ratio)
    min_tree_size = np.min(tree_size_per_ratio)
    
    accuracies.append((mean_accuracy, max_accuracy, min_accuracy))
    tree_sizes.append((mean_tree_size, max_tree_size, min_tree_size))
    
    print(f"Split Ratio {ratio}:")
    print(f"Mean Accuracy: {mean_accuracy}, Max Accuracy: {max_accuracy}, Min Accuracy: {min_accuracy}")
    print(f"Mean Tree Size: {mean_tree_size}, Max Tree Size: {max_tree_size}, Min Tree Size: {min_tree_size}")
    print()

# Plotting
mean_accuracies = [acc[0] for acc in accuracies]
mean_tree_sizes = [size[0] for size in tree_sizes]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot([ratio[1] * len(data) for ratio in split_ratios], mean_accuracies, marker='o')
plt.title('Mean Accuracy vs Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Accuracy')

plt.subplot(1, 2, 2)
plt.plot([ratio[1] * len(data) for ratio in split_ratios], mean_tree_sizes, marker='o')
plt.title('Mean Tree Size vs Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Tree Size')

plt.tight_layout()
plt.show()



# Define KNN function
from collections import Counter

from collections import Counter

def knn(train_data, test_data, k):
    predictions = []
    for test_instance in test_data:
        distances = []
        for idx, train_instance in enumerate(train_data):
            distance = np.sqrt(np.sum((test_instance[:-1] - train_instance[:-1]) ** 2))
            distances.append((distance, idx, train_instance[-1]))  # Add idx to retain the original indices
        
        # Sort distances and select k nearest neighbors
        sorted_distances = sorted(distances)
        if sorted_distances:  # Check if sorted_distances is not empty
            nearest_neighbors = sorted_distances[:k]
            neighbors = [train_data[neighbor[1]][-1] for neighbor in nearest_neighbors]  # Access class label using original index
        
            if neighbors:  # Check if neighbors list is not empty
                # Count the class votes
                vote = Counter(neighbors).most_common(1)[0][0]
                predictions.append(vote)
            else:
                print("Warning: No neighbors found for the test instance.")
        else:
            print("Warning: sorted_distances is empty.")
    
    return predictions



# Normalize the data
normalized_data = (data - data.mean()) / data.std()

# Split data into train and test sets
train_size = int(0.7 * len(data))
train_data = normalized_data[:train_size].values
test_data = normalized_data[train_size:].values

# Run KNN for different values of k
for k in range(1, 10):
    predictions = knn(train_data, test_data, k)
    correct_predictions = sum(predictions == test_data[:,-1])
    total_instances = len(test_data)
    accuracy = correct_predictions / total_instances
    
    print(f"K = {k}:")
    print(f"Correctly classified instances: {correct_predictions}")
    print(f"Total instances in test set: {total_instances}")
    print(f"Accuracy: {accuracy}")
    print()
