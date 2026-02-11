import numpy as np
import torch

def get_class_rep(post_train_embed, post_train_label):
    """
    Calculate class representatives by averaging embeddings for each class.
    
    Parameters:
    - post_train_embed: list of tensors, embeddings from training data
    - post_train_label: list of labels corresponding to embeddings
    
    Returns:
    - representatives: list of numpy arrays, one representative per class
    """
    representatives = [None] * 5
    for label in range(5):
        indices = np.where(np.atleast_1d(post_train_label) == label)[0]
        class_vectors = [post_train_embed[i] for i in indices]
        class_vectors = [x.cpu().numpy() for x in class_vectors]
        representatives[label] = np.mean(class_vectors, axis=0)
    return representatives


def calculate_normalized_distance(vec1, vec2):
    """
    Calculate normalized Euclidean distance between two vectors.
    
    Parameters:
    - vec1: first vector (tensor or numpy array)
    - vec2: second vector (tensor or numpy array)
    
    Returns:
    - distance: normalized Euclidean distance
    """
    # Ensure vec1 and vec2 are numpy arrays
    if not isinstance(vec1, np.ndarray):
        vec1 = vec1.cpu().detach().numpy()
    if not isinstance(vec2, np.ndarray):
        vec2 = vec2.cpu().detach().numpy()
    
    # Normalize each vector to have unit length
    norm_vec1 = vec1 / np.linalg.norm(vec1)
    norm_vec2 = vec2 / np.linalg.norm(vec2)
    
    # Calculate Euclidean (L2) distance between the normalized vectors
    distance = np.linalg.norm(norm_vec1 - norm_vec2)
    
    return distance


def get_closest_cluster(cluster_representatives, projected_vector, int_to_label):
    """
    Find the closest cluster for a given vector.
    
    Parameters:
    - cluster_representatives: list of representative vectors for each cluster
    - projected_vector: vector to classify
    - int_to_label: dictionary mapping integers to label names
    
    Returns:
    - closest_cluster_label: string, the label of the closest cluster
    """
    distances = [calculate_normalized_distance(rep, projected_vector) for rep in cluster_representatives]
    for i in range(len(distances)):
        distances[i] = np.mean(distances[i])
    closest_cluster_idx = np.argmin(distances)
    return int_to_label[closest_cluster_idx]


def predict(input_vector, siamese_network, embed, labels, int_to_label):
    """
    Predict the class label for an input vector.
    
    Parameters:
    - input_vector: input embedding vector
    - siamese_network: trained Siamese network model
    - embed: list of training embeddings
    - labels: list of training labels
    - int_to_label: dictionary mapping integers to label names
    
    Returns:
    - predicted_label: string, the predicted class label
    """
    modified_vector = siamese_network(input_vector)
    representatives = get_class_rep(embed, labels)
    return get_closest_cluster(representatives, modified_vector, int_to_label)