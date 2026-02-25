import numpy as np
import torch
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score

def multiclass_roc_auc_score(truth, pred, average="weighted"):
    """Calculate multiclass ROC AUC score"""
    lb = LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)
    return roc_auc_score(truth, pred, average=average)

def get_class_rep(post_train_embed, post_train_label):
    """Get class representatives by averaging embeddings"""
    representatives = [None] * 5
    for label in range(5):
        indices = np.where(np.atleast_1d(post_train_label) == label)[0]
        class_vectors = [post_train_embed[i] for i in indices]
        class_vectors = [x.cpu().numpy() for x in class_vectors]
        representatives[label] = np.mean(class_vectors, axis=0)
    return representatives

def calculate_normalized_distance(vec1, vec2):
    """Calculate normalized Euclidean distance between two vectors"""
    if not isinstance(vec1, np.ndarray):
        vec1 = vec1.cpu().detach().numpy()
    if not isinstance(vec2, np.ndarray):
        vec2 = vec2.cpu().detach().numpy()
    
    norm_vec1 = vec1 / np.linalg.norm(vec1)
    norm_vec2 = vec2 / np.linalg.norm(vec2)
    
    distance = np.linalg.norm(norm_vec1 - norm_vec2)
    
    return distance

def get_closest_cluster(cluster_representatives, projected_vector, int_to_label):
    """Find the closest cluster for a given vector"""
    distances = [calculate_normalized_distance(rep, projected_vector) 
                for rep in cluster_representatives]
    for i in range(len(distances)):
        distances[i] = np.mean(distances[i])
    closest_cluster_idx = np.argmin(distances)
    return int_to_label[closest_cluster_idx]

def extract_projections(siamese_network, dataloader, device):
    """Extract projections from a trained Siamese network"""
    projections = []
    labels = []
    for batch in dataloader:
        label = batch["label"]
        anchor = batch["anchor"].to(device)
        projection = siamese_network(anchor)
        
        projections.append(projection.cpu().detach().numpy())
        labels.append(label.numpy())
    projections = np.vstack(projections)
    labels = np.hstack(labels)
    return projections, labels