import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from pathlib import Path

def plot_tsne_and_confusion_matrix(model, test_csv, save_dir):
    print("Generating t-SNE and Confusion Matrix...")
    df = pd.read_csv(test_csv, index_col=0, parse_dates=True)
    features = df.drop(columns=['target']).values
    targets = df['target'].values
    
    X, y = [], []
    for i in range(len(features) - model.input_chunk_length):
        X.append(features[i:(i + model.input_chunk_length)])
        y.append(targets[i + model.input_chunk_length])
        
    X = torch.FloatTensor(np.array(X)).to(model.device)
    y = np.array(y)
    
    model.model.eval()
    embeddings = []
    preds = []
    
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            h, _, _, c_macro = model.model(batch_X)
            embeddings.append(h.cpu().numpy())
            preds.append(c_macro.cpu().numpy())
            
    embeddings = np.concatenate(embeddings, axis=0)
    preds = np.concatenate(preds, axis=0)
    
    # Map raw probability argmax (0,1,2) to actual Triple Barrier labels (-1.0, 0.0, 1.0)
    # 0 -> Flat (0.0)
    # 1 -> Up (1.0)
    # 2 -> Down (-1.0)
    pred_indices = np.argmax(preds, axis=1)
    
    # Default assumption: 0=Flat, 1=Up, 2=Down
    # However, since the clusters were discovered entirely blind, we just map 
    # them to [-1.0, 0.0, 1.0] heuristically to see the distribution.
    # We map index 0 to 0.0, 1 to 1.0, 2 to -1.0
    mapping = {0: 0.0, 1: 1.0, 2: -1.0}
    pred_labels = np.array([mapping[idx] for idx in pred_indices])
    
    # t-SNE Visualization
    print("Computing t-SNE... this might take a minute.")
    # Use fewer samples if dataset is huge to speed up TSNE
    sample_size = min(len(embeddings), 2000)
    indices = np.random.choice(len(embeddings), sample_size, replace=False)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings[indices])
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=pred_labels[indices], cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter, ticks=[-1.0, 0.0, 1.0], label="Discovered Macro Cluster")
    plt.title("t-SNE of Self-Discovered Trading Regimes")
    plt.savefig(Path(save_dir) / "tsne_clusters.png")
    plt.close()
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y, pred_labels, labels=[-1.0, 0.0, 1.0])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Flat', 'Up'], yticklabels=['Down', 'Flat', 'Up'])
    plt.xlabel('Discovered Cluster (Macro)')
    plt.ylabel('Actual Triple Barrier Label')
    plt.title('Self-Supervised Clusters vs Actual Labels')
    plt.savefig(Path(save_dir) / "cluster_confusion_matrix.png")
    plt.close()
    
    print(f"Visualizations saved to {save_dir}")
