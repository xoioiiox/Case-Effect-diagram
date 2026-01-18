import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification


# Define Notears MLP model
class NotearsMLP(nn.Module):
    def __init__(self, n_features, hidden_dim=64):
        super(NotearsMLP, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.W = nn.Parameter(torch.randn(n_features, n_features))  # Initialize W as a learnable parameter

    def forward(self, X):
        x = torch.relu(self.fc1(X))
        x = torch.relu(self.fc2(x))
        return self.W  # Return the learned weight matrix W


# Define the Notears-MLP loss function
def notears_loss(W, X, reg_lambda=0.1):
    """
    Loss function for Notears. Includes a reconstruction loss and a sparsity penalty
    to enforce a directed acyclic graph (DAG) structure.
    """
    # Reconstruction loss (L2 norm of the error)
    recon_loss = torch.norm(X - torch.matmul(X, W), p='fro') ** 2

    # Sparsity penalty (encouraging sparse connections)
    sparsity_penalty = reg_lambda * torch.norm(W, p='fro') ** 2

    # The total loss
    return recon_loss + sparsity_penalty


# Simulate some data
def generate_data(n_samples=100, n_features=5):
    """Generate synthetic data for testing"""
    X, _ = make_multilabel_classification(n_samples=n_samples, n_features=n_features, random_state=42)
    return torch.tensor(X, dtype=torch.float32)


# Training the model
def train_notears(X, hidden_dim=64, reg_lambda=0.1, epochs=1000, lr=0.001):
    n_samples, n_features = X.shape
    model = NotearsMLP(n_features, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        # Forward pass
        W = model(X)

        # Compute loss
        loss = notears_loss(W, X, reg_lambda)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    return model


# Visualize the learned weight matrix (causal graph)
def plot_causal_graph(model, X):
    W = model(X).detach().numpy()
    plt.imshow(W, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Learned Causal Graph (Weight Matrix)")
    plt.show()


# Main function to run the demo
def main():
    n_samples = 100
    n_features = 5
    hidden_dim = 64

    # Step 1: Generate synthetic data
    X = generate_data(n_samples, n_features)

    # Step 2: Train the Notears-MLP model
    model = train_notears(X, hidden_dim=hidden_dim)

    # Step 3: Visualize the learned weight matrix (causal graph)
    plot_causal_graph(model, X)


if __name__ == "__main__":
    main()
