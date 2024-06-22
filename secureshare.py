import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds

# Load IMDB dataset
def load_imdb_dataset(max_samples=25000):
    (ds_train, ds_test), ds_info = tfds.load(
        'imdb_reviews',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def preprocess_text(text, label):
        return text.decode('utf-8'), label

    train_reviews = []
    train_labels = []
    for text, label in tfds.as_numpy(ds_train.take(max_samples)):
        review, sentiment = preprocess_text(text, label)
        train_reviews.append(review)
        train_labels.append(sentiment)

    test_reviews = []
    test_labels = []
    for text, label in tfds.as_numpy(ds_test.take(max_samples // 5)):
        review, sentiment = preprocess_text(text, label)
        test_reviews.append(review)
        test_labels.append(sentiment)

    return train_reviews, train_labels, test_reviews, test_labels

# Federated Learning Server
class FederatedServer:
    def __init__(self, num_clients, initial_lr=0.1):
        self.num_clients = num_clients
        self.global_model = None
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.clients = []
        self.learning_rate = initial_lr

    def initialize_model(self, X):
        self.vectorizer.fit(X)
        vocab_size = len(self.vectorizer.get_feature_names_out())
        self.global_model = LogisticRegression(warm_start=True, max_iter=1000, C=1.0, solver='lbfgs')
        self.global_model.classes_ = np.array([0, 1])
        self.global_model.coef_ = np.random.randn(1, vocab_size)
        self.global_model.intercept_ = np.random.randn(1)

    def aggregate_models(self):
        client_coefs = []
        client_intercepts = []
        for client in self.clients:
            client_coefs.append(client.model.coef_)
            client_intercepts.append(client.model.intercept_)

        new_coef = np.mean(client_coefs, axis=0)
        new_intercept = np.mean(client_intercepts, axis=0)

        self.global_model.coef_ = self.global_model.coef_ + self.learning_rate * (new_coef - self.global_model.coef_)
        self.global_model.intercept_ = self.global_model.intercept_ + self.learning_rate * (new_intercept - self.global_model.intercept_)

    def distribute_model(self):
        for client in self.clients:
            client.update_model(self.global_model)

    def decay_learning_rate(self, decay_factor=0.95):
        self.learning_rate *= decay_factor

# Federated Learning Client
class FederatedClient:
    def __init__(self, client_id, X, y, server):
        self.client_id = client_id
        self.X = X
        self.y = y
        self.server = server
        self.model = None

    def preprocess_data(self):
        self.X_vec = self.server.vectorizer.transform(self.X)
        self.X_vec = normalize(self.X_vec, norm='l2', axis=1)

    def train(self, epochs=5):
        if self.model is None:
            self.model = LogisticRegression(warm_start=True, max_iter=1000, C=1.0, solver='lbfgs')
            self.model.classes_ = np.array([0, 1])
            self.model.coef_ = self.server.global_model.coef_.copy()
            self.model.intercept_ = self.server.global_model.intercept_.copy()

        for _ in range(epochs):
            self.model.fit(self.X_vec, self.y)

    def update_model(self, global_model):
        self.model.coef_ = global_model.coef_.copy()
        self.model.intercept_ = global_model.intercept_.copy()

# Non-uniform data distribution
def distribute_data_non_uniform(X, y, num_clients):
    client_data = []
    class_indices = [np.where(y == c)[0] for c in np.unique(y)]

    for i in range(num_clients):
        client_indices = []
        for class_idx in class_indices:
            proportion = np.random.beta(2, 5)
            num_samples = int(proportion * len(class_idx))
            client_indices.extend(np.random.choice(class_idx, num_samples, replace=False))

        client_X = [X[j] for j in client_indices]
        client_y = [y[j] for j in client_indices]
        client_data.append((client_X, client_y))

    return client_data

# Main federated learning process
def run_federated_learning(num_clients, num_rounds, train_reviews, train_labels, test_reviews, test_labels):
    server = FederatedServer(num_clients)

    client_data = distribute_data_non_uniform(train_reviews, train_labels, num_clients)

    for i, (X, y) in enumerate(client_data):
        client = FederatedClient(i, X, y, server)
        server.clients.append(client)

    server.initialize_model(train_reviews)

    for client in server.clients:
        client.preprocess_data()

    global_metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
    client_accuracies = [[] for _ in range(num_clients)]

    for round in range(num_rounds):
        print(f"Round {round + 1}/{num_rounds}")

        for client in server.clients:
            client.train()

        server.aggregate_models()
        server.distribute_model()
        server.decay_learning_rate()

        X_test_vec = server.vectorizer.transform(test_reviews)
        X_test_vec = normalize(X_test_vec, norm='l2', axis=1)
        global_preds = server.global_model.predict(X_test_vec)

        global_metrics['accuracy'].append(accuracy_score(test_labels, global_preds))
        global_metrics['f1'].append(f1_score(test_labels, global_preds, average='weighted'))
        global_metrics['precision'].append(precision_score(test_labels, global_preds, average='weighted'))
        global_metrics['recall'].append(recall_score(test_labels, global_preds, average='weighted'))

        for i, client in enumerate(server.clients):
            client_accuracy = accuracy_score(client.y, client.model.predict(client.X_vec))
            client_accuracies[i].append(client_accuracy)

        print(f"Global model accuracy: {global_metrics['accuracy'][-1]:.4f}, F1 score: {global_metrics['f1'][-1]:.4f}")

    return server, global_metrics, client_accuracies

# Run federated learning
num_rounds = 50
client_numbers = [3, 5, 10]
all_global_metrics = []
all_client_accuracies = []

train_reviews, train_labels, test_reviews, test_labels = load_imdb_dataset()

for num_clients in client_numbers:
    print(f"\nRunning federated learning with {num_clients} clients")
    server, global_metrics, client_accuracies = run_federated_learning(num_clients, num_rounds, train_reviews, train_labels, test_reviews, test_labels)
    all_global_metrics.append(global_metrics)
    all_client_accuracies.append(client_accuracies)

# Centralized model for comparison
centralized_model = LogisticRegression(max_iter=1000)
X_train_vec = server.vectorizer.transform(train_reviews)
X_train_vec = normalize(X_train_vec, norm='l2', axis=1)
centralized_model.fit(X_train_vec, train_labels)

X_test_vec = server.vectorizer.transform(test_reviews)
X_test_vec = normalize(X_test_vec, norm='l2', axis=1)
centralized_accuracy = accuracy_score(test_labels, centralized_model.predict(X_test_vec))

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
colors = sns.color_palette("deep", n_colors=max(client_numbers))

# 1. Privacy preservation visualization
fig, ax = plt.subplots(figsize=(12, 8))
client_data_sizes = [len(client.X) for client in server.clients]
total_data = sum(client_data_sizes)

ax.bar(range(1, len(client_data_sizes) + 1), client_data_sizes, color=colors)
ax.set_xlabel('Client', fontsize=14)
ax.set_ylabel('Number of Reviews', fontsize=14)
ax.set_title('Data Distribution Among Clients\nShowcasing Privacy Preservation', fontsize=18)
ax.text(0.5, -0.15, f'Each client keeps their data locally, processing only {client_data_sizes[0]/total_data:.1%} of total data on average',
        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)

for i, v in enumerate(client_data_sizes):
    ax.text(i+1, v, str(v), ha='center', va='bottom')

ax.grid(False)
plt.tight_layout()
plt.savefig('privacy_preservation.png', dpi=300, bbox_inches='tight')
#plt.close()

# 2. Model performance comparison
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(client_numbers))
width = 0.35

centralized_accuracies = [centralized_accuracy for _ in client_numbers]
federated_accuracies = [global_metrics['accuracy'][-1] for global_metrics in all_global_metrics]

rects1 = ax.bar(x - width/2, centralized_accuracies, width, label='Centralized', color=colors[0])
rects2 = ax.bar(x + width/2, federated_accuracies, width, label='Federated', color=colors[1])

ax.set_xlabel('Number of Clients', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('Centralized vs Federated Learning Performance', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(client_numbers)
ax.legend(fontsize=12)

ax.bar_label(rects1, padding=3, fmt='%.3f')
ax.bar_label(rects2, padding=3, fmt='%.3f')

ax.grid(False)
plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
#plt.close()

# 3. Communication efficiency
fig, ax = plt.subplots(figsize=(12, 8))

for i, client_num in enumerate(client_numbers):
    ax.plot(range(1, num_rounds + 1), all_global_metrics[i]['accuracy'],
            label=f'{client_num} clients ({sum(len(client.X) for client in server.clients)} reviews)',
            color=colors[i], linewidth=2)

ax.set_xlabel('Communication Rounds', fontsize=14)
ax.set_ylabel('Global Model Accuracy', fontsize=14)
ax.set_title('Communication Efficiency in Federated Learning', fontsize=18)
ax.legend(fontsize=12)

ax.grid(False)
plt.tight_layout()
plt.savefig('communication_efficiency.png', dpi=300, bbox_inches='tight')
#plt.close()

# 4. Model convergence across clients
fig, ax = plt.subplots(figsize=(12, 8))

for i, accuracies in enumerate(all_client_accuracies[-1]):
    ax.plot(range(1, num_rounds + 1), accuracies,
            label=f'Client {i+1} ({len(server.clients[i].X)} reviews)',
            color=colors[i], alpha=0.7, linewidth=2)

ax.plot(range(1, num_rounds + 1), all_global_metrics[-1]['accuracy'],
        label=f'Global Model ({sum(len(client.X) for client in server.clients)} total reviews)',
        color='black', linewidth=3)

ax.set_xlabel('Rounds', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('Model Convergence Across Clients', fontsize=18)
ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

ax.grid(False)
plt.tight_layout()
plt.savefig('model_convergence.png', dpi=300, bbox_inches='tight')
#plt.close()

# 5. Data heterogeneity handling
fig, ax = plt.subplots(figsize=(12, 8))

client_class_distributions = []
for client in server.clients:
    class_dist = np.bincount(client.y) / len(client.y)
    client_class_distributions.append(class_dist)

client_class_distributions = np.array(client_class_distributions)
sns.heatmap(client_class_distributions, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax)

ax.set_xlabel('Class', fontsize=14)
ax.set_ylabel('Client', fontsize=14)
ax.set_title('Data Heterogeneity Across Clients', fontsize=18)

plt.tight_layout()
plt.savefig('data_heterogeneity.png', dpi=300, bbox_inches='tight')
#plt.close()

print("Visualizations have been saved as PNG files.")
