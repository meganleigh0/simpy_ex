import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
df = pd.read_csv("your_dataset.csv")  # Replace with actual file path

# Standardize column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Function to clean text
def clean_text(text):
    if pd.isna(text) or text.strip() == "":
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    return text

# Apply cleaning to symptoms and failure modes
df["symptom"] = df["symptom"].apply(clean_text)
df["fmode"] = df["fmode"].apply(clean_text)

# Ensure failure = 0 has no symptom
df.loc[df["failure"] == 0, "symptom"] = ""

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=100)  # Convert symptoms into 100-dimension embeddings
X_symptoms = vectorizer.fit_transform(df["symptom"]).toarray()

# Prepare labels (binary failure classification)
y_failure = df["failure"].values.reshape(-1, 1)

# Split into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X_symptoms, y_failure, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define Simple PyTorch Model
class FailureClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FailureClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 32)  # Second hidden layer
        self.fc3 = nn.Linear(32, 1)  # Output layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize model, loss function, optimizer
model = FailureClassifier(input_dim=100)
criterion = nn.BCELoss()  # Binary cross-entropy loss for failure classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}")

# Evaluate Model on Test Data
with torch.no_grad():
    test_outputs = model(X_test_tensor).squeeze()
    test_predictions = (test_outputs > 0.5).float()  # Convert to 0/1 labels
    accuracy = (test_predictions == y_test_tensor.squeeze()).sum().item() / len(y_test_tensor)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save cleaned and classified dataset
df.to_csv("processed_failures.csv", index=False)
print("Processed dataset saved as 'processed_failures.csv'")