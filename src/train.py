import torch
from model import CatClassifier
from utils import load_data
import mlflow
import mlflow.pytorch

def train():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("cat-classifier")
    with mlflow.start_run():
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("epochs", 5)

        train_loader, val_loader = load_data()
        model = CatClassifier()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(5):
            model.train()
            running_loss = 0
            
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            # Validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_acc = correct / total
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        torch.save(model.state_dict(), "model.pth")
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    train()
