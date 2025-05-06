# import flwr as fl
# import torch
# import sys
# from task import get_model, load_data

# # Get client ID from command-line argument
# client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # Default: client 0

# # Initialize device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Client {client_id} using device: {device}")

# # Load dataset
# print("Loading training data...")
# train_loader = load_data("C:/Users/Dell/Documents/Downloads/Finance/Finance_Data")
# print("Loading test data...")
# test_loader = load_data("C:/Users/Dell/Documents/Downloads/Finance/Finance_Data_Test")

# # Validate data loading
# if train_loader is None or test_loader is None:
#     raise ValueError("Failed to load training or test data. Check file paths and format.")
# print("Data loaded successfully.")

# # Load model
# model = get_model().to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
# criterion = torch.nn.CrossEntropyLoss()

# # Define Flower Client
# class FlowerClient(fl.client.NumPyClient):
#     def get_parameters(self, config):
#         print(f"Client {client_id}: Sending parameters to server...")
#         params = [val.cpu().numpy() for val in model.state_dict().values()]
#         for i, param in enumerate(params[:2]):  # Print first 2 parameters for visibility
#             print(f"Client {client_id} - Param {i}: {param.flatten()[:5]} ...")  
#         return params

#     def set_parameters(self, parameters):
#         print(f"Client {client_id}: Received updated parameters from server...")
#         params_dict = zip(model.state_dict().keys(), parameters)
#         for i, (name, param) in enumerate(params_dict):
#             param_tensor = torch.tensor(param, dtype=model.state_dict()[name].dtype, device=device)
#             model.state_dict()[name].copy_(param_tensor.reshape_as(model.state_dict()[name]))
#             if i < 2:  # Print first 2 parameters
#                 print(f"Client {client_id} - Updated Param {i}: {param_tensor.flatten()[:5]} ...")

#     def fit(self, parameters, config):
#         print("\n Client: Received parameters from server...")
#         self.set_parameters(parameters)
#         print(" Client: Training started...", flush=True)

#         model.train()
#         running_loss, correct, total = 0.0, 0, 0
#         num_batches = len(train_loader)

#         for batch_idx, (images, labels) in enumerate(train_loader):
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#             # 🔹 Print progress every 10 batches
#             if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
#                 print(f" Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f}")

#         accuracy = 100. * correct / total if total > 0 else 0
#         print(f"\n Training complete. Final Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%", flush=True)

#         updated_params = self.get_parameters({})
#         print("\n📤 Client: Sending updated parameters to server...")
#         return updated_params, len(train_loader.dataset), {"loss": running_loss, "accuracy": accuracy}

#     def evaluate(self, parameters, config):
#         print("\nClient: Running evaluation...", flush=True)
#         self.set_parameters(parameters)
#         model.eval()

#         correct, total, total_loss = 0, 0, 0.0

#         with torch.no_grad():
#             for images, labels in test_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 total_loss += loss.item()

#                 _, predicted = outputs.max(1)
#                 correct += predicted.eq(labels).sum().item()
#                 total += labels.size(0)

#         accuracy = 100.0 * correct / total if total > 0 else 0
#         avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0

#         print(f"Test Accuracy: {accuracy:.2f}% | Avg Loss: {avg_loss:.4f}", flush=True)
#         return avg_loss, len(test_loader.dataset), {"accuracy": accuracy}

# # Start client
# if __name__ == "__main__":
#     fl.client.start_client(server_address="192.168.31.102:8080", client=FlowerClient().to_client())
import flwr as fl
import torch
import sys
from task import get_model, load_data

# Get client ID from command-line argument
client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # Default: client 0

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Client {client_id} using device: {device}")

# Load dataset
print("Loading training data...")
train_loader = load_data("Finance_Data")
print("Loading test data...")
test_loader = load_data("Finance_Data_Test")

# Validate data loading
if train_loader is None or test_loader is None:
    raise ValueError("Failed to load training or test data. Check file paths and format.")
print("Data loaded successfully.")

# Load model
model = get_model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
criterion = torch.nn.CrossEntropyLoss()

# Define Flower Client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        print(f"Client {client_id}: Sending parameters to server...")
        params = [val.cpu().numpy() for val in model.state_dict().values()]
        return params
    def set_parameters(self, parameters):
        print(f"Client {client_id}: Received updated parameters from server...")
        params_dict = zip(model.state_dict().keys(), parameters)
        for i, (name, param) in enumerate(params_dict):
            param_tensor = torch.tensor(param, dtype=model.state_dict()[name].dtype, device=device)
            model.state_dict()[name].copy_(param_tensor.reshape_as(model.state_dict()[name]))

    def fit(self, parameters, config):
        print("\n📩 Client: Received parameters from server...")
        self.set_parameters(parameters)
        print("🚀 Client: Training started...", flush=True)

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        num_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 🔹 Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"🟢 Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f}")

        accuracy = 100. * correct / total if total > 0 else 0
        print(f"\n✅ Training complete. Final Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%", flush=True)

        updated_params = self.get_parameters({})
        print("\n📤 Client: Sending updated parameters to server...")
        for i, (name, param) in enumerate(model.named_parameters()):
            param_data = param.data.cpu().numpy()
            print(f"\nClient {client_id} - Layer {i} - Param: {name}")

            # Handle different dimensions
            if param_data.ndim == 1:
                # For 1D params (e.g., biases, layer norm weights)
                print("Values:", param_data[:10], "...\n" if len(param_data) > 20 else " ")

            elif param_data.ndim >= 2:
                # For 2D or more (e.g., weights)
                rows = min(2, param_data.shape[0])
                cols = min(5, param_data.shape[1]) if param_data.ndim > 1 else 20
                print("Values (first 2 rows × 5 cols):")
                for row in range(rows):
                    print(param_data[row][:cols], "...")
                print()

            else:
                # Unusual case
                print("Values:", param_data[:10], "\n")
        return updated_params, len(train_loader.dataset), {"loss": running_loss, "accuracy": accuracy}

    def evaluate(self, parameters, config):
        print("\nClient: Running evaluation...", flush=True)
        self.set_parameters(parameters)
        model.eval()

        correct, total, total_loss = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        accuracy = 100.0 * correct / total if total > 0 else 0
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0

        print(f"Test Accuracy: {accuracy:.2f}% | Avg Loss: {avg_loss:.4f}", flush=True)
        return avg_loss, len(test_loader.dataset), {"accuracy": accuracy}

# Start client
if __name__ == "__main__":
    fl.client.start_client(server_address="172.16.20.42:8080", client=FlowerClient().to_client())

