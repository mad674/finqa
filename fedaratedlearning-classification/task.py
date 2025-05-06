# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms, datasets
# from torch.utils.data import DataLoader, Subset
# import numpy as np
# from torch.optim.lr_scheduler import OneCycleLR

# # Define the Vision Transformer model
# class PatchEmbedding(nn.Module):
#     def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
#         super(PatchEmbedding, self).__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = (224 // patch_size) ** 2
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         x = self.proj(x)  # (n_samples, embed_dim, n_patches**0.5, n_patches**0.5)
#         x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
#         x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)
#         return x

# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.embed_dim = embed_dim
#         self.head_dim = embed_dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
#         self.fc = nn.Linear(embed_dim, embed_dim)

#     def forward(self, x):
#         batch_size, n_patches, embed_dim = x.shape
#         qkv = self.qkv(x).reshape(batch_size, n_patches, 3, self.num_heads, self.head_dim)
#         q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, n_patches, head_dim)
#         attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, n_patches, self.embed_dim)
#         out = self.fc(out)
#         return out

# class TransformerBlock(nn.Module):
#     def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.attn = MultiHeadAttention(embed_dim, num_heads)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, mlp_dim),
#             nn.GELU(),
#             nn.Linear(mlp_dim, embed_dim),
#         )
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = x + self.dropout(self.attn(self.norm1(x)))
#         x = x + self.dropout(self.mlp(self.norm2(x)))
#         return x

# class VisionTransformer(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=4, embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, dropout=0.1):
#         super().__init__()
#         self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
#         num_patches = self.patch_embed.num_patches
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
#         self.dropout = nn.Dropout(dropout)
#         self.blocks = nn.Sequential(
#             *[TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
#         )
#         self.norm = nn.LayerNorm(embed_dim)
#         self.mlp_head = nn.Linear(embed_dim, num_classes)

#     def forward(self, x):
#         B = x.shape[0]
#         x = self.patch_embed(x)
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat([cls_tokens, x], dim=1)
#         x += self.pos_embed
#         x = self.dropout(x)
#         x = self.blocks(x)
#         x = self.norm(x[:, 0])  # CLS token
#         return self.mlp_head(x)

# batch_size = 16
# epochs = 1
# lr = 0.0005
# num_classes = 3
# img_size = 224        

# # Data transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
# ])

# # load data
# def load_data(data_dir):
#     batch_size = 32  # Adjust as needed
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Resize images if needed
#         transforms.ToTensor()
#     ])

#     selected_classes = ['budget', 'form', 'invoice']
#     full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

#     # Filter only selected classes
#     filtered_samples = [
#         (path, selected_classes.index(full_dataset.classes[label_idx]))
#         for path, label_idx in full_dataset.samples if full_dataset.classes[label_idx] in selected_classes
#     ]

#     # Update dataset attributes
#     full_dataset.samples = filtered_samples
#     full_dataset.targets = [s[1] for s in filtered_samples]
#     full_dataset.classes = selected_classes
#     full_dataset.class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}

#     print(f"Total number of training images: {len(full_dataset)}")  # Should now be 1500
#     train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#     return train_loader


# # Define test dataset and DataLoader
# def load_test_data(test_dir):
#     batch_size = 32  # Adjust as needed
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
#     ])
    
#     selected_classes = ['budget', 'form', 'invoice']
#     full_test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

#     # Filter only selected classes
#     filtered_samples = [
#         (path, selected_classes.index(full_test_dataset.classes[label_idx]))
#         for path, label_idx in full_test_dataset.samples if full_test_dataset.classes[label_idx] in selected_classes
#     ]

#     # Update dataset attributes
#     full_test_dataset.samples = filtered_samples
#     full_test_dataset.targets = [s[1] for s in filtered_samples]
#     full_test_dataset.classes = selected_classes
#     full_test_dataset.class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}

#     print(f"Total number of testing images: {len(full_test_dataset)}")
#     test_loader = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
#     return test_loader

# def get_model():
#     return VisionTransformer(num_classes=3)
# torch.cuda.empty_cache()
# # Train and save the model
# def train_and_save_model(train_loader):

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = get_model().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
#     for epoch in range(epochs):
#         model.train()
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}",flush=True)
    
#     torch.save(model.state_dict(), "vit_model.pth")
#     print("Model saved as vit_model.pth")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from torch.optim.lr_scheduler import OneCycleLR

# Define the Vision Transformer model
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (224 // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (n_samples, embed_dim, n_patches*0.5, n_patches*0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, n_patches, embed_dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, n_patches, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, n_patches, head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, n_patches, self.embed_dim)
        out = self.fc(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=4, embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embed
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x[:, 0])  # CLS token
        return self.mlp_head(x)

batch_size = 16
epochs = 1
lr = 0.0005
num_classes = 3
img_size = 224        

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
])

# load data
def load_data(data_dir):
    batch_size = 32  # Adjust as needed
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images if needed
        transforms.ToTensor()
    ])

    selected_classes = ['budget', 'form', 'invoice']
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Filter only selected classes
    filtered_samples = [
        (path, selected_classes.index(full_dataset.classes[label_idx]))
        for path, label_idx in full_dataset.samples if full_dataset.classes[label_idx] in selected_classes
    ]
    filtered_samples = filtered_samples[:10]

    # Update dataset attributes
    full_dataset.samples = filtered_samples
    full_dataset.targets = [s[1] for s in filtered_samples]
    full_dataset.classes = selected_classes
    full_dataset.class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}

    print(f"Total number of training images: {len(full_dataset)}")  # Should now be 1500
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader


# Define test dataset and DataLoader
def load_test_data(test_dir):
    batch_size = 32  # Adjust as needed
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])
    
    selected_classes = ['budget', 'form', 'invoice']
    full_test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # Filter only selected classes
    filtered_samples = [
        (path, selected_classes.index(full_test_dataset.classes[label_idx]))
        for path, label_idx in full_test_dataset.samples if full_test_dataset.classes[label_idx] in selected_classes
    ]
    filtered_samples = random.sample(filtered_samples, 10)
    # Update dataset attributes
    full_test_dataset.samples = filtered_samples
    full_test_dataset.targets = [s[1] for s in filtered_samples]
    full_test_dataset.classes = selected_classes
    full_test_dataset.class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}

    print(f"Total number of testing images: {len(full_test_dataset)}")
    test_loader = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return test_loader

def get_model():
    return VisionTransformer(num_classes=3)
torch.cuda.empty_cache()
# Train and save the model
def train_and_save_model(train_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}",flush=True)
    
    torch.save(model.state_dict(), "vit_model.pth")
    print("Model saved as vit_model.pth")
