import torch
real_features = torch.load("dataset/extracted_features/fake.pt")
print(real_features.shape)  # Expected: (N, 7, 1024)
