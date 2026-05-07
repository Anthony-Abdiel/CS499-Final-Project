import torch

from src_counting.models.lightweight_counting_cnn import LightweightCountingCNN


model = LightweightCountingCNN()

dummy_images = torch.randn(8, 1, 128, 128)

outputs = model(dummy_images)

print("Output shape:", outputs.shape)
print("Outputs:", outputs.squeeze().tolist())