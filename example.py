import torch
from vortex_fusion import VortexFusion

# Generate random input tensor
x = torch.randint(0, 10000, (1, 10))

# Create an instance of the VortexFusion model with dimension 512
model = VortexFusion(dim=512)

# Pass the input tensor through the model to get the output
output = model(x)

# Print the shape of the output tensor
print(output.shape)
