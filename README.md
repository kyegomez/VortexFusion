[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Vortex Fusion
This is the first ever implementation of a joint Transformer + Mamba + LSTM architecture. The flow is the following: `mamba -> transformer -> lstm` in a loop. Perhaps with more iteration on model design, we can find a better architecture but this architecture is the future.


## install

```bash
$ pip3 install -U vortex-fusion

```

## Usage
```python
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
```

# License
MIT

# Todo
- [ ] Train this on the same dataset as Llama. Create one script that trains this model on a massive dataset to experiment with performance metrics. 
