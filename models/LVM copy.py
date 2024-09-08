import torch
import torch.nn as nn

class Config:
    ''' Configurations for LVM '''
    name = 'LVM'
    slice_size = (128, 128, 21)

    batch_size = 12
    loss_type = 'diceCE'
    optimizer = 'adam'


class LVM(nn.Module):
    ''' Architecture for LVM '''

    def __init__(self, input_dim, hidden_dim=640, n_heads=10, n_layers=22):
        super(LVM, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Positional Encoding to encode slice position along the z-axis
        self.pos_embedding = nn.Parameter(torch.zeros(1, hidden_dim))
        
        # Linear layers to project 2D slice and hidden state to the same dimension
        self.slice_projection = nn.Linear(input_dim, hidden_dim)
        self.hidden_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layer to produce the segmentation map
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, slices):
        batch_size, num_slices, _, _ = slices.shape
        
        # Initial hidden state (e.g., zeros or learned initial state)
        hidden_state = torch.zeros(batch_size, self.hidden_dim, device=slices.device)
        
        # Store the predicted segmentation maps
        segmentation_maps = []
        
        for i in range(num_slices):
            current_slice = slices[:, i, :, :]  # Get the i-th slice
            current_slice = current_slice.flatten(start_dim=1)  # Flatten to 2D vector

            # Project current slice and hidden state to the same dimension
            slice_emb = self.slice_projection(current_slice)
            hidden_emb = self.hidden_projection(hidden_state)
            
            # Add positional encoding
            hidden_emb += self.pos_embedding
            
            # Combine and prepare input to transformer (batch_size x seq_len=2 x hidden_dim)
            transformer_input = torch.stack([slice_emb, hidden_emb], dim=1)
            
            # Pass through transformer
            transformer_output = self.transformer(transformer_input)
            
            # Use the output corresponding to the slice (index 0) for prediction
            predicted_segmentation = self.output_layer(transformer_output[:, 0, :])
            segmentation_maps.append(predicted_segmentation.view_as(current_slice))
            
            # Update hidden state with the new prediction
            hidden_state = transformer_output[:, 1, :]
        
        # Stack the segmentation maps to return them as output
        segmentation_maps = torch.stack(segmentation_maps, dim=1).view(batch_size, num_slices, -1, -1)
        return segmentation_maps

# Example instantiation:
# input_dim: Flattened size of each 2D slice (e.g., 256*256 if slices are 256x256)
# hidden_dim: Size of the hidden state
# n_heads: Number of attention heads
# n_layers: Number of Transformer layers

# model = LVM(input_dim=256*256, hidden_dim=512, n_heads=8, n_layers=6)

'''
for hidden state transformer (wrapped inside a RNN)

model = MyModel()
slices # batch size, num slices, sequence length (which are indices for codebook)

def forward(slices):
    output = torch.zeros(batch_size, num_slices, sequence_length) # output and hidden state are the same
    hidden = torch.zeros(batch_size, num_slices, sequence_length)
    for i in range(slices.size[1]):
        current_slice = slices[:, i, :]
        inputs = torch.cat(current_slice, hidden[:, i, :])
        output[:, i, :], hidden[:, i+1, :] = self.transformer(inputs)
    
    return output, hidden
    
    
loss = criterion(output, target)

'''