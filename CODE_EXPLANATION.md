# Complete Code Explanation for VideoMAE Training Codebase

This document provides a comprehensive explanation of all components in the VideoMAE training codebase, organized by module.

## Table of Contents

1. [Overview](#overview)
2. [Data Module (`data/video_dataset.py`)](#data-module)
3. [Model Module (`models/videomae.py`)](#model-module)
4. [Training Module (`training/lightning_module.py`)](#training-module)
5. [Training Script (`training/train.py`)](#training-script)
6. [Data Flow and Training Process](#data-flow-and-training-process)

---

## Overview

This codebase implements a **Video Masked Autoencoder (VideoMAE)** using the **EVEREST** training method. The model learns video representations by:

1. **Masking**: Randomly masking 75% of video patches
2. **Encoding**: Processing only visible (unmasked) patches through a Vision Transformer encoder
3. **Decoding**: Reconstructing all patches (visible + masked) using a lightweight decoder
4. **Learning**: Computing reconstruction loss only on masked patches

The architecture supports three Vision Transformer backbones:
- **ViT-S** (Small): 384 embedding dim, 12 layers, 6 heads
- **ViT-B** (Base): 768 embedding dim, 12 layers, 12 heads  
- **ViT-L** (Large): 1024 embedding dim, 24 layers, 16 heads

---

## Data Module

### File: `data/video_dataset.py`

#### Class: `VideoDataset`

**Purpose**: PyTorch Dataset class for loading and processing video files from CSV file paths using torchvision.

**Implementation Note**: This module uses `torchvision.io.read_video()` instead of OpenCV for video loading. Benefits include:
- **Native PyTorch Integration**: All operations use PyTorch tensors (no numpy/OpenCV conversions)
- **RGB by Default**: torchvision returns videos in RGB format (no BGR→RGB conversion needed)
- **Better Performance**: Direct tensor operations without intermediate conversions
- **Simpler Code**: Fewer dependencies and cleaner implementation

#### `__init__` Method

```python
def __init__(
    self,
    csv_file: str,
    transform: Optional[Callable] = None,
    num_frames_to_sample: int = 32,
    temporal_stride: int = 2,
    frame_size: tuple = (224, 224)
)
```

**What it does**:
1. **Reads CSV file**: Loads video file paths from CSV
   - Handles both CSV files with and without headers
   - If no header, assumes first column contains paths
   - If header exists, looks for 'path' column or uses first column
   
2. **Stores configuration**:
   - `num_frames_to_sample`: Number of consecutive frames to sample (default: 32)
   - `temporal_stride`: Stride for temporal downsampling (default: 2)
   - `frame_size`: Expected frame dimensions (default: 224x224)
   - Calculates final number of frames: `num_frames = 32 // 2 = 16`

**Why**: Provides flexible CSV reading and stores processing parameters for video loading.

#### `__len__` Method

```python
def __len__(self) -> int:
    return len(self.video_paths)
```

**What it does**: Returns the total number of videos in the dataset.

**Why**: Required by PyTorch Dataset protocol to enable iteration and batching.

#### `__getitem__` Method

```python
def __getitem__(self, idx: int) -> torch.Tensor:
```

**What it does** (step by step):

1. **Load Video**:
   ```python
   video, audio, info = torchvision.io.read_video(video_path)
   ```
   - Loads video file using torchvision's `read_video()` function
   - Returns video tensor in `(T, H, W, C)` format, already in RGB color space
   - Video is returned as uint8 tensor (values 0-255)
   - **Why torchvision?**: Native PyTorch integration, no color space conversion needed
   - **Benefits**: All operations stay in PyTorch tensors, better performance

2. **Normalize and Rearrange**:
   ```python
   video = video.float() / 255.0  # Normalize to [0, 1]
   video = video.permute(0, 3, 1, 2)  # (T, C, H, W)
   ```
   - Converts from uint8 to float32 and normalizes to [0, 1]
   - Rearranges from `(T, H, W, C)` to `(T, C, H, W)` for easier manipulation
   - All operations use native PyTorch tensors (no numpy conversion)

3. **Handle Insufficient Frames**:
   ```python
   if num_frames < self.num_frames_to_sample:
       num_padding = self.num_frames_to_sample - num_frames
       last_frame = video[-1:].expand(num_padding, -1, -1, -1)
       video = torch.cat([video, last_frame], dim=0)
   ```
   - If video has fewer than 32 frames, pads by repeating the last frame
   - Uses PyTorch's `expand()` and `cat()` for efficient tensor operations
   - Ensures we always have enough frames for sampling

4. **Random Sampling**:
   ```python
   start_idx = random.randint(0, max_start_idx)
   sampled_frames = video[start_idx:start_idx + self.num_frames_to_sample]
   ```
   - Randomly selects a starting index
   - Samples 32 consecutive frames using tensor slicing
   - **Why random?**: Provides data augmentation - different clips from same video each epoch
   - Result: `(32, C, H, W)` tensor

5. **Temporal Downsampling**:
   ```python
   downsampled_frames = sampled_frames[::self.temporal_stride]  # (16, C, H, W)
   ```
   - Applies stride of 2: takes every 2nd frame using tensor slicing
   - Reduces 32 frames to 16 frames
   - **Why?**: Reduces computational cost while maintaining temporal information

6. **Final Rearrangement**:
   ```python
   video_tensor = downsampled_frames.permute(1, 0, 2, 3)  # (C, T, H, W)
   ```
   - Rearranges from `(T, C, H, W)` to `(C, T, H, W)` format
   - Final output: `(3, 16, 224, 224)` = `(Channels, Time, Height, Width)`

**Returns**: Tensor of shape `(3, 16, 224, 224)` ready for model input.

**Returns**: Tensor of shape `(3, 16, 224, 224)` ready for model input.

**Why this format**: 
- Channels first (C, T, H, W) is standard for PyTorch
- Normalized values help with training stability
- 16 frames is a good balance between temporal information and computational cost
- All operations use native PyTorch tensors (no numpy/OpenCV conversions)
- torchvision provides better integration with PyTorch ecosystem

---

## Model Module

### File: `models/videomae.py`

This module contains the core VideoMAE architecture with several components.

### Class: `PatchEmbed`

**Purpose**: Converts video into patch embeddings using 3D convolution.

#### `__init__` Method

```python
def __init__(
    self,
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    embed_dim: int = 768,
    t_patch_size: int = 2
)
```

**What it does**:
1. **Calculates patch dimensions**:
   - Spatial patches per frame: `(224 // 16)² = 14² = 196` patches
   - Temporal patches: `16 // 2 = 8` temporal segments
   - Total patches: `8 × 196 = 1,568` patches

2. **Creates 3D Convolution**:
   ```python
   self.proj = nn.Conv3d(
       in_chans, embed_dim,
       kernel_size=(t_patch_size, patch_size, patch_size),  # (2, 16, 16)
       stride=(t_patch_size, patch_size, patch_size)        # (2, 16, 16)
   )
   ```
   - **Kernel size**: `(2, 16, 16)` extracts 2×16×16 spatio-temporal patches
   - **Stride**: Same as kernel size, so no overlap
   - **Why 3D?**: Captures both spatial (H×W) and temporal (T) information together

#### `forward` Method

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Input: (B, C, T, H, W) = (B, 3, 16, 224, 224)
    x = self.proj(x)  # (B, embed_dim, T', H', W') = (B, 768, 8, 14, 14)
    x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim) = (B, 1568, 768)
    return x
```

**What it does**:
1. Applies 3D convolution to extract patches
2. Flattens spatial and temporal dimensions
3. Rearranges to sequence format: `(batch, num_patches, embed_dim)`

**Why**: Transforms video into a sequence of patch embeddings that transformers can process.

---

### Class: `PositionalEncoding`

**Purpose**: Adds learnable positional embeddings to patches.

#### `__init__` Method

```python
def __init__(self, num_patches: int, embed_dim: int):
    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
    nn.init.trunc_normal_(self.pos_embed, std=0.02)
```

**What it does**:
- Creates learnable parameter tensor: `(1, num_patches, embed_dim)`
- Initializes with truncated normal distribution (std=0.02)
- **Why learnable?**: Model learns optimal positional encodings during training

#### `forward` Method

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.pos_embed
```

**What it does**: Element-wise addition of positional embeddings to input.

**Why**: Transformers need positional information since they process sequences without inherent order.

---

### Class: `TransformerBlock`

**Purpose**: Standard transformer block with self-attention and MLP.

#### `__init__` Method

```python
def __init__(
    self,
    dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = False,
    drop: float = 0.0,
    attn_drop: float = 0.0,
    ...
)
```

**What it creates**:

1. **Layer Normalization**: `self.norm1` and `self.norm2`
   - Normalizes inputs before attention and MLP
   - Helps with training stability

2. **Multi-Head Attention**:
   ```python
   self.attn = nn.MultiheadAttention(
       dim, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True
   )
   ```
   - **Compatibility**: Tries `batch_first=True` (PyTorch >= 1.9), falls back if not available
   - **Why multi-head?**: Allows model to attend to different types of information simultaneously

3. **MLP (Feedforward Network)**:
   ```python
   self.mlp = nn.Sequential(
       nn.Linear(dim, mlp_hidden_dim),      # Expand: 768 -> 3072
       act_layer(),                          # GELU activation
       nn.Dropout(drop),                     # Regularization
       nn.Linear(mlp_hidden_dim, dim),       # Contract: 3072 -> 768
       nn.Dropout(drop)
   )
   ```
   - Hidden dimension: `dim × mlp_ratio` (typically 4x expansion)
   - **Why expansion?**: Provides model capacity for complex transformations

#### `forward` Method

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Self-attention with residual
    x_norm = self.norm1(x)
    attn_out, _ = self.attn(x_norm, x_norm, x_norm)
    x = x + attn_out  # Residual connection
    
    # MLP with residual
    x = x + self.mlp(self.norm2(x))  # Residual connection
    return x
```

**What it does**:
1. **Self-Attention**: 
   - Normalizes input
   - Computes attention (each patch attends to all patches)
   - Adds residual connection
   
2. **MLP**:
   - Normalizes input
   - Applies feedforward network
   - Adds residual connection

**Why residual connections?**: Help with gradient flow and enable deeper networks.

**Architecture**: `Input → Norm → Attention → +Input → Norm → MLP → +Input → Output`

---

### Class: `VisionTransformer`

**Purpose**: Vision Transformer encoder backbone for VideoMAE.

#### `__init__` Method

**What it creates**:

1. **Patch Embedding**: `self.patch_embed`
   - Converts video to patch embeddings

2. **Class Token**: `self.cls_token`
   - Learnable token (for compatibility, though not heavily used in MAE)
   - Shape: `(1, 1, embed_dim)`

3. **Positional Encoding**: `self.pos_embed`
   - Size: `num_patches + 1` (patches + class token)

4. **Transformer Blocks**: `self.blocks`
   - List of N transformer blocks (12 for ViT-S/B, 24 for ViT-L)

5. **Final Normalization**: `self.norm`

**Weight Initialization**:
- Class token: Truncated normal (std=0.02)
- Linear layers: Truncated normal (std=0.02)
- LayerNorm: Bias=0, Weight=1

**Why this initialization?**: Standard ViT initialization for stable training.

#### `forward` Method

```python
def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
    # Patch embedding
    x = self.patch_embed(x)  # (B, num_patches, embed_dim)
    
    # Add class token
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
    
    # Add positional encoding
    x = self.pos_embed(x)
    x = self.pos_drop(x)
    
    # Apply transformer blocks
    for block in self.blocks:
        x = block(x)
    
    x = self.norm(x)
    return x
```

**What it does**: Standard ViT forward pass (though in VideoMAE, masking happens before this).

---

### Class: `VideoMAE`

**Purpose**: Main VideoMAE model implementing masked autoencoding.

#### `__init__` Method

**What it creates**:

1. **Encoder**: Full Vision Transformer
   - Processes visible patches only
   - Heavy architecture (12-24 layers)

2. **Decoder**: Lightweight transformer
   - 4 layers (vs 12-24 in encoder)
   - Same embedding dimension as encoder
   - **Why lightweight?**: Only needs to reconstruct, not learn complex features

3. **Decoder Components**:
   - `decoder_embed`: Projects encoder outputs to decoder dimension
   - `decoder_pos_embed`: Positional encoding for decoder
   - `decoder_blocks`: 4 transformer blocks
   - `decoder_pred`: Linear layer to predict pixel values

**Backbone Configurations**:
```python
BACKBONE_CONFIGS = {
    'vit_s': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
    'vit_b': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
    'vit_l': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16}
}
```

#### `load_pretrained` Method

```python
def load_pretrained(self, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Remove 'model.' prefix if present
    # Load with strict=False to allow partial loading
    self.load_state_dict(new_state_dict, strict=False)
```

**What it does**: 
- Loads checkpoint from file
- Handles different checkpoint formats (PyTorch Lightning, standard, etc.)
- Removes common prefixes
- Uses `strict=False` to allow partial loading

**Why flexible loading?**: Different frameworks save checkpoints in different formats.

---

#### `random_masking` Method

```python
def random_masking(self, x: torch.Tensor, mask_ratio: float):
    B, N, D = x.shape  # (batch, num_patches, embed_dim)
    len_keep = int(N * (1 - mask_ratio))  # Number of visible patches
    
    # Random shuffle
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)  # Random permutation
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # Inverse permutation
    
    # Keep first len_keep patches
    ids_keep = ids_shuffle[:, :len_keep]
    
    # Get visible patches
    visible_patches = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    
    # Create mask (1 for masked, 0 for visible)
    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return visible_patches, mask, ids_restore, ids_keep
```

**What it does** (step by step):

1. **Calculate visible patches**: 
   - If `mask_ratio=0.75` and `N=1568` patches
   - `len_keep = 1568 × 0.25 = 392` visible patches
   - `1568 - 392 = 1176` masked patches

2. **Random Shuffle**:
   - Generates random noise for each patch
   - Sorts by noise to create random permutation
   - `ids_shuffle`: Random order of patch indices
   - `ids_restore`: Inverse mapping to restore original order

3. **Select Visible Patches**:
   - Takes first `len_keep` patches from shuffled order
   - Uses `torch.gather` to extract those patches

4. **Create Mask**:
   - Binary mask: `1` for masked, `0` for visible
   - Restored to original patch order using `ids_restore`

**Why random masking?**: Forces model to learn robust representations, not just memorize patterns.

**Returns**:
- `visible_patches`: `(B, 392, embed_dim)` - patches to encode
- `mask`: `(B, 1568)` - binary mask for loss computation
- `ids_restore`: `(B, 1568)` - indices to restore original order
- `ids_keep`: `(B, 392)` - indices of visible patches

---

#### `forward_encoder` Method

```python
def forward_encoder(self, x: torch.Tensor, mask_ratio: float):
    # Get patches
    patches = self.encoder.patch_embed(x)  # (B, 1568, embed_dim)
    
    # Apply masking
    visible_patches, mask, ids_restore, ids_keep = self.random_masking(patches, mask_ratio)
    # visible_patches: (B, 392, embed_dim)
    
    # Add class token
    cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
    
    # Get positional embeddings for visible patches
    pos_embed = self.encoder.pos_embed.pos_embed  # (1, 1569, embed_dim)
    cls_pos_embed = pos_embed[:, 0:1, :]  # Class token position
    
    # Get positional embeddings for visible patches
    ids_keep_pos = ids_keep + 1  # Shift by 1 (index 0 is class token)
    patch_pos_embed = torch.gather(
        pos_embed[:, 1:, :],  # Skip class token
        dim=1,
        index=ids_keep_pos.unsqueeze(-1).expand(-1, -1, embed_dim)
    )  # (B, 392, embed_dim)
    
    # Combine
    pos_embed_visible = torch.cat([cls_pos_embed.expand(B, -1, -1), patch_pos_embed], dim=1)
    visible_patches = torch.cat([cls_tokens, visible_patches], dim=1) + pos_embed_visible
    
    # Apply encoder
    for block in self.encoder.blocks:
        visible_patches = block(visible_patches)
    visible_patches = self.encoder.norm(visible_patches)
    
    # Remove class token
    encoded_patches = visible_patches[:, 1:, :]  # (B, 392, embed_dim)
    
    return encoded_patches, mask, ids_restore
```

**What it does**:

1. **Extract Patches**: Converts video to patch embeddings

2. **Apply Masking**: Randomly masks 75% of patches

3. **Handle Positional Embeddings** (Key Fix):
   - Full positional embedding: `(1, 1569, embed_dim)` for all patches + class token
   - Visible sequence: `(B, 393, embed_dim)` for visible patches + class token
   - **Solution**: Extract relevant positional embeddings:
     - Class token positional embedding (index 0)
     - Visible patch positional embeddings (using `ids_keep + 1`)
   - **Why +1?**: Positional embedding index 0 is for class token, patch indices start at 1

4. **Encode Visible Patches**: Passes through transformer blocks

5. **Remove Class Token**: Returns only patch encodings for decoder

**Why this approach?**: Positional embeddings must match the sequence length, so we extract only the ones we need.

---

#### `forward_decoder` Method

```python
def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
    # x: (B, 392, embed_dim) - encoded visible patches
    
    # Project to decoder dimension
    x = self.decoder_embed(x)  # (B, 392, decoder_embed_dim)
    
    # Restore full sequence
    B, len_keep, D = x.shape
    num_patches = self.encoder.patch_embed.num_patches  # 1568
    
    # Create full sequence with zeros for masked tokens
    x_full = torch.zeros(B, num_patches, D, device=x.device, dtype=x.dtype)
    x_full[:, :len_keep] = x  # Place visible patches at start
    
    # Restore original order
    x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))
    # Now: visible patches in correct positions, masked tokens are zeros
    
    # Add class token
    cls_token = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
    x_full = torch.cat([cls_token, x_full], dim=1)  # (B, 1569, D)
    
    # Add positional encoding
    x_full = self.decoder_pos_embed(x_full)
    
    # Apply decoder blocks
    for block in self.decoder_blocks:
        x_full = block(x_full)
    x_full = self.decoder_norm(x_full)
    
    # Remove class token
    x_full = x_full[:, 1:, :]  # (B, 1568, D)
    
    # Predict pixels
    pred = self.decoder_pred(x_full)  # (B, 1568, patch_pixels)
    
    return pred
```

**What it does** (step by step):

1. **Project to Decoder Dimension**: Linear projection of encoded patches

2. **Restore Full Sequence**:
   - Creates tensor of zeros: `(B, 1568, D)`
   - Places visible patches at start: `x_full[:, :392] = x`
   - Uses `ids_restore` to restore original patch order
   - **Result**: Visible patches in correct positions, masked tokens are zeros

3. **Add Class Token and Positional Encoding**

4. **Decode**: Passes through 4 transformer blocks

5. **Predict Pixels**: Linear layer predicts pixel values for each patch
   - Output: `(B, 1568, patch_pixels)`
   - `patch_pixels = 2 × 16 × 16 × 3 = 1536` (temporal × spatial × channels)

**Why zeros for masked tokens?**: Decoder learns to reconstruct from encoded visible patches + zero placeholders.

---

#### `patchify` Method

```python
def patchify(self, x: torch.Tensor) -> torch.Tensor:
    # Input: (B, C, T, H, W) = (B, 3, 16, 224, 224)
    B, C, T, H, W = x.shape
    t_patch_size = 2
    patch_size = 16
    
    # Reshape to patches
    x = x.reshape(B, C, T // t_patch_size, t_patch_size, 
                 H // patch_size, patch_size, 
                 W // patch_size, patch_size)
    # (B, 3, 8, 2, 14, 16, 14, 16)
    
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    # (B, 8, 14, 14, 3, 2, 16, 16)
    
    x = x.reshape(B, -1, C * t_patch_size * patch_size * patch_size)
    # (B, 1568, 1536)
    
    return x
```

**What it does**: Converts video to patches (same structure as patch embedding, but without projection).

**Why**: Creates target patches for loss computation - we need to compare predictions to original patch pixel values.

---

#### `compute_loss` Method

```python
def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    # pred: (B, 1568, 1536) - predicted patch pixels
    # target: (B, 1568, 1536) - target patch pixels
    # mask: (B, 1568) - binary mask (1=masked, 0=visible)
    
    if self.norm_pix_loss:
        # Normalize target patches
        mean = target.mean(dim=-1, keepdim=True)  # Per-patch mean
        var = target.var(dim=-1, keepdim=True)    # Per-patch variance
        target = (target - mean) / (var + 1e-6) ** 0.5  # Normalize
    
    # Compute MSE loss
    loss = (pred - target) ** 2  # (B, 1568, 1536)
    loss = loss.mean(dim=-1)      # (B, 1568) - mean over pixels
    
    # Apply mask (only compute loss on masked patches)
    loss = (loss * mask).sum() / mask.sum()  # Mean over masked patches only
    
    return loss
```

**What it does**:

1. **Optional Normalization**: 
   - If `norm_pix_loss=True`, normalizes each patch by its mean and variance
   - **Why?**: Helps with training stability, reduces effect of brightness variations

2. **Compute MSE**: Mean squared error between predicted and target pixels

3. **Apply Mask**: 
   - Only computes loss on masked patches (where `mask=1`)
   - **Why?**: Model should learn to reconstruct masked patches, not visible ones

**Loss Formula**:
```
loss = mean((pred - target)²) for masked patches only
```

---

#### `forward` Method

```python
def forward(self, x: torch.Tensor):
    # Input: (B, C, T, H, W)
    
    # Encoder: encode visible patches
    encoded_patches, mask, ids_restore = self.forward_encoder(x, self.mask_ratio)
    
    # Decoder: reconstruct all patches
    pred = self.forward_decoder(encoded_patches, ids_restore)
    
    # Compute loss
    target = self.patchify(x)
    loss = self.compute_loss(pred, target, mask)
    
    return loss, pred, mask
```

**What it does**: Complete forward pass through VideoMAE.

**Returns**:
- `loss`: Scalar reconstruction loss
- `pred`: Predicted patch pixels `(B, 1568, 1536)`
- `mask`: Binary mask `(B, 1568)`

---

## Training Module

### File: `training/lightning_module.py`

#### Class: `VideoMAELightningModule`

**Purpose**: PyTorch Lightning wrapper for VideoMAE training. Handles training/validation steps, optimizer configuration, and logging.

#### `__init__` Method

```python
def __init__(
    self,
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    warmup_steps: int = 1000,
    track_grad_norm: bool = False
):
    super().__init__()
    self.save_hyperparameters(ignore=['model'])
    
    self.model = model
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.warmup_steps = warmup_steps
    self.track_grad_norm = track_grad_norm
```

**What it does**:
- Wraps VideoMAE model
- Stores hyperparameters
- `save_hyperparameters()`: Saves hyperparameters to checkpoint for reproducibility

**Why PyTorch Lightning?**: Simplifies training loop, multi-GPU support, logging, checkpointing.

---

#### `forward` Method

```python
def forward(self, x: torch.Tensor) -> tuple:
    return self.model(x)
```

**What it does**: Simple wrapper around model's forward pass.

**Why**: PyTorch Lightning convention - `forward()` is used for inference.

---

#### `training_step` Method

```python
def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
    # Forward pass
    loss, pred, mask = self.model(batch)
    
    # Log training loss
    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    # Track gradient norm if enabled
    if self.track_grad_norm:
        grad_norm = self.compute_grad_norm()
        self.log('grad_norm', grad_norm, on_step=True, on_epoch=False, logger=True)
    
    return loss
```

**What it does**:

1. **Forward Pass**: Computes loss through model

2. **Logging**:
   - `on_step=True`: Logs every training step
   - `on_epoch=True`: Also logs epoch average
   - `prog_bar=True`: Shows in progress bar
   - `logger=True`: Saves to TensorBoard

3. **Gradient Norm Tracking** (optional):
   - Computes L2 norm of all gradients
   - Useful for monitoring gradient flow
   - Helps detect vanishing/exploding gradients

**Returns**: Loss tensor (PyTorch Lightning uses this for backpropagation).

---

#### `validation_step` Method

```python
def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
    # Forward pass (no gradients)
    loss, pred, mask = self.model(batch)
    
    # Log validation loss
    self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    return loss
```

**What it does**: Similar to training step, but:
- No gradient computation (automatic in validation)
- Only logs on epoch (not every step)
- Used for model selection and monitoring

---

#### `configure_optimizers` Method

```python
def configure_optimizers(self):
    optimizer = AdamWScheduleFree(
        self.parameters(),
        lr=self.learning_rate,
        weight_decay=self.weight_decay,
        warmup_steps=self.warmup_steps
    )
    return optimizer
```

**What it does**: Configures AdamWScheduleFree optimizer.

**AdamWScheduleFree Features**:
- **Schedule-free**: No learning rate scheduling needed
- Combines benefits of AdamW with schedule-free learning
- **Warmup**: Gradually increases learning rate over `warmup_steps`
- **Weight Decay**: L2 regularization for generalization

**Why schedule-free?**: Simplifies training - no need to tune learning rate schedules.

---

#### `compute_grad_norm` Method

```python
def compute_grad_norm(self) -> torch.Tensor:
    total_norm = 0.0
    
    for p in self.model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # L2 norm
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5  # Square root
    
    return torch.tensor(total_norm, device=self.device)
```

**What it does**: Computes L2 norm of all gradients.

**Formula**: `||g||₂ = sqrt(Σ ||g_i||₂²)`

**Why useful?**:
- **Vanishing gradients**: Very small norm → model not learning
- **Exploding gradients**: Very large norm → training unstable
- **Normal range**: Typically 0.1 - 10.0

---

## Training Script

### File: `training/train.py`

Main entry point for training. Handles argument parsing, configuration loading, and training setup.

---

#### `load_config` Function

```python
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
```

**What it does**: Loads YAML configuration file.

**Why**: Enables hyperparameter management via config files.

---

#### `parse_args` Function

**Purpose**: Parses command-line arguments and merges with config file.

**Process**:

1. **Define Arguments**: Creates argument parser with all hyperparameters

2. **Parse Arguments**: Gets command-line arguments (may be None)

3. **Load Config File**:
   - If `--config` specified, loads that file
   - Otherwise, tries default `config/config.yaml`
   - If not found, uses command-line args and defaults

4. **Merge Config and Arguments**:
   - **Priority**: Command-line args > Config file > Defaults
   - For each parameter:
     - If CLI arg is None, use config value
     - If config value is None, use default
     - CLI args always override config

**Why this approach?**: 
- Config files for stable experiments
- CLI args for quick overrides
- Best of both worlds

---

#### `create_data_loaders` Function

```python
def create_data_loaders(args):
    # Create datasets
    train_dataset = VideoDataset(...)
    val_dataset = VideoDataset(...)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(...)
    
    return train_loader, val_loader
```

**What it does**:

1. **Creates Datasets**: Training and validation VideoDataset instances

2. **Creates DataLoaders**:
   - **Batch Size**: Number of videos per batch
   - **Shuffle**: Randomize order (train only)
   - **Num Workers**: Parallel data loading processes
   - **Pin Memory**: Faster GPU transfer (if CUDA available)
   - **Drop Last**: Drop incomplete batches (train only)

**Why these settings?**:
- **Shuffle**: Data augmentation through random sampling
- **Num Workers**: Parallel loading speeds up training
- **Pin Memory**: Faster CPU→GPU transfer
- **Drop Last**: Consistent batch sizes

---

#### `create_model` Function

```python
def create_model(args):
    model = VideoMAE(
        backbone=args.backbone,
        img_size=args.img_size,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=args.norm_pix_loss,
        pretrained=args.pretrained
    )
    return model
```

**What it does**: Creates VideoMAE model with specified configuration.

**Why separate function?**: Clean separation, easier to test/modify.

---

#### `create_lightning_module` Function

```python
def create_lightning_module(args, model):
    lightning_module = VideoMAELightningModule(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        track_grad_norm=args.track_grad_norm
    )
    return lightning_module
```

**What it does**: Wraps model in PyTorch Lightning module.

---

#### `create_callbacks` Function

```python
def create_callbacks(args):
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f'{args.checkpoint_prefix}-{{epoch:02d}}-{{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks
```

**What it does**:

1. **ModelCheckpoint**:
   - **Dirpath**: Directory to save checkpoints
   - **Filename**: Pattern for checkpoint names
   - **Monitor**: Metric to track (`val_loss`)
   - **Mode**: `'min'` (minimize validation loss)
   - **Save Top K**: Keeps best 3 models
   - **Save Last**: Always saves last checkpoint
   - **Why?**: Enables model selection and resuming training

2. **LearningRateMonitor**:
   - Logs learning rate to TensorBoard
   - Useful even with schedule-free optimizer (for monitoring)

---

#### `main` Function

```python
def main():
    args = parse_args()
    
    # Set random seed
    pl.seed_everything(42)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args)
    
    # Create model
    model = create_model(args)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create lightning module
    lightning_module = create_lightning_module(args, model)
    
    # Create callbacks and logger
    callbacks = create_callbacks(args)
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.experiment_name)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus,
        strategy='ddp' if args.gpus > 1 else 'auto',
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
        gradient_clip_val=1.0,
        accumulate_grad_batches=1
    )
    
    # Start training
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from_checkpoint
    )
```

**What it does** (step by step):

1. **Parse Arguments**: Gets configuration from CLI and/or config file

2. **Set Random Seed**: Ensures reproducibility

3. **Create Data Loaders**: Training and validation datasets

4. **Create Model**: VideoMAE with specified backbone

5. **Count Parameters**: Prints model size for reference

6. **Create Lightning Module**: Wraps model for training

7. **Create Callbacks and Logger**:
   - Checkpoint callback for saving models
   - TensorBoard logger for monitoring

8. **Create Trainer**:
   - **Accelerator**: GPU or CPU
   - **Devices**: Number of GPUs
   - **Strategy**: DDP for multi-GPU, auto for single
   - **Precision**: 16 or 32 bit
   - **Validation**: Twice per epoch (0.5 interval)
   - **Gradient Clipping**: Prevents exploding gradients

9. **Start Training**: Calls `trainer.fit()`

**Trainer Configuration Explained**:
- **`val_check_interval=0.5`**: Validates halfway through each epoch
- **`gradient_clip_val=1.0`**: Clips gradients to max norm of 1.0
- **`strategy='ddp'`**: Distributed Data Parallel for multi-GPU
- **`precision=16`**: Mixed precision training (faster, less memory)

---

## Data Flow and Training Process

### Complete Data Flow

```
1. CSV File (video paths)
   ↓
2. VideoDataset.__getitem__()
   - Load video with torchvision.io.read_video()
   - Normalize to [0, 1] and rearrange to (T, C, H, W)
   - Pad if insufficient frames
   - Sample 32 consecutive frames (random start)
   - Temporal downsampling (stride=2) → 16 frames
   - Rearrange to (C, T, H, W)
   - Tensor: (3, 16, 224, 224)
   ↓
3. DataLoader batches: (B, 3, 16, 224, 224)
   ↓
4. VideoMAE.forward()
   ↓
5. Patch Embedding → (B, 1568, embed_dim)
   ↓
6. Random Masking → Visible: (B, 392, embed_dim), Mask: (B, 1568)
   ↓
7. Encoder → Encoded: (B, 392, embed_dim)
   ↓
8. Decoder → Predictions: (B, 1568, 1536)
   ↓
9. Loss Computation (masked patches only)
   ↓
10. Backpropagation
```

### Training Loop

```
For each epoch:
    For each training batch:
        1. Load video batch (B, 3, 16, 224, 224)
        2. Forward pass:
           - Mask 75% of patches
           - Encode visible patches
           - Decode all patches
           - Compute loss on masked patches
        3. Backward pass (compute gradients)
        4. Optimizer step (AdamWScheduleFree)
        5. Log metrics (train_loss, grad_norm)
    
    For each validation batch (twice per epoch):
        1. Load video batch
        2. Forward pass (no gradients)
        3. Compute loss
        4. Log metrics (val_loss)
    
    Save checkpoint (if best or last)
```

### Key Design Decisions

1. **High Masking Ratio (75%)**: Forces model to learn strong representations
2. **Asymmetric Architecture**: Heavy encoder, light decoder (computational efficiency)
3. **3D Patch Embedding**: Captures spatio-temporal information together
4. **Positional Embedding Extraction**: Handles variable sequence lengths correctly
5. **Loss on Masked Patches Only**: Model learns reconstruction, not memorization
6. **PyTorch Lightning**: Simplifies multi-GPU, logging, checkpointing
7. **Schedule-Free Optimizer**: Reduces hyperparameter tuning

---

## Summary

This codebase implements a complete VideoMAE training pipeline with:

- **Flexible Data Loading**: Handles variable-length videos with proper sampling
- **Modular Architecture**: Separate components for easy modification
- **Multiple Backbones**: Support for ViT-S, ViT-B, and ViT-L
- **EVEREST Training**: Implements masked autoencoding for video
- **Production-Ready**: PyTorch Lightning for easy scaling
- **Well-Documented**: Comprehensive comments throughout
- **Configurable**: Easy hyperparameter tuning via YAML and CLI

The implementation follows best practices for deep learning research code, with proper separation of concerns, error handling, and extensive documentation.

