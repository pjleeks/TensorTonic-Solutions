import torch

def info_nce_loss(z1, z2, temperature=0.1):
    # Convert lists to tensors
    z1 = torch.tensor(z1, dtype=torch.float32)
    z2 = torch.tensor(z2, dtype=torch.float32)
    
    # 1. Similarity Matrix (Raw dot product)
    # For vectors like [1,0] and [0,1], no normalization is needed to see the logic,
    # but in SimCLR/CLIP, we'd use F.normalize(z) here.
    sim_matrix = torch.matmul(z1, z2.T) 
    
    # 2. Scale by temperature
    logits = sim_matrix / temperature
    
    # 3. Targets are the diagonal indices
    # Row 0 should match Col 0, Row 1 matches Col 1
    labels = torch.arange(z1.shape[0])
    
    # 4. Cross Entropy (includes Softmax + Log)
    return torch.nn.functional.cross_entropy(logits, labels)

# --- Test Cases ---
# Case 1: Perfect Alignment -> Loss ~0.0
# Logits diag will be 1/0.1 = 10, off-diag will be 0/0.1 = 0
print(f"Perfect: {info_nce_loss([[1,0],[0,1]], [[1,0],[0,1]], 0.1):.6f}")

# Case 2: Misaligned -> Loss ~10.0
# Logits diag will be 0/0.1 = 0, off-diag will be 1/0.1 = 10
# CrossEntropy(-log(e^0 / (e^0 + e^10))) ≈ 10
print(f"Misaligned: {info_nce_loss([[1,0],[0,1]], [[0,1],[1,0]], 0.1):.6f}")

# Case 3: High Temp -> Loss ~0.31
# Logits diag will be 1/1.0 = 1, off-diag will be 0/1.0 = 0
print(f"High Temp: {info_nce_loss([[1,0],[0,1]], [[1,0],[0,1]], 1.0):.6f}")