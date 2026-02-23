import torch
import math

def compute_spectral_stats(tensor):
    """Computes spectral statistics of a 2D tensor on CPU to save VRAM."""
    with torch.no_grad():
        if tensor.ndim < 2:
            return {"max": 0.0, "mean": 0.0, "gap": 0.0, "stable_rank": 0.0}
        
        # Flatten to 2D if needed (e.g. for conv or multiple heads)
        t = tensor.view(-1, tensor.size(-1)).detach().cpu().float()
        
        # We need at least 2 singular values for gap
        if t.size(0) < 2 or t.size(1) < 2:
            s = torch.linalg.svdvals(t)
            spectral_max = s[0].item() if len(s) > 0 else 0.0
            return {
                "max": spectral_max,
                "mean": s.mean().item() if len(s) > 0 else 0.0,
                "gap": 1.0,
                "stable_rank": 1.0
            }

        s = torch.linalg.svdvals(t)
        
        spectral_max = s[0].item()
        spectral_mean = s.mean().item()
        spectral_gap = (s[0] / s[1]).item() if len(s) > 1 else 1.0
        stable_rank = (torch.norm(t, p='fro')**2 / (s[0]**2)).item() if spectral_max > 0 else 1.0
        
        return {
            "max": spectral_max,
            "mean": spectral_mean,
            "gap": spectral_gap,
            "stable_rank": stable_rank
        }

def compute_singular_values(tensor, n=10):
    """Returns top n singular values (computed on CPU)."""
    with torch.no_grad():
        if tensor.ndim < 2:
            return [0.0] * n
        t = tensor.view(-1, tensor.size(-1)).detach().cpu().float()
        s = torch.linalg.svdvals(t)
        vals = s[:n].tolist()
        if len(vals) < n:
            vals += [0.0] * (n - len(vals))
        return vals

def compute_subspace_alignment(W, delta_W, k=5):
    """
    Measures how much the top-k singular directions of the update 
    align with the top-k singular directions of the weight.
    Returns: (left_alignment, right_alignment) - Average cosine of principal angles [0, 1]
    """
    with torch.no_grad():
        # Handle 1D or skinny tensors
        if W.ndim < 2 or W.size(0) < k or W.size(1) < k:
            return 0.0, 0.0
            
        W_2d = W.view(-1, W.size(-1)).detach().cpu().float()
        dW_2d = delta_W.view(-1, delta_W.size(-1)).detach().cpu().float()
        
        # SVD for both (left and right singular vectors)
        try:
            U_w, _, V_w = torch.linalg.svd(W_2d, full_matrices=False)
            U_dw, _, V_dw = torch.linalg.svd(dW_2d, full_matrices=False)
            
            # Principal angles between top-k subspaces (left/output subspace)
            M_left = U_w[:, :k].mT @ U_dw[:, :k]
            cosines_left = torch.linalg.svdvals(M_left)
            left_align = cosines_left.mean().item()
            
            # Principal angles between top-k subspaces (right/input subspace)
            # V_w shape is (K, N) where K is min(M,N), N is width. V_w.mT is (N, K).
            # The top-k right singular vectors are the first k rows of V (or cols of V.mT).
            M_right = V_w[:k, :] @ V_dw[:k, :].mT
            cosines_right = torch.linalg.svdvals(M_right)
            right_align = cosines_right.mean().item()
            
            return left_align, right_align
        except Exception:
            return 0.0, 0.0

def compute_spectral_entropy(tensor):
    """Computes Shannon entropy of the squared singular value distribution."""
    with torch.no_grad():
        if tensor.ndim < 2:
            return 0.0
        t = tensor.view(-1, tensor.size(-1)).detach().cpu().float()
        s = torch.linalg.svdvals(t)
        if len(s) == 0:
            return 0.0
            
        p = (s**2) / (torch.sum(s**2) + 1e-10)
        p = p[p > 1e-8] # Stability
        entropy = -torch.sum(p * torch.log(p)).item()
        # Normalize by max possible entropy (log of rank)
        max_entropy = math.log(len(s)) if len(s) > 1 else 1.0
        return entropy / max_entropy

def compute_orthogonality_error(tensor):
    """Measures || (W/||W||_2)^T (W/||W||_2) - I ||_F / sqrt(d)"""
    with torch.no_grad():
        if tensor.ndim < 2:
            return 0.0
        t = tensor.view(-1, tensor.size(-1)).detach().cpu().float()
        
        # SVD to get spectral norm for normalization
        s = torch.linalg.svdvals(t)
        if len(s) == 0 or s[0] == 0:
            return 1.0
            
        t_norm = t / s[0]
        
        if t_norm.size(0) >= t_norm.size(1):
            m = t_norm.mT @ t_norm
        else:
            m = t_norm @ t_norm.mT
            
        eye = torch.eye(m.size(0))
        err = torch.norm(m - eye, p='fro').item() / math.sqrt(m.size(0))
        return err
def compute_effective_rank(tensor):
    """Computes effective rank: exp(H(singular_values))"""
    with torch.no_grad():
        if tensor.ndim < 2:
            return 0.0
        t = tensor.view(-1, tensor.size(-1)).detach().cpu().float()
        s = torch.linalg.svdvals(t)
        if len(s) == 0 or s.sum() == 0:
            return 0.0
        p = s / s.sum()
        # Shannon entropy
        ent = -torch.sum(p * torch.log(p + 1e-10))
        return torch.exp(ent).item()

def compute_full_singular_values(tensor):
    """Returns all singular values (computed on CPU)."""
    with torch.no_grad():
        if tensor.ndim < 2:
            return [0.0]
        t = tensor.view(-1, tensor.size(-1)).detach().cpu().float()
        s = torch.linalg.svdvals(t)
        return s.tolist()
