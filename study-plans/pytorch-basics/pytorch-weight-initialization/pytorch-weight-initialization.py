import torch

def initialize_weights(fan_in: int, fan_out: int, method: str) -> torch.Tensor:
    """
    Returns a float32 weight tensor with shape (fan_out, fan_in).
    """
    w = torch.empty(fan_out, fan_in)
    match method:
        case "xavier_uniform":
            return w.uniform_(-(6/(fan_in + fan_out)) ** 0.5, (6/(fan_in + fan_out)) ** 0.5)
            
        case "xavier_normal":
            return w.normal_(mean=0, std=(float(2) / (fan_in + fan_out))**0.5)
            
        case "he_uniform":
            return w.uniform_(-(6/(fan_in)) ** 0.5, (6/(fan_in)) ** 0.5)
            
        case "he_normal":
            return w.normal_(0, (float(2) / fan_in)**0.5)
            
