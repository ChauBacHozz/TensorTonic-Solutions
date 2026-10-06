import torch

def gradient_accumulation(w_init: torch.Tensor, micro_batches: list, lr: float, accum_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (final_weights, last_mean_gradient) as float32 tensors.
    """
    w = w_init.detach().clone().requires_grad_()

    last_grad = torch.zeros_like(w_init)

    total_steps = len(micro_batches)
    for index, (input, target) in enumerate(micro_batches, start=1):
        y_pred = input @ w.T
        loss = ((y_pred - target)**2).mean()
        loss /= accum_steps
        loss.backward()

        if index % accum_steps == 0 or index == total_steps:
            last_grad = w.grad.detach().clone()

            with torch.no_grad():
                w -= lr * w.grad

            w.grad.zero_()

    return w.detach().to(torch.float32), last_grad