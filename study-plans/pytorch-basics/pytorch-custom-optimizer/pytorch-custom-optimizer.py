import torch

class CustomSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Nếu momentum > 0: dùng buffer theo chuẩn PyTorch
                if momentum != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        # Bước đầu tiên: buffer clone từ grad
                        buf = state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = state['momentum_buffer']
                        # buf = momentum * buf + grad
                        buf.mul_(momentum).add_(grad)

                    # p = p - lr * buf
                    p.add_(buf, alpha=-lr)
                else:
                    # SGD thuần (momentum = 0): p = p - lr * grad
                    p.add_(grad, alpha=-lr)

        return loss