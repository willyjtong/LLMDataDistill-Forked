import torch
import torch_directml


def select_device(device: str = ''):
    if device.lower() == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda:0')
    elif device.lower() == 'dml' and torch_directml.is_available():
        return torch_directml.device(torch_directml.default_device())
    else:
        return torch.device('cpu')

if __name__ == '__main__':
    device = select_device('dml')
    print(device)