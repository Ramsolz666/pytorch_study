import torch
import torchvision
import torchvision.models as models
import time
import numpy as np

def test_on_device(model, dump_inputs, warn_up, loops, device_type):
    if device_type == 'cuda':
        assert torch.cuda.is_available()
    device = torch.device(device_type)

    # model = models.alexnet.alexnet(pretrained=False).to(device)
    model.to(device)
    model.eval()
    dump_inputs = dump_inputs.to(device)

    with torch.no_grad():
        executions = []
        for i in range(warn_up + loops):
            if device_type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            _ = model(dump_inputs)
            if device_type == 'cuda':
                torch.cuda.synchronize() # CUDA sync
            end = time.time()
            executions.append((end-start)*1000) # ms
    # print(f'Avg time:{np.mean(executions)} ms')
    return np.mean(executions[warn_up:])


if __name__ == "__main__":
    # print(torch.cuda.is_available())
    model_list = {
        'AlexNet': models.alexnet(),
        'ResNet-50': models.resnet50(),
        'ResNet-18': models.resnet18(),
        'ResNet-101': models.resnet101(),
        'MobileNet-v2':models.mobilenet_v2(),
        'SqueezeNet1-1': models.squeezenet1_1()
    }

    batch_size = 1
    for name, model in model_list.items():
        print('='*10+f'{name}'+'='*10)
        avg_time = test_on_device(model=model, dump_inputs=torch.rand(batch_size, 3, 224, 224), warn_up=3, loops=10, device_type='cuda')
        print(f'Avg time:{avg_time} ms')

    

    

