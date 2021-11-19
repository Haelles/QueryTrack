import torch.nn as nn
import torch
from mmcv.runner import auto_fp16, force_fp32


def demo():
    class ExampleModule(nn.Module):
        @auto_fp16(apply_to=('x'))
        def forward(self, x, y):
            return torch.cat([x, y])

    device = torch.device("cuda:0")
    model = ExampleModule().to(device)
    model.fp16_enabled = False
    # model.fp16_enabled = True
    input_x = torch.rand((2, 3), dtype=torch.float32).to(device)
    input_y = torch.rand((2, 3), dtype=torch.float32).to(device)
    output_x = model(input_x, input_y)
    print(output_x.dtype)


if __name__ == '__main__':
    demo()

