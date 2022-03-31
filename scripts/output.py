# PytorchモデルとMLモデルの推論結果が一致するかを
# Python側から確認するためのスクリプト

import coremltools as ct
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


def sample_output(input, device='cpu'):
    model = PoseEstimationWithMobileNet()

    checkpoint_path = '../checkpoints/checkpoint_iter_370000.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    load_state(model, checkpoint)

    model.eval()

    ys = model(input)

    print("Output [Sample]")
    for y in ys:
        print(y.sum().item())


def converted_output(input):
    use_cpu = True if torch.cuda.is_available() else False
    model = ct.models.MLModel('sample.mlmodel', useCPUOnly=use_cpu)

    # Convertした際に
    # input_image = ct.TensorType(name=..., shape=(...))
    # などとした場合はnumpy配列に直す必要があるらしい...
    predictions = model.predict({'my_input': input.to('cpu').numpy()})

    keys = predictions.keys()

    print('Output [Converted]')
    for key in keys:
        print(predictions[key].sum())


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 全て１のサンプル入力
    example_input = torch.ones(1, 3, 256, 456)

    sample_output(example_input, device)
    print()
    converted_output(example_input)
