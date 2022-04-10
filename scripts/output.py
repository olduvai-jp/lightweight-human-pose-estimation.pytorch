# PytorchモデルとMLモデルの推論結果が一致するかを
# Python側から確認するためのスクリプト
# 参考: https://coremltools.readme.io/docs/model-prediction

import coremltools as ct
import cv2
import torch
from PIL import Image

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


def get_posenet_model(device='cpu'):
    model = PoseEstimationWithMobileNet()
    checkpoint_path = '../resources/checkpoint_iter_370000.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    load_state(model, checkpoint)

    model.eval()
    return model


def sample_output(input, device='cpu'):
    model = get_posenet_model(device)

    ys = model(input)

    print("~~~ Torch Model ~~~ ")
    for y in ys:
        print(f"Shape: {list(y.shape)}")
        print(f"Sum: {y.sum().item()}")

        print()


def converted_output(input):
    use_cpu = True if torch.cuda.is_available() else False
    model = ct.models.MLModel('../resources/sample.mlmodel', useCPUOnly=use_cpu)

    # Convertした際に
    # input_image = ct.TensorType(name=..., shape=(...))
    # などとした場合はnumpy配列に直す必要があるらしい...
    predictions = model.predict({'my_input': input.to('cpu').numpy()})

    keys = predictions.keys()

    print('~~~ ML Model ~~~')
    for key in keys:
        values = predictions[key]
        print(f"[{key}]")
        print(f"Shape: {values.shape}")
        print(f"Sum: {values.sum()}")

        print()


def compare_outputs(input: torch.Tensor):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sample_output(input, device)
    print()
    converted_output(input)


if __name__ == '__main__':
    isImage = True

    example_input = None

    if isImage:
        img = cv2.imread("../resources/matsuo_w.jpg", cv2.IMREAD_COLOR)
        tensor = torch.from_numpy(img).float()
        # unsqueeze: 次元拡張
        # permute: 次元入れ替え
        example_input = tensor.unsqueeze(0).permute(0, 3, 1, 2)
    else:
        example_input = torch.ones(1, 3, 256, 456)

    compare_outputs(example_input)
