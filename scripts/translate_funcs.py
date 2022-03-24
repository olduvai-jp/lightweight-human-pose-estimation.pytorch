import json

import coremltools as ct
import torch
from coremltools.proto import FeatureTypes_pb2 as ft

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


# 　トレーニング済みPytorchモデルからTorchScriptファイルを生成
def my_convert():
    net = PoseEstimationWithMobileNet()
    checkpoint_path = ''  # checkpoint_iter_370000.pthのある場所

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    load_state(net, checkpoint)

    # Dropout層などを消したいので評価モードに
    net.eval()
    # print(net.state_dict().keys())

    # PytorchのDefine-by-runの仕様により、サンプル入力を通さないと
    # ネットワークのグラフが出せないため、通す必要がある
    example_input = torch.rand(1, 3, 256, 456)
    traced_model = torch.jit.trace(net, example_input)

    input_image = ct.ImageType(
        name="my_input", shape=(1, 3, 256, 456), scale=1 / 255,
    )

    # Pytorchモデルではoutputs引数を入れるとエラーがでる？
    model = ct.convert(
        traced_model,
        inputs=[input_image],
        # useCPUOnly=True,
    )

    model.save('sample.mlmodel')


# InputをMultiArray型で生成するとXcodeで使いにくいので変換
def change_shape(modelname):
    spec = ct.utils.load_spec(modelname)
    builder = ct.models.neural_network.NeuralNetworkBuilder(spec=spec)
    builder.inspect_input_features()

    nn_spec = builder.spec
    # nn_spec.description.input[0].type.imageType.height = 256
    # nn_spec.description.input[0].type.imageType.width = 456

    # outputの名前を変えたい場合は下のようにする
    # ct.utils.rename_feature(nn_spec, 'var_767', 'my_output')

    # print(nn_spec.description.output)

    scale = ft.ImageFeatureType.ColorSpace.Value('RGB')
    nn_spec.description.input[0].type.imageType.colorSpace = scale

    # builder.inspect_input_features()
    new_model = ct.models.MLModel(nn_spec)

    # poseEstimationタイプでXcode上でPreviewする時に必要？
    new_model.user_defined_metadata["com.apple.coreml.model.preview.type"] = "poseEstimation"
    params_json = {"width_multiplier": 1.0, "output_stride": 16}
    new_model.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(params_json)

    new_model.save(modelname)


# PoseNetのサンプルモデルにプレビューを持たせられるかテスト
def create_sample_model():
    model = ct.models.MLModel("PoseNetMobileNet075S16FP16.mlmodel")
    model.user_defined_metadata["com.apple.coreml.model.preview.type"] = "poseEstimation"
    params_json = {"width_multiplier": 1.0, "output_stride": 16}
    model.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(params_json)
    model.save("posenet_with_preview_type.mlmodel")


if __name__ == '__main__':
    my_convert()
    # change_shape()
