import json

import coremltools as ct
import torch
from coremltools.proto import FeatureTypes_pb2 as ft

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


# 　トレーニング済みPytorchモデルからTorchScriptファイルを生成
#   その後MLモデルに変換して保存
def my_convert(input_type="tensor"):
    net = PoseEstimationWithMobileNet()
    checkpoint_path = '../resources/checkpoint_iter_370000.pth'  # checkpoint_iter_370000.pthのある場所

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

    input_image = None
    model_name = None

    if input_type == "tensor":
        input_image = ct.TensorType(
            name="my_input", shape=(1, 3, 256, 456),
        )
        model_name = "../resources/SampleTensorType.mlmodel"
    elif input_type == "image":
        input_image = ct.ImageType(
            name="my_input", shape=(1, 3, 256, 456),  # scale=1 / 255,
        )
        model_name = "../resources/SampleImageType.mlmodel"

    # Pytorchモデルではoutputs引数を入れるとエラーがでる？
    model = ct.convert(
        traced_model,
        inputs=[input_image],
        # useCPUOnly=True,
    )

    model.save(model_name)


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


def rename_outputs(modelName: str):
    model = ct.models.MLModel(modelName)
    spec = model.get_spec()
    # output_names = [out.name for out in spec.description.output]
    ct.utils.rename_feature(spec, "var_489", "heat_map_1")
    ct.utils.rename_feature(spec, "var_508", "paf_1")
    ct.utils.rename_feature(spec, "var_768", "heat_map_2")
    ct.utils.rename_feature(spec, "var_787", "paf_2")
    output_names = [out.name for out in spec.description.output]
    print(output_names)

    model = ct.models.MLModel(spec)
    model.save(f"{modelName}")


if __name__ == '__main__':
    # 入力にMLMultiArrayを持つmlmodelを生成
    # my_convert("tensor")

    # 入力にImageを持つmlmodelを生成
    # my_convert("image")

    # 出力のプロパティ名を変更
    rename_outputs("../resources/SampleTensorType.mlmodel")
    rename_outputs("../resources/SampleImageType.mlmodel")
