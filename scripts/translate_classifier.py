import urllib

import coremltools as ct
import torch
import torchvision


# PyTorchの事前学習済みクラス分類モデルをmlmodelに変換する
# Xcode上でPreview可能
# https://coremltools.readme.io/docs/introductory-quickstart
from coremltools import convert

torch_model = torchvision.models.mobilenet_v2(pretrained=True)
torch_model.eval()

example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(torch_model, example_input)
out = traced_model(example_input)

label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
class_labels = urllib.request.urlopen(label_url).read().decode("utf-8").splitlines()
class_labels = class_labels[1:]  # remove the first class which is background
assert len(class_labels) == 1000

scale = 1 / (0.226 * 255.0)
bias = [- 0.485 / (0.229), - 0.456 / (0.224), - 0.406 / (0.225)]

image_input = ct.ImageType(name="input_1",
                           shape=example_input.shape,
                           scale=scale, bias=bias)

model = convert(
    traced_model,
    inputs=[image_input],
    classifier_config=ct.ClassifierConfig(class_labels),
    compute_units=ct.ComputeUnit.CPU_ONLY,
)

model.save("MyMobileNet.mlmodel")

print("model converted and saved")

# 以下はPythonでの確認用

#
# img_path = "daisy.jpeg"
# img = PIL.Image.open(img_path)
# img = img.resize([224, 224], PIL.Image.ANTIALIAS)
#
# # Get the protobuf spec of the model.
# spec = model.get_spec()
# for out in spec.description.output:
#     if out.type.WhichOneof('Type') == "dictionaryType":
#         coreml_dict_name = out.name
#         break
#
# # Make a prediction with the Core ML version of the model.
# coreml_out_dict = model.predict({"input_1": img})
# print("coreml predictions: ")
# print("top class label: ", coreml_out_dict["classLabel"])
#
# coreml_prob_dict = coreml_out_dict[coreml_dict_name]
#
# values_vector = np.array(list(coreml_prob_dict.values()))
# keys_vector = list(coreml_prob_dict.keys())
# top_3_indices_coreml = np.argsort(-values_vector)[:3]
# for i in range(3):
#     idx = top_3_indices_coreml[i]
#     score_value = values_vector[idx]
#     class_id = keys_vector[idx]
#     print("class name: {}, raw score value: {}".format(class_id, score_value))
