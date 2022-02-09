# OpenVINO™ Model Analyzer

Model Analyzer is the tool for estimating theoretical information on deep learning models layers.


## 📝 Description

Model Analyzer is used to estimate theoretical information on your model, such as the number of operations, memory consumption, and other characteristics. 

> **NOTE**: Model Analyzer works only with models in [Intermediate Representation](https://docs.openvino.ai/latest/openvino_docs_MO_DG_IR_and_opsets.html#intermediate_representation_used_in_openvino) (IR) 
> format. Refer to the [Model Optimizer](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) documentation to learn how to obtain IR model. 

The tool analyzes the following parameters:

Parameter | Explanation | Unit of Measurement| Example
---|---|---|---
GFlop | Total number of floating-point operations required to infer a model. Summed up over known layers only.| Number of operations| 0.88418 × 10<sup>9</sup>
GIop | Total number of integer operations required to infer a model. Summed up over known layers only.| Number of operations| 0.86748 × 10<sup>9</sup>
Total number of weights|Total number of trainable model parameters excluding custom constants. Summed up over known layers only.|Number of weights| 3.489 × 10<sup>6</sup>
Minimum Memory Consumption |Theoretical minimum of memory used by a model for inference given that the memory is reused as much as possible.  Minimum Memory Consumption does not depend on weights.|Number of activations|2.408 × 10<sup>6</sup>
Maximum Memory Consumption |Theoretical maximum of memory used by a model for inference given that the memory is not reused, meaning all internal feature maps are stored in the memory simultaneously. Maximum Memory Consumption does not depend on weights.|Number of activations| 26.833 × 10<sup>6</sup>
Sparsity |Percentage of zero weights| Percentage|20%


## ⚙️  Command-Line Interface

Run this command to get a list of available arguments:

```shell
python3 model_analyzer.py -h
```

Argument | Explanation |Example
---|---|---
`-h`, `--help`|Displays help message.|`N/A`|
`-m`, `--model`|Path to an .xml file of the Intermediate Representation (IR) model. Or path to .onnx or .prototxt file of ONNX model.|`./public/model/FP16/model.xml`|
`-w`, `--weights`|Path to the .bin file of the Intermediate Representation (IR) model. If not specified, the weights file name is expected to be the same as the .xml file passed with `--model` option.|`./public/model/FP16/model.bin`|
`--ignore-unknown-layers` | Ignores unknown types of layers when counting GFLOPs.|`N/A`|
`-o`, `--report-dir`|Output directory.|`/Home/user/Public/report`|
`--model-report`|Name for the file where theoretical analysis results are stored.|`model_report.csv`|
`--per-layer-mode`| Enables collecting per-layer complexity metrics.|`N/A`|
`--per-layer-report`| File name for the per-layer complexity metrics. Should be specified only when `--per-layer-mode` option is used.|`per_layer_report.csv`|
`--sparsity-ignored-layers`| Specifies ignored layers names separated by comma.|`Constant, Elu, Split`|
`--sparsity-ignore-first-conv` | Ignores first Convolution layer for sparsity computation.|`N/A`|
`--sparsity-ignore-fc`|  Ignores FullyConnected layers for sparsity computation.|`N/A`|

Model Analyzer supports counting GFLOPs and GIOPs for the following layers:

<details>
<summary>List of Supported Layers</summary>

- Acosh - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_arithmetic_Acosh_3.html)
- Asinh - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_arithmetic_Asinh_3.html)
- Atanh - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_arithmetic_Atanh_3.html)
- Add - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_arithmetic_Add_1.html)
- ArgMax
- AvgPool - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_pooling_AvgPool_1.html)
- BatchNormalization
- BinaryConvolution - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_convolution_BinaryConvolution_1.html)
- Clamp - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_activation_Clamp_1.html)
- Concat - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_movement_Concat_1.html)
- Const - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_infrastructure_Constant_1.html)
- Constant - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_infrastructure_Constant_1.html)
- Convolution - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_convolution_Convolution_1.html)
- ConvolutionBackPropData - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_convolution_ConvolutionBackpropData_1.html)
- Crop
- Deconvolution - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_convolution_ConvolutionBackpropData_1.html)
- DeformableConvolution - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_convolution_DeformableConvolution_1.html)
- Divide - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_arithmetic_Divide_1.html)
- Eltwise
- Elu - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_activation_Elu_1.html)
- Exp- [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_activation_Exp_1.html)
- FullyConnected
- GEMM - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_matrix_MatMul_1.html)
- GRN - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_normalization_GRN_1.html)
- Gather - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_movement_Gather_7.html)
- GatherND - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_movement_GatherND_5.html)
- Greater - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_comparison_Greater_1.html)
- GreaterEqual - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_comparison_GreaterEqual_1.html)
- GroupConvolution - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_convolution_GroupConvolution_1.html)
- GroupConvolutionBackpropData - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_convolution_GroupConvolutionBackpropData_1.html)
- HSigmoid - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_activation_HSigmoid_5.html)
- HSwish - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_activation_HSwish_4.html)
- Input [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_infrastructure_Parameter_1.html)
- Interp - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_image_Interpolate_4.html)
- Less - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_comparison_Less_1.html)
- LessEqual - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_comparison_LessEqual_1.html)
- Log - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_arithmetic_Log_1.html)
- MVN - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_normalization_MVN_6.html)
- MatMul - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_matrix_MatMul_1.html)
- MaxPool - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_pooling_MaxPool_1.html)
- Mish - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_activation_Mish_4.html)
- Multiply - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_arithmetic_Multiply_1.html)
- Norm 
- Normalize  - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_normalization_NormalizeL2_1.html)
- NormalizeL2 - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_normalization_NormalizeL2_1.html)
- OneHot - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_sequence_OneHot_1.html)
- Output - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_infrastructure_Result_1.html)
- PReLU - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_activation_PReLU_1.html)
- PSROIPooling - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_detection_PSROIPooling_1.html)
- Pad - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_movement_Pad_1.html)
- Parameter - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_infrastructure_Parameter_1.html)
- Permute 
- Pooling - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_pooling_MaxPool_1.html)
- Power - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_arithmetic_Power_1.html)
- Priorbox - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_detection_PriorBox_1.html)
- PriorboxClustered - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_detection_PriorBoxClustered_1.html)
- Proposal - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_detection_Proposal_4.html)
- ROIPooling - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_detection_ROIPooling_1.html)
- Range - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_generation_Range_4.html)
- ReLu - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_activation_ReLU_1.html)
- ReduceL1 - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_reduction_ReduceL1_4.html)
- ReduceL2 - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_reduction_ReduceL2_4.html)
- ReduceMin - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_reduction_ReduceMin_1.html)
- Reshape - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_shape_Reshape_1.html)
- Result - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_infrastructure_Result_1.html)
- ReverseSequence - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_movement_ReverseSequence_1.html)
- ScaleShift 
- ScatterNDUpdate - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_movement_ScatterNDUpdate_3.html)
- Select - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_condition_Select_1.html)
- Sigmoid - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_activation_Sigmoid_1.html)
- Softmax - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_activation_SoftMax_1.html)
- SoftPlus - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_activation_SoftPlus_4.html)
- SparseToDense 
- Split - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_movement_Split_1.html)
- Squeeze - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_shape_Squeeze_1.html)
- StridedSlice - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_movement_StridedSlice_1.html)
- Subtract - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_arithmetic_Subtract_1.html)
- Swish - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_activation_Swish_4.html)
- Tanh - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_arithmetic_Tanh_1.html)
- Tile - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_movement_Tile_1.html)
- Unsqueeze - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_shape_Unsqueeze_1.html)
</details>

## 💻 Run Model Analyzer

This is an example of running Model Analyzer with mobilenet-v2 model: 

1. Install [OpenVINO Toolkit Developer Package](https://pypi.org/project/openvino-dev/):
```shell
python3 pip install openvino-dev
```

2. Clone the repository:
```sh
git clone git@github.com:openvinotoolkit/model_analyzer.git
cd model_analyzer
```

2. Download a model with [OMZ Model Downloader](https://docs.openvino.ai/latest/omz_tools_downloader.html):
```shell
omz_downloader --name mobilenet-v2
```

3. Convert the model to IR with [Model Converter](https://docs.openvino.ai/latest/omz_tools_downloader.html#model_converter_usage):
```shell
omz_converter --name mobilenet-v2 
```

4. Run Model Analyzer:
```sh
 python3 model_analyzer.py  --model ./public/mobilenet-v2/FP16/mobilenet-v2.xml --ignore-unknown-layers
```
You will get the following output: 
```sh
OUTPUT:
[ INFO ] Loading network files:
	public/mobilenet-v2/FP16/mobilenet-v2.xml
	public/mobilenet-v2/FP16/mobilenet-v2.bin
Warning, GOPS for layer(s) wasn't counted - ReduceMean
[ INFO ] GFLOPs: 0.8842
[ INFO ] GIOPs: 0.0000
[ INFO ] MParams: 3.4889
[ INFO ] Sparsity: 0.0001
[ INFO ] Minimum memory consumption: 2.4084
[ INFO ] Maximum memory consumption: 26.8328
[ INFO ] Guessed type: classificator
[ INFO ] Network status information file name: /home/user/model_analyzer/model_report.csv
```

Find the instructions for contributors in the [DEVELOPER.md](https://github.com/openvinotoolkit/model_analyzer/blob/master/DEVELOPER.md) document.

## ⚠️ Limitations

1. Note that Model Analyzer provides approximate theoretical information, as some of the layers may be ignored due to the model structure.
2. At the analysis stage, Model Analyzer can detect layers for which no information is available. Use `--ignore-unknown-layers` option to avoid errors. Feel free to submit your suggestions on how to analyze these layers. 

<details>
<summary>List of Ignored Layers</summary>

- Abs - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_arithmetic_Abs_1.html)
- BatchToSpace - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_movement_BatchToSpace_2.html)
- Broadcast - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_movement_Broadcast_3.html)
- Bucketize - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_condition_Bucketize_3.html)
- Convert - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_type_Convert_1.html)
- CtcGreedyDecoder - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_sequence_CTCGreedyDecoder_1.html)
- DetectionOutput - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_detection_DetectionOutput_1.html)
- Erf - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_arithmetic_Erf_1.html)
- ExperimentalDetectronDetectionOutput - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_detection_ExperimentalDetectronDetectionOutput_6.html)
- ExperimentalDetectronGenerateProposalsSingleImage - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_detection_ExperimentalDetectronGenerateProposalsSingleImage_6.html)
- ExperimentalDetectronPriorGridGenerator - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_detection_ExperimentalDetectronPriorGridGenerator_6.html)
- ExperimentalDetectronRoiFeatureExtractor - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_detection_ExperimentalDetectronROIFeatureExtractor_6.html)
- ExperimentalDetectronTopkRois - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_sort_ExperimentalDetectronTopKROIs_6.html)
- ExperimentalSparseWeightedSum 
- FakeQuantize - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_quantization_FakeQuantize_1.html)
- Flatten 
- GatherTree - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_movement_GatherTree_1.html)
- NonMaxSuppression - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_sort_NonMaxSuppression_5.html)
- PredictionHeatMap 
- ReSample 
- RegionYolo - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_detection_RegionYolo_1.html)
- ReorgYolo - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_detection_ReorgYolo_1.html)
- Slice - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_movement_StridedSlice_1.html)
- SpaceToBatch - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_movement_SpaceToBatch_2.html)
- SpatialTransformer 
- TensorIterator - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_infrastructure_TensorIterator_1.html)
- TopK - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_sort_TopK_3.html)
- Transpose - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_movement_Transpose_1.html)
- VariadicSplit - [opset7](https://docs.openvino.ai/latest/openvino_docs_ops_movement_VariadicSplit_1.html)
</details>
