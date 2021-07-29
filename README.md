# model-analyzer

Model Analyzer is the Network Statistic Information tool

## Description

The tool is intended for theoretical computation of the following characteristics:

 - GFLOPs
 - MParams
 - Sparsity (number of zero weights divided by the total number of weights)
 - Minimum Memory Consumption
 - Maximum Memory Consumption

## Limitations

Works only with Intermediate Representation Models produced by OpenVINO Model Optimizer


##GFLOPs
Model analyzer support counting GFLOPs and GIOPs for these layers:
- Acosh
- Asinh
- Atanh
- Add
- ArgMax
- AvgPool
- BatchNormalization
- BinaryConvolution
- Clamp
- Concat
- Const
- Constant
- Convolution
- ConvolutionBackPropData
- Crop
- Deconvolution
- DeformableConvolution
- Divide
- Eltwise
- Elu
- Exp
- FullyConnected
- GEMM
- GRN
- Gather
- GatherND
- Greater
- GreaterEqual
- GroupConvolution
- GroupConvolutionBackpropData
- HSigmoid
- HSwish
- Input
- Interp
- Less
- LessEqual
- Log
- MVN
- MatMul
- MaxPool
- Mish
- Multiply
- Norm
- Normalize
- NormalizeL2
- OneHot
- Output
- PReLU
- PSROIPooling
- Pad
- Parameter
- Permute
- Pooling
- Power
- Priorbox
- PriorboxClustered
- Proposal
- ROIPooling
- Range
- ReLu
- ReduceL1
- ReduceL2
- ReduceMin
- Reshape
- Result
- ReverseSequence
- ScaleShift
- ScatterNDUpdate
- Select
- Sigmoid
- Softmax
- SoftPlus
- SparseToDense
- Split
- Squeeze
- StridedSlice
- Subtract
- Swish
- Tanh
- Tile
- Unsqueeze


These layers are recognized by Model Analyzer but counted as zero(realization in progress or there is no method how to count them):
- Abs
- BatchToSpace
- Broadcast
- Bucketize
- Convert
- CtcGreedyDecoder
- DetectionOutput
- Erf
- ExperimentalDetectronDetectionOutput
- ExperimentalDetectronGenerateProposalsSingleImage
- ExperimentalDetectronPriorGridGenerator
- ExperimentalDetectronRoiFeatureExtractor
- ExperimentalDetectronTopkRois
- ExperimentalSparseWeightedSum
- FakeQuantize
- Flatten
- GatherTree
- NonMaxSuppression
- PredictionHeatMap
- ReSample
- RegionYolo
- ReorgYolo
- Slice
- SpaceToBatch
- SpatialTransformer
- TensorIterator
- TopK
- Transpose
- VariadicSplit