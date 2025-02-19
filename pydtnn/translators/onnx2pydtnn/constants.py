# ONNX operations:
from .operations.A import *
from .operations.B import *
from .operations.C import *
from .operations.D import *
from .operations.E import *
from .operations.F import *
from .operations.G import *
from .operations.H import *
from .operations.I import *
# There are operations having 'J' or 'K' as first character.
from .operations.L import *
from .operations.M import *
from .operations.N import *
from .operations.O import *
from .operations.P import *
from .operations.Q import *
from .operations.R import *
from .operations.S import *
from .operations.T import *
from .operations.U import *
# There are operations having 'V' as first character.
from .operations.W import *
from .operations.X import *

# ONNX AI operations:
from .operations.ai_onnx_ml import *
from .operations.ai_onnx_preview_training import *

CONST_NODE = "node"
CONST_OPSET = "opset_version"
CONST_OUPTUS = "outputs"
CONST_ATTRIBUTES = "attributes"
CONST_INPUTS = "inputs"
CONST_ALL_INPUTS = "all_inputs"
CONST_LISTS_NODES = "lists_nodes"
CONST_WEIGHTS = "weights"
CONST_PREV_LAYERS = "previous_layers"

# Operations to do:
# DenseNet169 - {'Conv', 'BatchNormalization', 'Unsqueeze', 'Add', 'Mul', 'Relu', 'MaxPool', 'AveragePool', 'GlobalAveragePool', 'Concat'}
# ResNet50 - {'Conv', 'MaxPool', 'Relu', 'Add', 'BatchNormalization', 'GlobalAveragePool', 'Gemm', 'Flatten'}
# VGG19 - {'Dropout', 'Gemm', 'Flatten', 'Relu', 'MaxPool', 'BatchNormalization', 'Conv'}
# Union of the ones before - {'Add', 'AveragePool', 'BatchNormalization', 'Concat', 'Conv', 'Dropout', 'Flatten', 'Gemm', 'GlobalAveragePool', 'MaxPool', 'Mul', 'Relu', 'Unsqueeze'}

def pads_from_onnx_to_pydttn(pads: List[int]) -> Tuple[int, int]: #-> List[Tuple[int, int]]:
        # "pads format should be as follow [x1_begin, x2_begin…x1_end, x2_end,…]" from, for example, https://onnx.ai/onnx/operators/onnx__AveragePool.html
        # Onnx: [x1_begin, x2_begin, ..., x1_end, x2_end, ...] ==> "PyDTNN: [(x1_begin, x1_end), (x2_end, x2_begin), ...]"
        # ==> PyDTNN only admits a int or a (vpadding, hpadding) ==> It's assumed that is the first tuple.

        print(f"pads: {pads}") # TODO: Borrar
        num_pads = len(pads)//2
        _pads = [(0,0)] * (num_pads)
        for i in range(num_pads):
            _pads[i] = (int(pads[i]), int(pads[i + num_pads]))
        print(f"_pads: {_pads}") # TODO: Borrar

        return _pads[0]
# --- END pads_from_onnx_to_pydttn --- #

SWITCH_OPERATION_ONNX_TO_PYDTNN = {
    "Abs" : Abs,
    "Acos" : Acos,
    "Acosh" : Acosh,
    "Add" : Add,
    "AffineGrid" : AffineGrid,
    "And" : And,
    "ArgMax" : ArgMax,
    "ArgMin" : ArgMin,
    "Asin" : Asin,
    "Asinh" : Asinh,
    "Atan" : Atan,
    "Atanh" : Atanh,
    "AveragePool" : AveragePool,
    "BatchNormalization" : BatchNormalization,
    "Bernoulli" : Bernoulli,
    "BitShift" : BitShift,
    "BitwiseAnd" : BitwiseAnd,
    "BitwiseNot" : BitwiseNot,
    "BitwiseOr" : BitwiseOr,
    "BitwiseXor" : BitwiseXor,
    "BlackmanWindow" : BlackmanWindow,
    "Cast" : Cast,
    "CastLike" : CastLike,
    "Ceil" : Ceil,
    "Celu" : Celu,
    "CenterCropPad" : CenterCropPad,
    "Clip" : Clip,
    "Col2Im" : Col2Im,
    "Compress" : Compress,
    "Concat" : Concat,
    "ConcatFromSequence" : ConcatFromSequence,
    "Constant" : Constant,
    "ConstantOfShape" : ConstantOfShape,
    "Conv" : Conv,
    "ConvInteger" : ConvInteger,
    "ConvTranspose" : ConvTranspose,
    "Cos" : Cos,
    "Cosh" : Cosh,
    "CumSum" : CumSum,
    "DFT" : DFT,
    "DeformConv" : DeformConv,
    "DepthToSpace" : DepthToSpace,
    "DequantizeLinear" : DequantizeLinear,
    "Det" : Det,
    "Div" : Div,
    "Dropout" : Dropout,
    "DynamicQuantizeLinear" : DynamicQuantizeLinear,
    "Einsum" : Einsum,
    "Elu" : Elu,
    "Equal" : Equal,
    "Erf" : Erf,
    "Exp" : Exp,
    "Expand" : Expand,
    "EyeLike" : EyeLike,
    "Flatten" : Flatten,
    "Floor" : Floor,
    "GRU" : GRU,
    "Gather" : Gather,
    "GatherElements" : GatherElements,
    "GatherND" : GatherND,
    "Gelu" : Gelu,
    "Gemm" : Gemm,
    "GlobalAveragePool" : GlobalAveragePool,
    "GlobalLpPool" : GlobalLpPool,
    "GlobalMaxPool" : GlobalMaxPool,
    "Greater" : Greater,
    "GreaterOrEqual" : GreaterOrEqual,
    "GridSample" : GridSample,
    "GroupNormalization" : GroupNormalization,
    "HammingWindow" : HammingWindow,
    "HannWindow" : HannWindow,
    "HardSigmoid" : HardSigmoid,
    "HardSwish" : HardSwish,
    "Hardmax" : Hardmax,
    "Identity" : Identity,
    "If" : If,
    "ImageDecoder" : ImageDecoder,
    "InstanceNormalization" : InstanceNormalization,
    "IsInf" : IsInf,
    "IsNaN" : IsNaN,
    "LRN" : LRN,
    "LSTM" : LSTM,
    "LayerNormalization" : LayerNormalization,
    "LeakyRelu" : LeakyRelu,
    "Less" : Less,
    "LessOrEqual" : LessOrEqual,
    "Log" : Log,
    "LogSoftmax" : LogSoftmax,
    "Loop" : Loop,
    "LpNormalization" : LpNormalization,
    "LpPool" : LpPool,
    "MatMul" : MatMul,
    "MatMulInteger" : MatMulInteger,
    "Max" : Max,
    "MaxPool" : MaxPool,
    "MaxRoiPool" : MaxRoiPool,
    "MaxUnpool" : MaxUnpool,
    "Mean" : Mean,
    "MeanVarianceNormalization" : MeanVarianceNormalization,
    "MelWeightMatrix" : MelWeightMatrix,
    "Min" : Min,
    "Mish" : Mish,
    "Mod" : Mod,
    "Mul" : Mul,
    "Multinomial" : Multinomial,
    "Neg" : Neg,
    "NegativeLogLikelihoodLoss" : NegativeLogLikelihoodLoss,
    "NonMaxSuppression" : NonMaxSuppression,
    "NonZero" : NonZero,
    "Not" : Not,
    "OneHot" : OneHot,
    "Optional" : Optional_layer,
    "OptionalGetElement" : OptionalGetElement,
    "OptionalHasElement" : OptionalHasElement,
    "Or" : Or,
    "PRelu" : PRelu,
    "Pad" : Pad,
    "Pow" : Pow,
    "QLinearConv" : QLinearConv,
    "QLinearMatMul" : QLinearMatMul,
    "QuantizeLinear" : QuantizeLinear,
    "RNN" : RNN,
    "RandomNormal" : RandomNormal,
    "RandomNormalLike" : RandomNormalLike,
    "RandomUniform" : RandomUniform,
    "RandomUniformLike" : RandomUniformLike,
    "Range" : Range,
    "Reciprocal" : Reciprocal,
    "ReduceL1" : ReduceL1,
    "ReduceL2" : ReduceL2,
    "ReduceLogSum" : ReduceLogSum,
    "ReduceLogSumExp" : ReduceLogSumExp,
    "ReduceMax" : ReduceMax,
    "ReduceMean" : ReduceMean,
    "ReduceMin" : ReduceMin,
    "ReduceProd" : ReduceProd,
    "ReduceSum" : ReduceSum,
    "ReduceSumSquare" : ReduceSumSquare,
    "RegexFullMatch" : RegexFullMatch,
    "Relu" : Relu,
    "Reshape" : Reshape,
    "Resize" : Resize,
    "ReverseSequence" : ReverseSequence,
    "RoiAlign" : RoiAlign,
    "Round" : Round,
    "STFT" : STFT,
    "Scan" : Scan,
    "Scatter" : Scatter,
    "ScatterElements" : ScatterElements,
    "ScatterND" : ScatterND,
    "Selu" : Selu,
    "SequenceAt" : SequenceAt,
    "SequenceConstruct" : SequenceConstruct,
    "SequenceEmpty" : SequenceEmpty,
    "SequenceErase" : SequenceErase,
    "SequenceInsert" : SequenceInsert,
    "SequenceLength" : SequenceLength,
    "SequenceMap" : SequenceMap,
    "Shape" : Shape,
    "Shrink" : Shrink,
    "Sigmoid" : Sigmoid,
    "Sign" : Sign,
    "Sin" : Sin,
    "Sinh" : Sinh,
    "Size" : Size,
    "Slice" : Slice,
    "Softmax" : Softmax,
    "SoftmaxCrossEntropyLoss" : SoftmaxCrossEntropyLoss,
    "Softplus" : Softplus,
    "Softsign" : Softsign,
    "SpaceToDepth" : SpaceToDepth,
    "Split" : Split,
    "SplitToSequence" : SplitToSequence,
    "Sqrt" : Sqrt,
    "Squeeze" : Squeeze,
    "StringConcat" : StringConcat,
    "StringNormalizer" : StringNormalizer,
    "StringSplit" : StringSplit,
    "Sub" : Sub,
    "Sum" : Sum,
    "Tan" : Tan,
    "Tanh" : Tanh,
    "TfIdfVectorizer" : TfIdfVectorizer,
    "ThresholdedRelu" : ThresholdedRelu,
    "Tile" : Tile,
    "TopK" : TopK,
    "Transpose" : Transpose,
    "Trilu" : Trilu,
    "Unique" : Unique,
    "Unsqueeze" : Unsqueeze,
    "Upsample" : Upsample,
    "Where" : Where,
    "Xor" : Xor,
    
    # ---- ai.onnx.ml ---- #

    "Binarizer" : Binarizer,
    "CastMap" : CastMap,
    "CategoryMapper" : CategoryMapper,
    "DictVectorizer" : DictVectorizer,
    "FeatureVectorizer" : FeatureVectorizer,
    "Imputer" : Imputer,
    "LabelEncoder" : LabelEncoder,
    "LinearClassifier" : LinearClassifier,
    "LinearRegressor" : LinearRegressor,
    "Normalizer" : Normalizer,
    "OneHotEncoder" : OneHotEncoder,
    "SVMClassifier" : SVMClassifier,
    "SVMRegressor" : SVMRegressor,
    "Scaler" : Scaler,
    "TreeEnsemble" : TreeEnsemble,
    "TreeEnsembleClassifier" : TreeEnsembleClassifier,
    "TreeEnsembleRegressor" : TreeEnsembleRegressor,
    "ZipMap" : ZipMap,

    # ---- ai.onnx.preview.training ---- #

    "Adagrad" : Adagrad,
    "Adam" : Adam,
    "Gradient" : Gradient,
    "Momentum" : Momentum,

}