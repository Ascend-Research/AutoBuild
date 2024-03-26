from onnx_ir.graph import INPUT_DUMMY_OP_TYPE, OUTPUT_DUMMY_OP_TYPE


OPS = ( # NOTE: only add new ops to the end, DO NOT modify existing order
    "<OOV>",
    INPUT_DUMMY_OP_TYPE,
    OUTPUT_DUMMY_OP_TYPE,
    "Conv",
    "ConvTranspose",
    "MatMul",
    "Gemm",
    "BatchNormalization",
    "InstanceNormalization",
    "LayerNormalization",
    "AveragePool",
    "MaxPool",
    "Add",
    "Concat",
    "Mul",
    "Sum",
    "Sub",
    "Neg",
    "Sqrt",
    "Pad",
    "Pow",
    "Relu",
    "Clip",
    "Sigmoid",
    "Tanh",
    "PRelu",
    "Constant",
    "DepthToSpace",
    "SpaceToDepth",
    "Dropout",
    "Reshape",
    "Resize",
    "Squeeze",
    "Unsqueeze",
    "Split",
    "Slice",
    "Identity",
    "Transpose",
    "HardSwish",
    "LogSoftmax",
    "TopK",
    "Tile",
    "Shape",
    "Softmax",
    "Elu",
    "Exp",
    "Expand",
    "Flatten",
    "Floor",
    "GRU",
    "LSTM",
    "RNN",
    "Gather",
    "GatherND",
    "Sign",
    "GlobalAveragePool",
    "GlobalLpPool",
    "GlobalMaxPool",
    "Greater",
    "Equal",
    "Less",
    "HardSigmoid",
    "Hardmax",
    "If",
    "Or",
    "Xor",
    "LeakyRelu",
    "Log",
    "LpNormalization",
    "Max",
    "Mean",
    "Mod",
    "NonMaxSuppression",
    "NonZero",
    "Reciprocal",
    "ReduceL1",
    "ReduceL2",
    "Round",
    "ScatterND",
    "Selu",
    "Celu",
    "Range",
    "GreaterOrEqual",
    "LessOrEqual",
    "SoftmaxCrossEntropyLoss",
    "CastLike",
    "Unique",
    "Where",
    "Div",
    "ReduceMean",
)


class OP2IDX:

    def __init__(self, ignore_case=True):
        super(OP2IDX, self).__init__()
        self.ignore_case = ignore_case
        self.oov_idx = 0
        self._op2idx = {}
        self._idx2op = {}
        self._build()

    def __str__(self):
        return f"OP2IDX[size={len(self)}]"

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._op2idx)

    def contains_op(self, op):
        if op.casefold() == OPS[self.oov_idx].casefold():
            return True
        return self._query_idx(op) != self.oov_idx

    def _build(self):
        self._op2idx = {}
        self._idx2op = {}
        op_labels = []

        for op in OPS:
            if self.ignore_case:
                op = op.casefold()
            assert op not in op_labels, f"Duplicate op: {op}"
            op_labels.append(op)
        self.op_labels = tuple(op_labels)

        for i, op in enumerate(self.op_labels):
            self._idx2op[i] = op
            self._op2idx[op] = i

        assert len(self._op2idx) == len(self._idx2op)
        return self

    def _query_idx(self, op):
        if self.ignore_case:
            op = op.casefold()
        if op in self._op2idx:
            return self._op2idx[op]
        return self.oov_idx

    def query_op(self, idx):
        return self._idx2op[idx]

    def __getitem__(self, op):
        rv = self._query_idx(op)
        return rv

    def __contains__(self, item):
        return self.contains_op(item)
