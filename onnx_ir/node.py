import onnx
import copy
import collections
from abc import ABC
import onnx.numpy_helper as numpy_helper
from google.protobuf.json_format import ParseDict
import struct

"""
In-memory node IR of an ONNX model
Designed for easy modification in python and fast ONNX I/O 
"""


class Node(ABC):

    def __init__(self, str_id, name, op_type, inputs, outputs,
                 attribute=None, initializer=None):
        self.str_id = str_id
        self.name = name
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attribute = attribute
        self.initializer = initializer
 
    def state_dict(self):
        return vars(self)

    @property
    def input_shapes(self):
        """
        Will try to get the shape info from input info dicts
        If somehow no shape info is available, will return None for that input
        Users then must do None checks
        Shape might be in any order depending on the graph itself (BCHW or BHWC)
        """
        shapes = []
        for d in self.inputs:
            try:
                inner_dict =  d["type"]["tensorType"]["shape"]
                if "dim" not in inner_dict:
                    # Meaning the input has no dim, i.e., scalar
                    inst_shape = [0]
                else:
                    inst_shape = [
                        int(dim_dict["dimValue"])
                        for dim_dict in inner_dict["dim"]
                    ]
            except:
                inst_shape = None
            shapes.append(inst_shape)
        return shapes

    @property
    def output_shapes(self):
        """
        Will try to get the shape info from output info dicts
        If somehow no shape info is available, will return None for that output
        Users then must do None checks
        Shape might be in any order depending on the graph itself (BCHW or BHWC)
        """
        shapes = []
        for d in self.outputs:
            try:
                inner_dict =  d["type"]["tensorType"]["shape"]
                if "dim" not in inner_dict:
                    # Meaning the output has no dim, i.e., scalar
                    inst_shape = [0]
                else:
                    inst_shape = [
                        int(dim_dict["dimValue"])
                        for dim_dict in d["type"]["tensorType"]["shape"]["dim"]
                    ]
            except:
                inst_shape = None
            shapes.append(inst_shape)
        return shapes

    def get_input_shape(self, name):
        for d in self.inputs:
            if d["name"] == name:
                try:
                    inner_dict =  d["type"]["tensorType"]["shape"]
                    if "dim" not in inner_dict:
                        # Meaning the input has no dim, i.e., scalar
                        inst_shape = [0]
                    else:
                        inst_shape = [
                            int(dim_dict["dimValue"])
                            for dim_dict in inner_dict["dim"]
                        ]
                except:
                    inst_shape = None
                return inst_shape
        return None

    def get_output_shape(self, name):
        for d in self.outputs:
            if d["name"] == name:
                try:
                    inner_dict =  d["type"]["tensorType"]["shape"]
                    if "dim" not in inner_dict:
                        # Meaning the output has no dim, i.e., scalar
                        inst_shape = [0]
                    else:
                        inst_shape = [
                            int(dim_dict["dimValue"])
                            for dim_dict in d["type"]["tensorType"]["shape"]["dim"]
                        ]
                except:
                    inst_shape = None
                return inst_shape
        return None

    def find_attribute(self, name):
        """
        find a particular attribute dict in attribute list
        """
        if self.attribute is None:
            return None
        if isinstance(self.attribute, dict):
            attribute = [self.attribute]
        else:
            assert isinstance(self.attribute, list)
            attribute = self.attribute
        for d in attribute:
            if d["name"] == name: # TODO: enable non-exact matching?
                return d
        return None

    def find_str_val(self, attr_name):
        """
        find a particular attribute dict in attribute list
        """
        attr_dict = self.find_attribute(attr_name)
        assert attr_dict is not None
        return attr_to_readable_str(attr_dict)

    @property
    def kernel_size(self):
        """
        If the node is Conv, can fetch kernel size, otherwise None is returned
        """
        try:
            kernel_size = [int(v) for v in self.find_attribute("kernel_shape")["ints"]]
        except:
            kernel_size = None
        return kernel_size

    @property
    def initializer_dims(self):
        """
        Will try to get the initializer dims
        Return values sorted by the number of values, so weight channel should go before bias
        There is a special case for Conv and Matmul nodes
        If the node has more than 1 inputs and initializer is None, inputs after first will be treated as weights
        Return None if not available
        For Conv, C_out is the first dim, C_in is the second
        """
        try:
            rv = []
            if self.initializer is None and \
                self.op_type.casefold() in {"conv", "matmul", "gemm"} and \
                len(self.inputs) > 1:
                for d in self.inputs[1:]:
                    vals = [
                        int(dim_dict["dimValue"])
                        for dim_dict in d["type"]["tensorType"]["shape"]["dim"]
                    ]
                    rv.append(vals)
            else:
                for k, d in self.initializer.items():
                    vals = [int(v) for v in d["dims"]]
                    rv.append(vals)
            rv.sort(key=lambda t: len(t), reverse=True)
            return rv
        except:
            return None

    @property
    def initializer_names(self):
        try:
            rv = []
            if self.initializer is None and \
                self.op_type.casefold() in {"conv", "matmul", "gemm"} and \
                len(self.inputs) > 1:
                for d in self.inputs[1:]:
                    vals = [
                        int(dim_dict["dimValue"])
                        for dim_dict in d["type"]["tensorType"]["shape"]["dim"]
                    ]
                    rv.append( (d["name"], vals) )
            else:
                for k, d in self.initializer.items():
                    vals = [int(v) for v in d["dims"]]
                    rv.append( (d["name"], vals) )
            rv.sort(key=lambda t: len(t), reverse=True)
            return [t[0] for t in rv]
        except:
            return None

    @property
    def initializer_data(self):
        """
        Will try to get all the initializer data of the node as np array
        Return None if not available
        """
        try:
            # Naively make sure that weight appear before bias
            sorted_keys = [k for k, d in self.initializer.items()]
            sorted_keys.sort(key=lambda k:len([int(v) for v in self.initializer[k]["dims"]]), reverse=True)
            rv = collections.OrderedDict()
            for k in sorted_keys:
                d = self.initializer[k]
                msg = ParseDict(d, onnx.TensorProto())
                data = numpy_helper.to_array(msg)
                rv[d["name"]] = data
            return rv
        except:
            return None

    @property
    def strides(self):
        try:
            strides = [int(v) for v in self.find_attribute("strides")["ints"]]
        except:
            strides = None
        return strides

    @property
    def paddings(self):
        try:
            paddings = [int(v) for v in self.find_attribute("pads")["ints"]]
        except:
            paddings = None
        return paddings

    @property
    def dilations(self):
        try:
            dilations = [int(v) for v in self.find_attribute("dilations")["ints"]]
        except:
            dilations = None
        return dilations

    @property
    def group(self):
        try:
            group = int(self.find_attribute("group")["i"])
        except:
            group = None
        return group

    @property
    def has_bias(self):
        if self.initializer is not None:
            return any(s.endswith("bias") for s in self.initializer.keys())
        return None

    @property
    def axis(self):
        """
        Applicable to nodes like Concat or Split
        """
        try:
            axis = int(self.find_attribute("axis")["i"])
        except:
            axis = None
        return axis

    @property
    def clip_vals(self):
        """
        Applicable clip nodes
        """
        try:
            min_val = int(self.find_attribute("min")["f"])
            max_val = int(self.find_attribute("max")["f"])
            rv = (min_val, max_val)
        except:
            rv = None
        return rv
    
    @property
    def perm(self):
        """
        Applicable to nodes like Transpose
        """
        try:
            perm = [int(v) for v in self.find_attribute("perm")["ints"]]
        except:
            perm = None
        return perm

    @property
    def value(self):
        """
        Applicable to Constant nodes that act as inputs to other nodes like Mul nodes representing scalar multiplication
        """
        if 'constant' in self.name.lower():
            try:
                value = self.find_attribute('value')
                attr_proto = ParseDict(value, onnx.AttributeProto())
                if attr_proto.t.data_type == onnx.TensorProto.FLOAT or attr_proto.t.data_type == onnx.TensorProto.FLOAT16:
                    x = struct.unpack('f', attr_proto.t.raw_data)
                    value = x[0]
                elif attr_proto.t.data_type == onnx.TensorProto.INT8 or attr_proto.t.data_type == onnx.TensorProto.INT16 or attr_proto.t.data_type == onnx.TensorProto.INT32 or attr_proto.t.data_type == onnx.TensorProto.INT64:
                    x = struct.unpack('i', attr_proto.t.raw_data)
                    value = x[0]
            except:
                value = None
        else:
            value = None
        return value

    def __str__(self):
        label = self.op_type + f"\n"
        label += f"name: {self.name}\n"
        if self.kernel_size is not None:
            label += f"kernel_size: {self.kernel_size}\n"
        if self.input_shapes is not None:
            label += f"input_shapes: {self.input_shapes}\n"
        if self.output_shapes is not None:
            label += f"output_shapes: {self.output_shapes}\n"
        if self.initializer_dims is not None:
            label += f"initializer_dims: {self.initializer_dims}\n"
        if self.strides is not None:
            label += f"strides: {self.strides}\n"
        if self.paddings is not None:
            label += f"paddings: {self.paddings}\n"
        if self.dilations is not None:
            label += f"dilations: {self.dilations}\n"
        if self.group is not None:
            label += f"group: {self.group}\n"
        if self.axis is not None:
            label += f"axis: {self.axis}\n"
        if self.clip_vals is not None:
            label += f"clip_vals: {self.clip_vals}\n"
        if self.has_bias is not None:
            label += f"has_bias: {self.has_bias}\n"
        return label.rstrip()

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memodict={}):
        node = Node(self.str_id,
                    self.name,
                    self.op_type,
                    copy.deepcopy(self.inputs),
                    copy.deepcopy(self.outputs))
        if self.attribute is not None:
            node.attribute = copy.deepcopy(self.attribute)
        if self.initializer is not None:
            node.initializer = copy.deepcopy(self.initializer)
        return node


def attr_to_readable_str(attr_dict):
    attr_proto = ParseDict(attr_dict, onnx.AttributeProto())
    val = attr_proto.s.decode("utf-8", errors="ignore")
    return val