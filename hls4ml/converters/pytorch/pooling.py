import math
# from hls4ml.converters.pytorch_to_hls import parse_default_pytorch_layer
from hls4ml.converters.pytorch_to_hls import pytorch_handler
from hls4ml.converters.utils import compute_padding_1d, compute_padding_2d, parse_data_format

pooling_layers = ['MaxPool1d', 'MaxPool2d', 'AvgPool1d', 'AvgPool2d']


class_map = {
    'MaxPool1d': 'MaxPooling1D',
    'MaxPool2d': 'MaxPooling2D',
    'AvgPool1d': 'AveragePooling1D',
    'AvgPool2d': 'AveragePooling2D',
}

@pytorch_handler(*pooling_layers)
def parse_pooling_layer(pytorch_layer, layer_name, input_shapes, data_reader, config):
    assert('Pool' in pytorch_layer.__class__.__name__)
    
    layer = {}

    layer['name'] = layer_name
    layer['data_format'] = 'channels_first' #Pytorch default (can't change)
    layer['class_name'] = class_map[pytorch_layer.__class__.__name__]

    if int(layer['class_name'][-2]) == 1:
        (layer['n_in'], layer['n_filt']) = parse_data_format(input_shapes[0], layer['data_format'])

        layer['pool_width'] = pytorch_layer.kernel_size
        layer['stride_width'] = pytorch_layer.stride
        layer['padding'] = pytorch_layer.padding

        (layer['n_out'], layer['pad_left'], layer['pad_right']) = compute_padding_1d(
            layer['padding'], layer['n_in'], layer['stride_width'], layer['pool_width']
        )

        output_shape = [input_shapes[0][0], layer['n_filt'], layer['n_out']]
    elif int(layer['class_name'][-2]) == 2:
        (layer['in_height'], layer['in_width'], layer['n_filt']) = parse_data_format(input_shapes[0], layer['data_format'])

        if type(pytorch_layer.stride) is tuple:
            layer['stride_height'] = pytorch_layer.stride[0]
            layer['stride_width'] = pytorch_layer.stride[1]
        else: 
            layer['stride_height'] = pytorch_layer.stride
            layer['stride_width'] = pytorch_layer.stride
        if type(pytorch_layer.kernel_size) is tuple:
            layer['pool_height'] = pytorch_layer.kernel_size[0]
            layer['pool_width'] = pytorch_layer.kernel_size[1]
        else:
            layer['pool_height'] = pytorch_layer.kernel_size
            layer['pool_width'] = pytorch_layer.kernel_size
        if type(pytorch_layer.padding) is tuple:
            layer['pad_top'] = pytorch_layer.padding[0]
            layer['pad_bottom'] = pytorch_layer.padding[0]
            layer['pad_left'] = pytorch_layer.padding[1]
            layer['pad_right'] = pytorch_layer.padding[1]
        else:
            layer['pad_top'] = pytorch_layer.padding
            layer['pad_bottom'] = pytorch_layer.padding
            layer['pad_left'] = pytorch_layer.padding
            layer['pad_right'] = pytorch_layer.padding
        if type(pytorch_layer.dilation) is tuple:
            layer['dilation_height'] = pytorch_layer.dilation[0]
            layer['dilation_width'] = pytorch_layer.dilation[0]
        else:
            layer['dilation_height'] = pytorch_layer.dilation
            layer['dilation_width'] = pytorch_layer.dilation        
        if pytorch_layer.ceil_mode is True:
            round_op = math.ceil
        else:
            round_op = math.floor

        layer['out_height'] = round_op(((layer['in_height'] + 2*layer['pad_top'] - layer['dilation_height'] * (layer['pool_height']-1) - 1)/layer['stride_height'])+1)
        layer['out_width'] = round_op(((layer['in_width'] + 2*layer['pad_left'] - layer['dilation_width'] * (layer['pool_width']-1) - 1)/layer['stride_width'])+1)

        output_shape = [layer['n_filt'], layer['out_height'], layer['out_width']]
        # output_shape = [input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]

    return layer, output_shape