import torch
from src.var import Conv2d_output_shape
import torch.nn as nn


def test_Conv2d_output_shape():
    N, C_in, H_in, W_in = 20, 3, 20, 20
    vec = torch.randn(i["N"], i["C_in"], i["H_in"], i["W_in"])
    i = {
        "N": 20,
        "C_in": 3,
        "H_in": 20,
        "W_in": 20,
        "kernel_size": 5,
        "dilation": 1,
        "stride": 1,
        "padding": 0,
    }

    output_shape = tuple(
        nn.Conv2d(
            i["C_in"],
            i["C_out"],
            kernel_size=i["kernel_size"],
            stride=i["stride"],
            padding=i["padding"],
        )().size(vec)
    )
    assert output_shape == Conv2d_output_shape(**i)


def test_GetConv2dSequentialShape():
    layer = nn.Sequential(nn.Conv2d())
    shapes = GetConv2dSequentialShape
    assert shapes[-1] == tuple(
        nn.Conv2d(
            i["C_in"],
            i["C_out"],
            kernel_size=i["kernel_size"],
            stride=i["stride"],
            padding=i["padding"],
        )(torch.randn(shapes[-2])).shape
    )


def test_hello_world():
    s = "Hello world!"
    assert "Hello world!" == s
