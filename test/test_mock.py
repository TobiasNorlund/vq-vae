import torch
from src.vae import Conv2d_output_shape, GetConv2dSequentialShape
import torch.nn as nn


def test_Conv2d_output_shape():

    i = {
        "N": 20,
        "C_in": 3,
        "C_out": 20,
        "H_in": 20,
        "W_in": 20,
        "kernel_size": 5,
        "dilation": 1,
        "stride": 1,
        "padding": 0,
    }

    vec = torch.randn(i["N"], i["C_in"], i["H_in"], i["W_in"])

    output_shape = tuple(
        nn.Conv2d(
            i["C_in"],
            i["C_out"],
            kernel_size=i["kernel_size"],
            stride=i["stride"],
            padding=i["padding"],
        )(vec).size()
    )
    assert output_shape == Conv2d_output_shape(**i)


def test_GetConv2dSequentialShape():
    li = [
        {
            "N": 20,
            "C_in": 3,
            "C_out": 30,
            "H_in": 20,
            "W_in": 20,
            "kernel_size": 5,
            "dilation": 1,
            "stride": 1,
            "padding": 0,
        },
        {
            "N": 20,
            "C_in": 30,
            "C_out": 20,
            "kernel_size": 5,
            "dilation": 1,
            "stride": 1,
            "padding": 0,
        },
        {
            "N": 20,
            "C_in": 20,
            "C_out": 10,
            "kernel_size": 5,
            "dilation": 1,
            "stride": 1,
            "padding": 0,
        },
    ]
    vec = torch.randn(20, 3, 20, 20)
    output_shapes = []
    output_shapes.append((li[0]["N"], li[0]["C_in"], li[0]["H_in"], li[0]["W_in"]))
    for i in li:
        vec = nn.Conv2d(
            i["C_in"],
            i["C_out"],
            kernel_size=i["kernel_size"],
            stride=i["stride"],
            padding=i["padding"],
        )(vec)
        output_shapes.append(tuple(vec.size()))

    to_try = GetConv2dSequentialShape((20, 3, 20, 20), li)
    assert output_shapes == to_try


def test_hello_world():
    s = "Hello world!"
    assert "Hello world!" == s
