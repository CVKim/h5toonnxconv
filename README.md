# TensorFlow to ONNX Converter for EfficientNetB2~7
# h5 to onnx

This repository contains a script that takes a pre-trained EfficientNetB2 model in TensorFlow and converts it into the ONNX (Open Neural Network Exchange) format. 

## Overview

The script primarily does the following:

1. Define a basic data augmentation sequence (currently commented out).
2. Build an EfficientNetB2 model, optionally adding a couple of extra dimensions to its output.
3. Load weights from a pre-trained `.h5` model.
4. Convert the TensorFlow model to ONNX format using `tf2onnx`.

## Prerequisites

Ensure you have the following libraries installed:

- `tensorflow`
- `tf2onnx`

You can install them using:

```bash
pip install tensorflow tf2onnx
