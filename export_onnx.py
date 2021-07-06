import argparse
import os
import torch

device = torch.device('cuda')

def main():
    parser = argparse.ArgumentParser(
        description='Convert a PyTorch model to a Tensorflow model and export it as a TF graph')
    parser.add_argument('--file', '-f', help='path to the PyTorch .pth file')
    parser.add_argument(
        '--output', '-o', help='filename of the resulting ONNX exported file')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=1, help='batch size of the exported model')
    parser.add_argument('--height', type=int, default=224,
                        help='Height dimension of the input image')
    parser.add_argument('--width', type=int, default=224,
                        help='Width dimension of the input image')
    parser.add_argument('--channels', '-c', type=int, default=3,
                        help='Channels dimension of the input image. Defaults to 3 (for color images).')

    args = parser.parse_args()

    # Load torch model
    model = torch.load(args.file).to(device)

    # convert into ONNX file
    dummy_input = torch.randn(
        args.batch_size, args.channels, args.height, args.width, requires_grad=True).to(device)
    torch_out = model(dummy_input).to(device)
    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        # where to save the model (can be a file or file-like object)
        "tmp.onnx",
        export_params=True,        # store the trained parameter weights inside the model file
        input_names=['input'],   # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},    # variable length axes
            'output': {0: 'batch_size'}
        })


if __name__ == '__main__':
    main()
