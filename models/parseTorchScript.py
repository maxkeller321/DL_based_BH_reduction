import argparse
import os
import torch

from CNN_ai_ct_old import CNN_AICT
from IRR_CNN_ai_ct import IRR_CNN_AICT
from Unet import Unet


### parseTorchScript.py ###
# Script to generate JIT traces for the usage with c++ libtorch
# Loads given model from checkpoint then generates its computational
# trace with a dummy input https://pytorch.org/tutorials/advanced/cpp_frontend.html
# our newest CNN-AI-CT is not compatible for JIT tracing due to the if/else statements we added in forward/training
# to get traces for old models use the CNN-AI-CT_old

def runJitTrace(model, name, input_data):
    """ Generate the actual jit trace and save the result, 
        Additionally also save the torchscript file .pt

    Args:
        model: pytorch model
        name: model name
        input_data: dummy data for graph inference

    """
    torch.jit.save(model.to_torchscript(), name+".pt")
    if os.path.isfile(name+".pt"):
        torch.jit.save(model.to_torchscript(file_path=name+"_trace.pt", method='trace',
                                            example_inputs=input_data), name+"_trace.pt")
        return os.path.isfile(name+"_trace.pt")


def loadJitTrace(path):
    """ Loads model from jit trace

    Args:
        path: jit trace file path

    """
    model = torch.jit.load(path)
    print(model)
    print(model.code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-point", "-cp", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--trace-name", "-tn", required=False, default=None,
                        help="JIT trace naming")
    parser.add_argument("--model-name", "-m", required=True, default="cnn-ai-ct",
                        help="model name [cnn-ai-ct, unet, irr-cnn-ai-ct, cnn-ai-ct-silu]")
    parser.add_argument("--forward-iterations", "-fi", required=False, default=10,
                        help="Number of forward iterations: See IRR-Networks for details")
    parser.add_argument("--load-mode", "-lm", required=False, action="store_true", default=False,
                        help="If argument is given (-lm) load jit trace instead of writing it")
    args = parser.parse_args()

    if not args.load_mode:
        if str(args.model_name).lower() == "cnn-ai-ct":
            model = CNN_AICT.load_from_checkpoint(args.check_point, strict=False)
            jit_data = torch.randn(1, 5, 1024, 1024)
        elif str(args.model_name).lower() == "unet":
            model = Unet.load_from_checkpoint(args.check_point)
            jit_data = torch.randn(1, 1, 1024, 1024)
        elif str(args.model_name).lower() == "irr-cnn-ai-ct":
            jit_data = torch.randn(1, 5, 1024, 1024)
            model = IRR_CNN_AICT(
                args.check_point, forward_iterations=args.forward_iterations)
        else:
            return

        new_name = args.trace_name if args.trace_name is not None else args.model_name

        res = runJitTrace(model, new_name, jit_data)
        if res:
            print("Jit trace successful. Created files {} and {}".format(
                new_name+".pt", new_name+"_trace.pt"))
        else:
            print("Jit trace failed")
    else:
        loadJitTrace(args.check_point)


if __name__ == "__main__":
    main()
