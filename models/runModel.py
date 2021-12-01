import argparse
from genericpath import exists
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from visualization import make_grid, plot_pred_gt, plot_ct
from CNN_ai_ct import CNN_AICT, LossTypes, NormTypes
from CNN_ai_ct_skip import CNN_AICT_SKIP
from CNN_ai_ct_trans_skip import CNN_AICT_TRANS_SKIP
from CNN_ai_ct_silu import CNN_AICT_SILU
from IRR_CNN_ai_ct import IRR_CNN_AICT
from Unet import Unet
from PerceiverModel import PerceiverIO
from dataloader import CtVolumeData, update_noisy_indexes, get_noisy_indexes
from utils import parse_dataset_paths


### runModel.py ###
# Script to perform custom model evaluation run from given checkpoints
# Loads the train set with the same seed and puts model in eval mode for inference


def switch_model(model_str):
    """ returns image indices based on model type
    Args:
        model_str: name of used model

    Returns:
        neighbour_img: image indices
    """
    neighbour_img = None
    if str(model_str).lower() == "cnn-ai-ct":
        # defines range of neighbour slices e.g. here -2 to +3 --> 5 slices
        neighbour_img = [-2, 3]
    elif str(model_str).lower() == "unet":
        # defines range of neighbour slices e.g. here 0 to 1 --> 1 slice
        neighbour_img = [0, 1]
    elif str(model_str).lower() == "perceiver-io":
        # defines range of neighbour slices e.g. here 0 to 1 --> 1 slice
        neighbour_img = [0, 1]
    return neighbour_img


def runModel(parser):
    """
        Run model evaluation on test set

    Args used:
        model: model name
        chkpt_path: model checkpoint path
        log_path: path to log test results in Tensorboard
        datasets: dataset to load for evaluation
        nr_test_samples: number of samples to test
        tensorboard_name: name of result tensorboard
        workers: number of workers for data loading
        forward_iterations: IRR-CNN-AI-CT specific forward iterations
        identical_batches: batches shall be only from one volume

    """
    args = parser.parse_args()
    data_paths = parse_dataset_paths(args.data_path, args.dataset_names)
    print("Datasets: ")
    for path in data_paths:
        print(path[2] +", ")
    model = args.model
    chkpt_path = args.check_point
    log_path = args.log_dir
    datasets = data_paths
    nr_test_samples = args.test_samples
    tensorboard_name = args.tensorboard_name
    num_workers = int(args.nr_workers) if args.nr_workers != None else None
    forward_iterations = args.forward_iterations
    identical_batches = args.identical_batches

    # Check noisy indice usage
    if args.remove_noisy_slices:
        noisy_indexes = get_noisy_indexes(
            args.file_noisy_indexes, data_paths)
    else:
        noisy_indexes = None

    # CT data loading
    ct_volumes = CtVolumeData(
        paths=datasets,
        batch_size=5,
        num_workers=num_workers,
        dataset_stride=128,
        num_pixel=256,
        test_split=0.1,
        val_split=0.2,
        noisy_indexes=noisy_indexes,
        manual_test=nr_test_samples, 
        neighbour_img=switch_model(str(model)),
        batch_identical=identical_batches
    )

    if str(args.model).lower() == "cnn-ai-ct":
        model = CNN_AICT.load_from_checkpoint(chkpt_path, ref_img=None, plot_test_step=args.plot_test_nr,
                                                  plot_val_step=args.plot_val_nr, plot_weights=args.plot_weights, custom_init=args.custom_init,
                                                  norm=args.norm, norm_modes=args.norm_modes, vol=ct_volumes, loss_function=args.loss_function,
                                                  linear_start=args.linear_start, norm_type=args.norm_function, strict=False)
        plugin = DDPPlugin(find_unused_parameters=True)
    elif str(args.model).lower() == "unet":
        model = Unet.load_from_checkpoint(chkpt_path, strict=False, ref_img=None, plot_test_step=args.plot_test_nr,
                     plot_val_step=args.plot_val_nr, plot_weights=args.plot_weights)
        plugin = DDPPlugin(find_unused_parameters=True)
    elif str(args.model).lower() == "irr-cnn-ai-ct":
        model = IRR_CNN_AICT.load_from_checkpoint(chkpt_path, strict=False, forward_iterations=int(args.forward_iterations), ref_img=None, plot_test_step=args.plot_test_nr,
                             plot_val_step=args.plot_val_nr, plot_weights=args.plot_weights)  # pass batch for visualization to CNN
        plugin = DDPPlugin(find_unused_parameters=True)
    elif str(args.model).lower() == "cnn-ai-ct-silu":
        model = CNN_AICT_SILU.load_from_checkpoint(chkpt_path, strict=False, ref_img=None, plot_test_step=args.plot_test_nr,
                              plot_val_step=args.plot_val_nr, plot_weights=args.plot_weights)
        plugin = DDPPlugin(find_unused_parameters=False)

    elif str(args.model).lower() == "cnn-ai-ct-skip":
        model = CNN_AICT_SKIP.load_from_checkpoint(chkpt_path, strict=False, ref_img=None, plot_test_step=args.plot_test_nr,
                              plot_val_step=args.plot_val_nr, plot_weights=args.plot_weights)
        plugin = DDPPlugin(find_unused_parameters=False)

    elif str(args.model).lower() == "cnn-ai-ct-trans-skip":
        model = CNN_AICT_TRANS_SKIP.load_from_checkpoint(chkpt_path, strict=False, ref_img=None, plot_test_step=args.plot_test_nr,
                                    plot_val_step=args.plot_val_nr, plot_weights=args.plot_weights)
        plugin = DDPPlugin(find_unused_parameters=False)

    elif str(args.model).lower() == "perceiver-io":
        model = PerceiverIO.load_from_checkpoint(
            chkpt_path,
            dim=256,                    # dimension of sequence to be encoded
            queries_dim=32,            # dimension of decoder queries
            logits_dim=256,            # dimension of final logits
            depth=6,                   # depth of net
            # number of latents, or induced set points, or centroids. different papers giving it different names
            num_latents=256,
            latent_dim=256,            # latent dimension
            cross_heads=1,             # number of heads for cross attention. paper said 1
            latent_heads=8,            # number of heads for latent self attention, 8
            cross_dim_head=64,         # number of dimensions per cross attention head
            latent_dim_head=64,        # number of dimensions per latent self attention head
            # whether to weight tie layers (optional, as indicated in the diagram)
            weight_tie_layers=False,
            strict=False
        )
        plugin = DDPPlugin(find_unused_parameters=True)

    model.to(args.device)

    model.eval()
    experiment_name = tensorboard_name+"-" + \
        chkpt_path.split("/")[-2]+"-"+chkpt_path.split("/")[-1]
    os.makedirs(os.path.join(log_path, experiment_name), exist_ok=True)
    tb_logger = TensorBoardLogger(
        log_path, default_hp_metric=False, name=experiment_name)

    # train model
    trainer = pl.Trainer.from_argparse_args(
        parser,
        logger=tb_logger,
        log_every_n_steps=10,
        plugins=plugin
    )
    # run model evaluation on test set
    trainer.test(model, datamodule=ct_volumes)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", "-dp", required=True, type=str,
                        help="Path to dataset json")
    parser.add_argument("--model", "-m", required=True, default="cnn-ai-ct",
                        help="model name [cnn-ai-ct, unet, irr-cnn-ai-ct, cnn-ai-ct-silu, cnn-ai-ct-skip, cnn-ai-ct-trans-skip, perceiver-io]")
    parser.add_argument("--batch-size", "-bs", required=False, default=5,
                        help="Batch size")
    parser.add_argument("--dataset-names", "-dn", required=False, nargs='+', default=["all"],
                        help="Names of the datasets of --file-in that should be used for training")
    parser.add_argument("--file-noisy-indexes", "-nf", required=False,
                        help="Path to the json file that contains the noisy indexes")
    parser.add_argument("--check-point", "-cp", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--nr_workers", "-w", required=False, default=10,
                        help="number of worker subproccesses to prefetch")
    parser.add_argument("--dir", "-d", required=False, default="",
                        help="directory where training artefacts are saved")
    parser.add_argument("--forward-iterations", "-fi", required=False, default=10,
                        help="Number of forward iterations: See IRR-Networks for details")
    parser.add_argument("--remove-noisy-slices", "-rn", required=False, default=None,
                        help="Parameter to activate/ deactive the removement of noisy slices")
    parser.add_argument("--plot-test-nr", "-pt", required=False, default=15,
                        help="number of images to plot from test set")
    parser.add_argument("--plot-weights", "-pw", required=False, action="store_true", default=False,
                        help="If argument is given (-pw) plot model weights")
    parser.add_argument("--plot-val-nr", "-pv", required=False, default=None,
                        help="number of images to plot from val set in each validation epoch")
    parser.add_argument("--tensorboard-name", "-tn", required=True, default="default", type=str,
                        help="name of datasets to load")
    parser.add_argument("--custom-init", "-ci", required=False, action="store_true", default=False,
                        help="If argument is given (-ci) custom init the model weights")
    parser.add_argument("--norm", "-no", required=False, action="store_true", default=False,
                        help="If argument is given (-no) use normalization in CNN-AI-CT")
    parser.add_argument("--norm-modes", "-nm", required=False, action="store_true", default=False,
                        help="If argument is given (-nm) use mode normalization from Json files in CNN-AI-CT")
    parser.add_argument("--norm-function", "-nfunc", type=NormTypes, choices=list(NormTypes), default=NormTypes.BatchNorm,
                        help="Norm function to use: batch, layer")
    parser.add_argument("--log-dir", "-ld", required=False, default="/net/pasnas01/pool1/enpro-2021-voxie/training/manual_runs",
                        help="directory where training artefacts are saved")
    parser.add_argument("--test-samples", "-ts", required=False, default=None, type=int,
                        help="test samples to subsample from testset for fast plotting & eval")
    parser.add_argument("--identical_batches", "-ib", required=False, action="store_true", default=False,
                        help="If argument is given (-ib) a batch only onsists of data from one dataset")
    parser.add_argument("--linear-start", "-ls", required=False, action="store_true", default=False,
                        help="If argument is given (-ls) use linear layer at the model input")
    parser.add_argument("--device", "-dv", required=False, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda/cpu)")
    parser.add_argument("--loss-function", "-lf", type=LossTypes, choices=list(LossTypes), default=LossTypes.MSE,
                        help="Loss function to use: mse, rmse")
    parser = pl.Trainer.add_argparse_args(parser)   
    runModel(parser)


if __name__ == "__main__":
    main()
