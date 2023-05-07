import argparse
from manager import NNManager


def parse_argument():
    """parse argument

    Returns:
        Namespace: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["CNN", "RNN", "baseline"],
        default="CNN",
        help="The model to be used.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate."
    )
    parser.add_argument("--epochs", type=float, default=10, help="Train epochs.")
    parser.add_argument("--batch_size", type=float, default=16, help="Batch size.")
    parser.add_argument(
        "--seq_length", type=float, default=120, help="Sequence length."
    )
    parser.add_argument(
        "--train", action="store_true", help="Whether to train the model."
    )
    parser.add_argument(
        "--save", action="store_true", help="Whether to save the model."
    )
    parser.add_argument(
        "--load_cnn_model_path",
        type=str,
        default="SaveModel/cnn.pth",
        help="Path of cnn model to be loaded.",
    )
    parser.add_argument(
        "--load_rnn_model_path",
        type=str,
        default="SaveModel/rnn.pth",
        help="Path of rnn model to be loaded.",
    )
    parser.add_argument(
        "--load_baseline_model_path",
        type=str,
        default="SaveModel/baseline.pth",
        help="Path of baseline model to be loaded.",
    )
    parser.add_argument(
        "--save_cnn_model_path",
        type=str,
        default="SaveModel/cnn.pth",
        help="Path to save cnn model.",
    )
    parser.add_argument(
        "--save_rnn_model_path",
        type=str,
        default="SaveModel/rnn.pth",
        help="Path to save rnn model.",
    )
    parser.add_argument(
        "--save_baseline_model_path",
        type=str,
        default="SaveModel/baseline.pth",
        help="Path to save baseline model.",
    )
    parser.add_argument(
        "--drop_out", type=float, default=0.6, help="Drop out probabilty."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight Decay."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("parsing arguments...")
    args = parse_argument()
    print("finish parsing arguments")

    print("initializing...")
    manager = NNManager(args=args)
    print("finish initializing")

    if manager.train:
        print("training start...")
        manager._train()
        print("training end...")
    if args.train and args.save:
        print("saving start...")
        manager._save()
        print("saving end")

    print("***** Validate Results *****")
    manager._print_results("validate")

    print("***** Test Results *****")
    manager._print_results("test")
