import argparse
from datetime import date, timedelta

def add_train_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    
    parser.add_argument("--model_arch", default="EasyBartLinear", type=str, help="model architecture")
    parser.add_argument("--train_path", default="/opt/datasets/aihub_news_summ/Train/train.parquet", type=str, help="train dataset path")
    parser.add_argument("--eval_path", default="/opt/datasets/aihub_news_summ/Validation/valid.parquet", type=str, help="valid dataset path")
    parser.add_argument("--output_dir", default="./saved", type=str, help="path to save the trained model")
    parser.add_argument('--prediction_module', type=str, choices=["lpm", "rpm"])
    parser.add_argument('--pred_loss_function', type=str, help="prediction module loss function type")
    parser.add_argument('--freeze_backbone', action='store_true', help="freeze backbone layer")
    
    parser.add_argument("--per_device_train_batch_size", default=4, type=int, help="train batch size per device (default: 4)")
    parser.add_argument("--per_device_eval_batch_size", default=8, type=int, help="eval batch size per device (default: 8)")
    parser.add_argument("--num_train_epochs", default=1.0, type=float, help="num train epochs")
    parser.add_argument("--eval_steps", default=500, type=int, help="num train epochs")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="num gradient accumulation steps")

    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--seed", type=int, help="random seed number")

    parser.add_argument("--learning_rate", default=1e-05, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weigth decay in AdamW optimizer")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="beta1 in AdamW optimizer")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="beta2 in AdamW optimizer")
    parser.add_argument("--loss_alpha", default=0.5, type=float, help="extraction loss weight")
    parser.add_argument("--loss_beta", default=0.1, type=float, help="prediction module loss weight")

    return parser


def add_inference_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser():

    parser.add_argument('--model', type=str, default="./saved")
    parser.add_argument('--tokenizer', type=str, default="gogamza/kobart-summarization")
    parser.add_argument('--test_file_path', type=str, default="/opt/datasets/aihub_news_summ/Test/test.parquet")
    parser.add_argument('--save_json_name', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--compute_metrics', action='store_true')

    parser.add_argument("--no_cuda", action='store_true', help="run on cpu if True")
    parser.add_argument("--per_device_eval_batch_size", default=8, type=int, help="inference batch size per device (default: 8)")

    return parser


def add_predict_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser():
    
    parser.add_argument('--extractive', action='store_true')
    parser.add_argument('--generate_method', type=str, default="beam", choices=["greedy", "beam", "sampling"])
    parser.add_argument('--num_beams', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--min_length', type=int)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=int)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default = 3)

    return parser


def add_wandb_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser():

    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--wandb_run_name", default="run", type=str, help="wandb run name")
    parser.add_argument("--wandb_project", default="easybart", type=str, help="wandb project name")
    parser.add_argument("--wandb_entity", default="kkmjkim", type=str, help="wandb entity name")

    return parser