from utils import collate_fn, compute_metrics
import argparse
import os
import pyarrow.parquet as pq
import pandas as pd


def _get_sentences(file_path):
    file_ext = os.path.splitext(file_path)[-1].lower()
    if file_ext == ".parquet":
        ref_df = pq.read_table(file_path, columns=["abstractive"])
        return ref_df["abstractive"].to_pylist()
    elif file_ext == ".json":
        ref_df = pd.read_json(file_path)
        return ref_df["summary"].tolist()

def main(args):
    assert args.ref_path is not None and args.pred_path is not None, "must specify both ref_path and pred_path"
    
    # TODO: assert equal ids

    pred_sents = _get_sentences(args.pred_path)  # json
    ref_sents = _get_sentences(args.ref_path)  # parquet

    print("="*30)
    print("Rouge Scores:\n", compute_metrics(pred_sents, ref_sents, args.apply_none))
    print("="*30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path', type=str, default="/opt/datasets/aihub_news_summ/Test/test.parquet")
    parser.add_argument('--pred_path', type=str)
    parser.add_argument('--apply_none', action='store_true')

    args = parser.parse_args()

    main(args)




