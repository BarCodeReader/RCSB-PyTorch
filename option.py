import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)

    # models
    parser.add_argument("--pretrain", type=str, default="")
    parser.add_argument("--model", type=str, default="RCSB")
    parser.add_argument("--GPU_ID", type=int, default=0)

    # dataset
    parser.add_argument("--dataset_root", type=str, default="dataset/")
    parser.add_argument("--dataset", type=str, default="DUTSTR")
    parser.add_argument("--test_dataset", type=str, default="benchmark_DUTSTE")

    # training setups
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay", type=str, default="20-40-60-80")
    parser.add_argument("--decay_step", type=int, default=20)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_features", type=int, default=64)
    parser.add_argument("--gclip", type=int, default=0)
    parser.add_argument("--R", type=int, default=3, help="recursion number")
    parser.add_argument("--G", type=int, default=1, help="config of G is written in model file, keep this G=1 here")
    
    # loss
    parser.add_argument("--lmbda", type=int, default=3, 
                        help="lambda in loss function, it is divided by 10 to make it float, so here use integer")

    # misc
    parser.add_argument("--test_only", action="store_true", help="test mode")
    parser.add_argument("--save_every_ckpt", action="store_true", help="save every ckpt")
    parser.add_argument("--save_result", action="store_true", help="save last stage's pred")
    parser.add_argument("--save_all", action="store_true", help="save all stages' pred")
    parser.add_argument("--ckpt_root", type=str, default="./ckpt")
    parser.add_argument("--save_root", type=str, default="./output")

    return parser.parse_args()

def get_option():
    opt = parse_args()
    return opt