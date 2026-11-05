import argparse
import sys


def _explicit_dests(parser):
    explicit = set()
    for action in parser._actions:
        for opt in action.option_strings:
            for arg in sys.argv[1:]:
                if arg == opt or arg.startswith(opt + "="):
                    explicit.add(action.dest)
    return explicit


def get_args():
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--log_period", default=100)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--patience", default=15, type=int, help="stop after this many validations without a better bidirectional mR")
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=False, action='store_true')

    ## cross modal transfomer setting
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--MLM", default=False, action='store_true', help="whether to use Mask Language Modeling dataset")

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='sdm+id+mlm', help="which loss to use ['mlm', 'amlm', 'cmpm', 'id', 'itc', 'sdm', 'ot']")
    parser.add_argument("--sdm_loss_weight", type=float, default=1.0, help="SDM loss weight")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="implicit MLM loss weight")
    parser.add_argument("--amlm_loss_weight", type=float, default=0.5, help="explicit keyword MLM loss weight")
    parser.add_argument("--keyword_file", default="", help="keyword list for explicit MLM; empty uses the dataset vocab for RSICD/RSITMD")
    parser.add_argument("--amlm_keyword_mask_ratio", type=float, default=0.5, help="probability of masking each matched keyword span")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
    parser.add_argument("--ot_loss_weight", type=float, default=1.0, help="OT loss weight")
    parser.add_argument("--ot_hid_dim", type=int, default=512, help="hidden size of the OT matcher")
    parser.add_argument("--ot_num_dustbins", type=int, default=8, help="dustbin nodes on each side of the transport plan")
    parser.add_argument("--ot_sinkhorn_eps", type=float, default=0.1, help="entropic regularization of Sinkhorn")
    parser.add_argument("--ot_sinkhorn_iters", type=int, default=20, help="Sinkhorn iterations")
    parser.add_argument("--ot_k_candidates", type=int, default=50, help="global top-k kept for OT rescoring in each direction")
    parser.add_argument("--ot_eval_batch_size", type=int, default=128, help="pairs per Sinkhorn batch at evaluation")
    parser.add_argument("--ot_num_hard_negatives", type=int, default=3, help="hard negatives per direction in the OT loss")
    parser.add_argument("--ot_gate_init", type=float, default=0.1, help="initial sigmoid weight of the OT score")
    
    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=int, nargs=2, default=(256, 256), help="image height and width")
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="RSICD", help="[RSICD, RSITMD, Sydney, UCM]")
    parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    args = parser.parse_args()
    args.img_size = tuple(args.img_size)
    return args, _explicit_dests(parser)