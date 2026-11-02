import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "amlm_loss": AverageMeter(),
        "tc_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter(),
        "amlm_acc": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_mr = -1.0
    best_epoch = 0
    has_best = False
    patience = int(getattr(args, "patience", 15))
    arguments["epoch"] = 0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()
        module = model.module if args.distributed else model
        if getattr(module, 'use_tc', False):
            module.set_tc_epoch(epoch)

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)
            meters['amlm_loss'].update(ret.get('amlm_loss', 0), batch_size)
            meters['tc_loss'].update(ret.get('tc_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)
            meters['amlm_acc'].update(ret.get('amlm_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        use_tc = 'tc' in [name.strip() for name in args.loss_names.split('+')]
        stop_training = False
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                num_bins = args.tc_num_bins if use_tc else 0
                if args.distributed:
                    mean_mr = evaluator.eval(model.module.eval(), num_bins=num_bins)
                else:
                    mean_mr = evaluator.eval(model.eval(), num_bins=num_bins)

                torch.cuda.empty_cache()
                if mean_mr > best_mr:
                    best_mr = mean_mr
                    best_epoch = epoch
                    has_best = True
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
                elif has_best and epoch - best_epoch >= patience:
                    stop_training = True
            if args.distributed:
                flag = torch.zeros(1, device=device)
                if get_rank() == 0 and stop_training:
                    flag[0] = 1
                torch.distributed.broadcast(flag, src=0)
                stop_training = bool(flag.item())
        if use_tc and epoch % eval_period == 0:
            _update_tc_gamma(args, model, evaluator, logger)
        if stop_training:
            if get_rank() == 0:
                logger.info(
                    "Early stop at epoch {}: bidirectional mR did not improve for {} epochs.".format(
                        epoch, epoch - best_epoch))
            break
    if get_rank() == 0:
        logger.info("best bidirectional mR: {:.3f} at epoch {}".format(
            best_mr, arguments["epoch"]))


def _update_tc_gamma(args, model, evaluator, logger):
    if args.distributed:
        payload = torch.zeros(2, device="cuda")
        if get_rank() == 0:
            payload[0] = evaluator.last_i2t_gap
            payload[1] = evaluator.last_t2i_gap
        torch.distributed.broadcast(payload, src=0)
        i2t_gap = float(payload[0])
        t2i_gap = float(payload[1])
        module = model.module
    else:
        i2t_gap = evaluator.last_i2t_gap
        t2i_gap = evaluator.last_t2i_gap
        module = model
    module.tc_loss.updategamma(i2t_gap, t2i_gap)
    if get_rank() == 0:
        logger.info(
            "TC gamma i2t: {:.4f}, t2i: {:.4f} (gaps {:.4f}, {:.4f})".format(
                float(module.tc_loss.image_to_textgamma),
                float(module.tc_loss.text_to_imagegamma),
                i2t_gap, t2i_gap))


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
