# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
This file is based on the YOLOv5 script by Ultralytics, which is released under
the AGPL-3.0 license. Modifications and additions have been made by Deeplite Inc.
"""

import argparse
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from deeplite_torch_zoo import (create_model, get_data_splits_by_name,
                                get_eval_function, get_model_by_name)
from torch.cuda import amp
from tqdm import tqdm

from utils.kd import KDTeacher
from utils.general import (LOGGER, WorkingDirectory, colorstr, increment_path,
                           init_seeds, print_args, yaml_save)
from utils.torch_utils import (GenericLogger, ModelEMA, select_device,
                               smart_DDP, smart_optimizer,
                               smartCrossEntropyLoss,
                               torch_distributed_zero_first,
                               get_layers_bytype,
                               get_output_shapes,
                               skip_layers)
from prune import APruner
from jinja2 import Environment, FileSystemLoader

ROOT = Path.cwd()
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def train(opt, device):
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir, bs, epochs, nw, imgsz, pretrained = \
        opt.save_dir, opt.batch_size, opt.epochs, min(os.cpu_count() - 1, opt.workers), \
        opt.imgsz, str(opt.pretrained).lower() == 'true'
    cuda = device.type != 'cpu'

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best, best_sd = wdir / 'last.pt', wdir / 'best.pt', wdir / 'best_state_dict.pt'

    # Save run settings
    yaml_save(save_dir / 'opt.yaml', vars(opt))
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0} else None

    # Dataloaders
    dataloaders = get_data_splits_by_name(
        data_root=opt.data_root,
        dataset_name=opt.dataset,
        model_name=opt.model,
        batch_size=bs,
        test_batch_size=opt.test_batch_size,
        img_size=imgsz,
        num_workers=nw,
    )
    trainloader, testloader = dataloaders['train'], dataloaders['test']

    # Model
    opt.num_classes = len(trainloader.dataset.classes)
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        if opt.model_pth is not None:
            model = torch.load(opt.model_pth, map_location=device)
            for n, p in model.named_parameters():
                if p.ndim > 1:
                    LOGGER.info(n, p.shape)
        else:
            if opt.dataset == 'cifar100':
                model = get_model_by_name(
                        model_name=opt.model,
                        dataset_name='cifar100',
                        pretrained=pretrained,
                )
            else:
                model = create_model(
                    model_name=opt.model,
                    pretraining_dataset=opt.pretraining_dataset,
                    num_classes=opt.num_classes,
                    pretrained=pretrained,
                )

        model_kd = None
        if opt.kd_model_name is not None:
            model_kd = KDTeacher(opt)

    for p in model.parameters():
        p.requires_grad = True  # for training
    model = model.to(device)

    # Pruner Setup
    modules_toprune = get_layers_bytype(model)
    num_steps = len(trainloader) * epochs
    begin_step = int(opt.prune_begin * num_steps)
    end_step = int(opt.prune_end * num_steps)
    if opt.layers2skip is not None:
        modules_toprune, modules_toskip = skip_layers(modules_toprune, opt.layers2skip.copy())
    LOGGER.info("-" * 100)
    LOGGER.info("  >> Modules to prune:")
    LOGGER.info(list(modules_toprune.values()))
    LOGGER.info("\n  >> Modules to skip:")
    LOGGER.info(list(modules_toskip.values()))
    LOGGER.info("-" * 100)
    if opt.prune_type == 'activation':
        input_shape = [opt.batch_size, opt.in_channels, opt.imgsz, opt.imgsz]
        output_shapes = get_output_shapes(
                model, input_shape, modules_toprune, device
        )
        prune_configs = {
                'begin_step': begin_step,
                'end_step': end_step,
                'freeze_step': end_step,
                'frequency': opt.prune_freq,
                'init_value': opt.prune_init_value,
                'final_value': opt.prune_final_value,
                'criterion': opt.prune_criterion,
                'dim': opt.prune_dim,
                'shapes': output_shapes,
                'imgsz': opt.imgsz,
        }
        pruner = APruner(modules_toprune, prune_configs, device)
    else:
        raise NotImplementedError

    # Eval function
    evaluation_fn = get_eval_function(
        model_name=opt.model,
        dataset_name=opt.pretraining_dataset,
    )

    # Info
    if RANK in {-1, 0}:
        if opt.verbose:
            LOGGER.info(model)

    # Optimizer
    optimizer = smart_optimizer(model, opt.optimizer, opt.lr0, momentum=0.9, decay=opt.decay)

    # Scheduler
    lrf = 0.01  # final lr (fraction of lr0)
    # lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    lf = lambda x: (1 - x / epochs) * (1 - lrf) + lrf  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr0, total_steps=epochs, pct_start=0.1,
    #                                    final_div_factor=1 / 25 / lrf)

    # EMA
    # ema = ModelEMA(model) if RANK in {-1, 0} else None
    ema = None

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Train
    step = 0 # train step
    t0 = time.time()
    criterion = smartCrossEntropyLoss(label_smoothing=opt.label_smoothing)  # loss function
    best_fitness = 0.0
    scaler = amp.GradScaler(enabled=cuda)
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} test\n'
                f'Using {nw * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting {opt.model} training on {opt.dataset} dataset for {epochs} epochs...\n'
                f'  >> Pruning every {opt.prune_freq} steps in the range [{begin_step}, {num_steps}] (freezing at step {end_step})\n\n'
                f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{'top1_acc':>12}{'top5_acc':>12}")
    l0_score = None
    best_loss = float("inf")
    for epoch in range(epochs):  # loop over the dataset multiple times
        tloss, vloss, fitness = 0.0, 0.0, 0.0  # train loss, val loss, fitness
        model.train()
        pbar = enumerate(trainloader)
        if RANK in {-1, 0}:
            pbar = tqdm(enumerate(trainloader), total=len(trainloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (images, labels) in pbar:  # progress bar
            images, labels = images.to(device, non_blocking=True), labels.to(device)

            # Forward
            is_prune_step = pruner.step(step)

            with amp.autocast(enabled=cuda):  # stability issues when enabled
                output = model(images)
                loss = criterion(output, labels)

                if model_kd is not None:
                    prob_s = F.log_softmax(output, dim=-1)
                    with torch.no_grad():
                        input_kd = model_kd.normalize_input(images, model)
                        out_t = model_kd.model(input_kd.detach())
                        prob_t = F.softmax(out_t, dim=-1)
                    if not opt.use_kd_only_loss:
                        loss += opt.alpha_kd * F.kl_div(prob_s, prob_t, reduction='batchmean')
                    else:
                        loss = opt.alpha_kd * F.kl_div(prob_s, prob_t, reduction='batchmean')

            if step >= begin_step and step < end_step:
                if loss < best_loss:
                    best_loss = loss
                    pruner.best_l0_score = pruner.l0_score

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Disable for sparse training
            if ema is not None:
                ema.update(model)
                model_ = ema.ema
                updates_ = ema.updates
            else:
                model_ = model
                updates_ = None

            if RANK in {-1, 0}:
                # Print
                tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                pbar.desc = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + ' ' * 36

                # Test
                if i == len(pbar) - 1:
                    if step >= pruner.scheduler.begin_step and pruner.sparsity == 0:
                        pruner.hard_step(step) # force pruning on last batch
                    metrics = evaluation_fn(model_, testloader, progressbar=False)
                    #metrics = evaluation_fn(ema.ema, testloader, progressbar=False)
                    top1, top5 = metrics['acc'], metrics['acc_top5']
                    fitness = top1  # define fitness as top1 accuracy
                    pbar.desc = f"{pbar.desc[:-36]}{top1:>12.3g}{top5:>12.3g}"

            step += 1

        # Scheduler
        scheduler.step()

        # Log metrics
        if RANK in {-1, 0}:
            # Best fitness
            if fitness > best_fitness:
                best_fitness = fitness

            # Log
            metrics = {
                "train/loss": tloss,
                "metrics/accuracy_top1": top1,
                "metrics/accuracy_top5": top5,
                "metrics/sparsity": pruner.sparsity,
                "lr/0": optimizer.param_groups[0]['lr']}  # learning rate
            logger.log_metrics(metrics, epoch)

            # Save model
            final_epoch = epoch + 1 == epochs
            if (not opt.nosave) or final_epoch:
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    # 'model': deepcopy(ema.ema).half(),
                    'model': deepcopy(model_).half(),
                    'ema': None,  # deepcopy(ema.ema).half(),
                    'updates': updates_,
                    # 'updates': ema.updates,
                    'optimizer': None,  # optimizer.state_dict(),
                    'opt': vars(opt),
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt['model'].state_dict(), last)
                if best_fitness == fitness:
                    # torch.save(ckpt, best)
                    torch.save(ckpt['model'].state_dict(), best_sd)
                del ckpt


    # Get binary masks
    if opt.save_masks:
        LOGGER.info('\n  {:25s} Sparsity (%)'.format('Module Name'))
        LOGGER.info('-' * 60)
        arrays = []
        for name, module in modules_toprune.items():
            mask = module.mask[0, :, :]
            _, _, oh, ow = output_shapes[name]
            array = {
                'name': name.replace('.', '_').upper(),
                'value': mask.int().tolist(),
                'rows': oh,
                'cols': ow,
            }
            arrays.append(array)
            LOGGER.info('  {:25s} {:.4f}'.format(
                name, (1 - torch.sum(mask).item() / mask.numel()) * 100)
            )
        code_dict = {'arrays': arrays}
        env = Environment(
            loader=FileSystemLoader(''), trim_blocks=True,
            lstrip_blocks=True
        )
        template = env.get_template('templates/mask_template.j2')
        filename = '{}/bin_mask.h'.format(save_dir)
        code = template.render(**code_dict)
        with open(filename, "w") as fp:
            print(code, file=fp)
            fp.close()
        LOGGER.info(
            f"\nBinary masks for inference saved as {filename.split('/')[-1]}"
        )

    # Train complete
    if RANK in {-1, 0} and final_epoch:
        LOGGER.info(f'\nTraining complete ({(time.time() - t0) / 3600:.3f} hours)'
                    f"\nResults saved to {colorstr('bold', save_dir)}")

        # Log results
        meta = {"epochs": epochs, "top1_acc": best_fitness, "date": datetime.now().isoformat()}
        logger.log_model(best, epochs, metadata=meta)

    return model


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./results')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--model_pth', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100, mnist, imagenet, ...')
    parser.add_argument('--pretraining-dataset', type=str, default='imagenet')
    parser.add_argument('--epochs', type=int, default=200, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--test-batch-size', type=int, default=256, help='testing batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='train, val image size (pixels)')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train-cls', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--optimizer', choices=['SGD', 'Adam', 'AdamW', 'RMSProp'], default='Adam', help='optimizer')
    parser.add_argument('--lr0', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--decay', type=float, default=5e-5, help='weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing epsilon')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    parser.add_argument('--kd_model_name', default=None, type=str)
    parser.add_argument('--kd_model_checkpoint', default=None, type=str)
    parser.add_argument('--alpha_kd', default=5, type=float)
    parser.add_argument('--use_kd_only_loss', action='store_true', default=False)

    parser.add_argument('--in_channels', type=int, default=3, metavar='N', help='Image input channels')
    parser.add_argument('--prune_type', type=str, default='activation', help='weight / activation')
    parser.add_argument('--prune_criterion', type=str, default='lnorm', help='importance criterion for pruning score matrix (lnorm, random)')
    parser.add_argument('--prune_dim', type=int, default=0, help='for weights: {filters: 0, channels: 1}, for activation: {channels: 0}')
    parser.add_argument('--prune_n', type=int, default=2)
    parser.add_argument('--prune_begin', type=float, default=.1)
    parser.add_argument('--prune_freq', type=int, default=1)
    parser.add_argument('--prune_end', type=float, default=.9)
    parser.add_argument('--prune_init_value', type=float, default=0.0)
    parser.add_argument('--prune_final_value', type=float, default=0.25)
    parser.add_argument('--layers2skip', nargs='+', type=str)
    parser.add_argument('--save_masks', action='store_true', default=False)

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert opt.batch_size != -1, 'AutoBatch is coming soon for classification, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Parameters
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # Train
    train(opt, device)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
