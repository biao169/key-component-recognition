# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
[1]	A. Kolesnikov, L. Beyer, X. Zhai, et al., "Big Transfer (BiT): General Visual Representation Learning," in Computer Vision – ECCV 2020, Cham, 2020, pp. 491-507: Springer International Publishing.
"""

# Lint as: python3
"""Fine-tune a BiT model on some downstream dataset."""
# !/usr/bin/env python3
# coding: utf-8
from os.path import join as pjoin  # pylint: disable=g-importing-member
import time

import numpy as np
import torch
import torchvision as tv

import networks.bit_pytorch.fewshot as fs
import networks.bit_pytorch.lbtoolbox as lb
import networks.bit_pytorch.models as models

from networks.bit_pytorch import bit_common
from networks.bit_pytorch import bit_hyperrule


def topk(output, target, ks=(1,)):
    """Returns one boolean vector for each k, whether the target is within the output's top-k."""
    _, pred = output.topk(max(ks), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].max(0)[0] for k in ks]


def recycle(iterable):
    """Variant of itertools.cycle that does not save iterates."""
    while True:
        for i in iterable:
            yield i


def mktrainval(args, logger):
    """Returns train and validation datasets."""
    precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)
    train_tx = tv.transforms.Compose([
        tv.transforms.Resize((precrop, precrop)),
        tv.transforms.RandomCrop((crop, crop)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if args.dataset == "cifar10":
        train_set = tv.datasets.CIFAR10(args.datadir, transform=train_tx, train=True, download=True)
        valid_set = tv.datasets.CIFAR10(args.datadir, transform=val_tx, train=False, download=True)
    elif args.dataset == "cifar100":
        train_set = tv.datasets.CIFAR100(args.datadir, transform=train_tx, train=True, download=True)
        valid_set = tv.datasets.CIFAR100(args.datadir, transform=val_tx, train=False, download=True)
    elif args.dataset == "imagenet2012":
        train_set = tv.datasets.ImageFolder(pjoin(args.datadir, "train"), train_tx)
        valid_set = tv.datasets.ImageFolder(pjoin(args.datadir, "val"), val_tx)
    else:
        raise ValueError(f"Sorry, we have not spent time implementing the "
                         f"{args.dataset} dataset in the PyTorch codebase. "
                         f"In principle, it should be easy to add :)")

    if args.examples_per_class is not None:
        logger.info(f"Looking for {args.examples_per_class} images per class...")
        indices = fs.find_fewshot_indices(train_set, args.examples_per_class)
        train_set = torch.utils.data.Subset(train_set, indices=indices)

    logger.info(f"Using a training set with {len(train_set)} images.")
    logger.info(f"Using a validation set with {len(valid_set)} images.")

    micro_batch_size = args.batch // args.batch_split

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=micro_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    if micro_batch_size <= len(train_set):
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=micro_batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=False)
    else:
        # In the few-shot cases, the total dataset size might be smaller than the batch-size.
        # In these cases, the default sampler doesn't repeat, so we need to make it do that
        # if we want to match the behaviour from the paper.
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=micro_batch_size, num_workers=args.workers, pin_memory=True,
            sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size))

    return train_set, valid_set, train_loader, valid_loader


def run_eval(model, data_loader, device, chrono, logger, step):
    # switch to evaluate mode
    model.eval()

    logger.info("Running validation...")
    # logger.flush()

    all_c, all_top1, all_top5 = [], [], []
    end = time.time()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # measure data loading time
            # chrono._done("eval load", time.time() - end)

            # compute output, measure accuracy and record loss.
            # with chrono.measure("eval fprop"):
            if True:
                logits = model(x)
                c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
                top1, top5 = topk(logits, y, ks=(1, 5))
                all_c.extend(c.cpu())  # Also ensures a sync point.
                all_top1.extend(top1.cpu())
                all_top5.extend(top5.cpu())

        # measure elapsed time
        end = time.time()

    model.train()
    logger.info(f"Validation@{step} loss {np.mean(all_c):.5f}, "
                f"top1 {np.mean(all_top1):.2%}, "
                f"top5 {np.mean(all_top5):.2%}")
    logger.flush()
    return all_c, all_top1, all_top5


def mixup_data(x, y, l):
    """Returns mixed inputs, pairs of targets, and lambda"""
    indices = torch.randperm(x.shape[0]).to(x.device)

    mixed_x = l * x + (1 - l) * x[indices]
    y_a, y_b = y, y[indices]
    return mixed_x, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, l):
    return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)


def getdata():
    from dataset.get_dataset import Dataset_cv as MydataSet
    img_size = (224, 224)
    transformer = tv.transforms.Compose([tv.transforms.RandomCrop(size=img_size[0])
                                         , tv.transforms.RandomHorizontalFlip(0.3)
                                         , tv.transforms.RandomVerticalFlip(0.3)
                                         , tv.transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.4, hue=0.1)
                                         , tv.transforms.RandomAffine(degrees=160)
                                         , tv.transforms.Resize(img_size[0])
                                         , tv.transforms.ToTensor()
                                         , tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                      ])
    test_transformer = tv.transforms.Compose([
        tv.transforms.ToTensor()
        , tv.transforms.Resize(size=img_size[0])
        , tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    imagePath = r'E:\01_workPicture\automobile components images\data_ori'
    trainLabelfile = r'train_list.txt'
    testLabelfile = r'test_list.txt'
    # 训练过程，所有记录的输出保存地址
    trainOutput_root = r'G:\pythonFiles\05-train\pro-04\output'  ## 训练的所有输出

    train_dataset = MydataSet(imagePath=imagePath, labelfile=trainLabelfile, imgSize=img_size,
                              transform=transformer, shuffer=False, online=True)
    test_dataset = MydataSet(imagePath=imagePath, labelfile=testLabelfile, imgSize=img_size,
                             transform=test_transformer, shuffer=False,
                             online=True)

    print('train_dataset=', len(train_dataset), 'test_dataset=', len(test_dataset))
    print('classification detail:', train_dataset.class_num)

    train_loader = torch.utils.data.DataLoader(
                                  train_dataset, batch_size=128, shuffle=True,
                                  pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
                                  test_dataset, batch_size=128, shuffle=True,
                                  pin_memory=True, drop_last=False)
    return train_dataset, test_dataset, train_loader, test_loader

def self_args():
    import ml_collections
    config = ml_collections.ConfigDict()
    config.model = 'BiT-M-R50x1'
    config.logdir = r'I:\python\00-work\BiT\result'
    config.name = 'BiT'
    config.base_lr = 0.01
    config.batch_split = 1
    config.eval_every = 1

    config.save = True
    config = config.lock()
    return config


def main(args=None):
    args = self_args()
    logger = bit_common.setup_logger(args)

    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Going to train on {device}")

    # train_set, valid_set, train_loader, valid_loader = mktrainval(args, logger)
    train_set, valid_set, train_loader, valid_loader = getdata()

    logger.info(f"Loading model from {args.model}.npz")
    # model = models.KNOWN_MODELS[args.model](head_size=len(valid_set.classes), zero_head=True)
    model = models.KNOWN_MODELS[args.model](head_size=6, zero_head=True)
    # model.load_from(np.load(f"{args.model}.npz"))
    #
    # logger.info("Moving model onto all GPUs")
    # model = torch.nn.DataParallel(model)

    # Optionally resume from a checkpoint.
    # Load it to CPU first as we'll move the model to GPU later.
    # This way, we save a little bit of GPU memory when loading.
    step = 0

    # Note: no weight-decay!
    optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    # Resume fine-tuning if we find a saved model.
    savename = pjoin(args.logdir, args.name, "bit.pth.tar")
    try:
        logger.info(f"Model will be saved in '{savename}'")
        checkpoint = torch.load(savename, map_location="cpu") #.to(device)
        logger.info(f"Found saved model to resume from at '{savename}'")

        step = checkpoint["step"]
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])
        logger.info(f"Resumed at step {step}")
    except FileNotFoundError:
        logger.info("Fine-tuning from BiT")

    from utils_tool.log_utils import Summary_Log
    heads = {'train': ['loss', 'lr'], 'test': ['loss', 'top1', 'top5']}
    writer_csv = Summary_Log(args.logdir, heads, write_csv=True, tm_str='',
                             save_log=False, tensorboard_mode='train-test',
                             new_thread=True)

    model = model.to(device)

    optim.zero_grad()

    model.train()
    mixup = bit_hyperrule.get_mixup(len(train_set))
    cri = torch.nn.CrossEntropyLoss().to(device)

    logger.info("Starting training!")
    chrono = lb.Chrono()
    accum_steps = 0
    mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1
    end = time.time()

    # with lb.Uninterrupt() as u:
    for i in range(20000):
        acc = 0
        lr0 = 0
        for ii, (x, y) in enumerate(train_loader):
            # measure data loading time, which is spent in the `for` statement.
            # chrono._done("load", time.time() - end)

            # if u.interrupted:
            #     break

            # Schedule sending to GPU(s)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Update learning-rate, including stop training if over.
            lr = bit_hyperrule.get_lr(step, len(train_set), args.base_lr)

            if lr is None:
                # lr=0
                break
            for param_group in optim.param_groups:
                param_group["lr"] = lr
            if mixup > 0.0:
                x, y_a, y_b = mixup_data(x, y, mixup_l)

            # compute output
            # with chrono.measure("fprop"):
            if True:
                logits = model(x)
                if mixup > 0.0:
                    c = mixup_criterion(cri, logits, y_a, y_b, mixup_l)
                else:
                    c = cri(logits, y)
                c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.

            # Accumulate grads
            # with chrono.measure("grads"):
            if True:
                (c / args.batch_split).backward()
                accum_steps += 1

            accstep = f" ({accum_steps}/{args.batch_split})" if args.batch_split > 1 else ""
            logger.info(
                f"[step {step}{accstep}]: loss={c_num:.5f} (lr={lr:.1e})")  # pylint: disable=logging-format-interpolation
            logger.flush()
            acc += c_num
            lr0 += lr
            data = {'loss': float(c_num), 'lr': lr}
            writer_csv.add_scalars('train', data, step=step, tolerant=True)

            # print('\n----  step:%d  '%step, 'i ',ii, '\n', flush=True)

            # Update params
            if accum_steps == args.batch_split:
                if True:
                # with chrono.measure("update"):
                    optim.step()
                    optim.zero_grad()
                step += 1
                accum_steps = 0
                # Sample new mixup ratio for next batch
                mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1

                # Run evaluation and save the model.
                if args.eval_every and step % args.eval_every == 0:
                    all_c, all_top1, all_top5 = run_eval(model, valid_loader, device, chrono, logger, step)
                    if args.save:
                        torch.save({
                            "step": step,
                            "model": model.state_dict(),
                            "optim": optim.state_dict(),
                        }, savename)
                    data = {'loss': np.mean(all_c), 'top1': np.mean(all_top1), 'top5': np.mean(all_top5)}
                    writer_csv.add_scalars('test', data, step=step, tolerant=True)


            end = time.time()
        # acc = acc/len(train_loader)
        # lr0 = lr0/len(train_loader)


        # writer_csv.write_csv_all(None, False)

        # Final eval at end of training.
        all_c, all_top1, all_top5 = run_eval(model, valid_loader, device, chrono, logger, step='end')

        # data = {'loss': np.mean(all_c), 'top1': np.mean(all_top1), 'top5': np.mean(all_top5)}
        # writer_csv.add_scalars('test', data, step=i, tolerant=True)

    logger.info(f"Timings:\n{chrono}")


def evaluation():
    # from dataset.get_dataset import Dataset_cv as MydataSet
    from dataset.get_dataset import Dataset_Concentration as MydataSet
    img_size = (224, 224)
    test_transformer = tv.transforms.Compose([
        tv.transforms.ToTensor()
        , tv.transforms.Resize(size=img_size[0])
        , tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    imagePath = r'E:\01_workPicture\automobile components images\data_device'
    trainLabelfile = r'train_list.txt'
    testLabelfile = r'test_list.txt'
    # 训练过程，所有记录的输出保存地址
    trainOutput_root = r'G:\pythonFiles\05-train\pro-04\output'  ## 训练的所有输出

    test_dataset = MydataSet(imagePath=imagePath, labelfile=testLabelfile, imgSize=img_size,
                             transform=test_transformer, shuffer=False,
                             online=True, k=1.2)
    test_loader = torch.utils.data.DataLoader(
                            test_dataset, batch_size=128, shuffle=True,
                            pin_memory=True, drop_last=False)

    args = self_args()
    logger = bit_common.setup_logger(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.KNOWN_MODELS[args.model](head_size=6, zero_head=True)
    savename = pjoin(args.logdir, args.name, "bit.pth.tar")
    checkpoint = torch.load(savename, map_location="cpu")  # .to(device)
    logger.info(f"Found saved model to resume from at '{savename}'")
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()


    all_c, all_top1, all_top5 = run_eval(model, test_loader, device, None, logger, step='end')
    print(all_c, all_top1, all_top5)
    data = {'loss': np.mean(all_c), 'top1': np.mean(all_top1), 'top5': np.mean(all_top5)}
    print(data)


if __name__ == "__main__":
    parser = bit_common.argparser(models.KNOWN_MODELS.keys())
    parser.add_argument("--datadir", required=True,
                        help="Path to the ImageNet data folder, preprocessed for torchvision.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")
    parser.add_argument("--no-save", dest="save", action="store_false")
    main(parser.parse_args())
