import argparse
import json

#import numpy as np
import tqdm
import torch
from torch import nn, optim

from src.utils2 import Profiler
from src.zoo.tsdvi_utils import inner_adapt_trident, setup

##############
# Parameters #
##############

parser = argparse.ArgumentParser()
parser.add_argument('--cnfg', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--root', type=str)
parser.add_argument('--n-ways', type=int)
parser.add_argument('--k-shots', type=int)
parser.add_argument('--q-shots', type=int)
parser.add_argument('--inner-adapt-steps-train', type=int)
parser.add_argument('--inner-adapt-steps-test', type=int)
parser.add_argument('--inner-lr', type=float)
parser.add_argument('--meta-lr', type=float)
parser.add_argument('--meta-batch-size', type=int)
parser.add_argument('--iterations', type=int)
parser.add_argument('--reconstr', type=str)
parser.add_argument('--wt-ce', type=float)
parser.add_argument('--klwt', type=str)
parser.add_argument('--rec-wt', type=float)
parser.add_argument('--beta-l', type=float)
parser.add_argument('--beta-a', type=float)
parser.add_argument('--zl', type=int, default=64)
parser.add_argument('--za', type=int, default=64)
parser.add_argument('--wm-channels', type=int, default=64)
parser.add_argument('--wn-channels', type=int, default=32)
parser.add_argument('--task-adapt', type=str)
parser.add_argument('--experiment', type=str)
parser.add_argument('--order', type=str)
parser.add_argument('--device', type=str)
parser.add_argument('--download', type=str)


args = parser.parse_args()
with open(args.cnfg) as f:
    parser = argparse.ArgumentParser()
    argparse_dict = vars(args)
    argparse_dict.update(json.load(f))

    args = argparse.Namespace()
    args.__dict__.update(argparse_dict)


# TODO: fix this bool/str shit

if args.order == 'True':
    args.order = True
elif args.order == 'False':
    args.order = False

if args.download == 'True':
    args.download = True
elif args.download == 'False':
    args.download = False

if args.klwt == 'True':
    args.klwt = True
elif args.klwt == 'False':
    args.klwt = False




# Generating Tasks, initializing learners, loss, meta - optimizer and profilers
train_tasks, valid_tasks, _, learner = setup(
    args.dataset, args.root, args.n_ways, args.k_shots, args.q_shots, args.order, args.inner_lr, args.device, download=args.download, args=args)



opt = optim.Adam(learner.parameters(), args.meta_lr)
reconst_loss = nn.MSELoss(reduction='none')
start = 0

profiler=None
modelpath=None

if args.order == False:
    modelpath='TRIDENT_{}_{}-way_{}-shot_{}-queries'.format(args.dataset,args.n_ways, args.k_shots, args.q_shots)
    profiler = Profiler(modelpath, args.experiment, args)

elif args.order == True:
    modelpath='FO-TRIDENT_{}_{}-way_{}-shot_{}-queries'.format(args.dataset, args.n_ways, args.k_shots, args.q_shots)
    profiler = Profiler(modelpath, args.experiment, args)


## Training ##
for iter in tqdm.tqdm(range(start, args.iterations)):
    opt.zero_grad()
    batch_losses = []
    tmp=None
    batch=0
    while batch < args.meta_batch_size:
        ttask = train_tasks.sample()
        model = learner.clone()

        evaluation_loss, evaluation_accuracy = inner_adapt_trident(
            ttask, reconst_loss, model, args.n_ways, args.k_shots, args.q_shots, args.inner_adapt_steps_train, args.device, False, args, "No")

        evaluation_loss['elbo'].backward()

        # Logging per train-task losses and accuracies
        tmp = [(iter*args.meta_batch_size)+batch, evaluation_accuracy.item()]
        tmp = tmp + [a.item() for a in evaluation_loss.values()]
        batch_losses.append(tmp)

        if not torch.lt(evaluation_loss['elbo'],1e5).item():
            print(evaluation_loss['elbo'])
            print("isnan")
            learner.load_state_dict(torch.load('logs/{}/{}/model_{}.pt'.format(modelpath,args.experiment, (iter-10)//500*500),
                                               map_location=args.device))
            opt.zero_grad()
            if batch==args.meta_batch_size-1:
                batch=batch-1
        batch+=1
    vtask = valid_tasks.sample()
    model = learner.clone()

    validation_loss, validation_accuracy = inner_adapt_trident(
        vtask, reconst_loss, model, args.n_ways, args.k_shots, args.q_shots, args.inner_adapt_steps_train, args.device, False, args, "No")

    # Logging per validation-task losses and accuracies
    tmp = [iter, validation_accuracy.item()]
    tmp = tmp + [a.item() for a in validation_loss.values()]

    # Gradient clipping to prevent explosion
    torch.nn.utils.clip_grad_norm_(learner.parameters(), 1)

    # Meta backpropagation of gradients
    for p in learner.parameters():
            p.grad.data.mul_(1.0 / args.meta_batch_size)
    opt.step()

    validation_loss['elbo'].backward()
    # Saving the Logs
    profiler.log_csv(batch_losses, 'train')
    profiler.log_csv(tmp, 'valid')

    # Checkpointing the learner
    if iter % 500 == 0:
        #learner = learner.to('cpu')
        profiler.log_model(learner, opt, iter)
        #learner = learner.to(args.device)
    else:
        continue

profiler.log_model(learner, opt, 'last')
profiler.log_model(learner, opt, iter)
