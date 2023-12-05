import random

import learn2learn as l2l
import numpy as np
import torch
from torch.nn import functional as F
from data.taskers import gen_tasks
from PIL.Image import LANCZOS
from torchvision import transforms

from src.zoo.archs import CCVAE


def setup(dataset, root, n_ways, k_shots, q_shots, order, inner_lr, device, download, args):

    if (dataset == 'miniimagenet'):
        # Generating tasks and model according to the MAML implementation for MiniImageNet
        train_tasks = gen_tasks(dataset, root, download=download, mode='train',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)
        valid_tasks = gen_tasks(dataset, root, download=download, mode='validation',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)
        test_tasks = gen_tasks(dataset, root, download=download, mode='test',
                               n_ways=n_ways, k_shots=k_shots, q_shots=q_shots, num_tasks=600)
        learner = CCVAE(in_channels=3, base_channels=32,
                        n_ways=n_ways,  args=args, latent_dim_l=args.zl, latent_dim_a=args.za)

    elif (dataset == 'tiered'):
        image_trans = transforms.Compose([transforms.ToTensor()])
        train_tasks = gen_tasks(dataset, root, image_transforms=image_trans, download=download, mode='train',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)  # , num_tasks=50000)
        valid_tasks = gen_tasks(dataset, root, image_transforms=image_trans, download=download, mode='validation',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)  # , num_tasks=10000)
        test_tasks = gen_tasks(dataset, root, image_transforms=image_trans, download=download, mode='test',
                               n_ways=n_ways, k_shots=k_shots, q_shots=q_shots, num_tasks=2000)
        learner = CCVAE(in_channels=3, base_channels=32,
                        n_ways=n_ways,  args=args, latent_dim_l=args.zl, latent_dim_a=args.za)

    elif (dataset == 'cifarfs'):
        image_trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([84,84])])
        train_tasks = gen_tasks(dataset, root, image_transforms=image_trans, download=download, mode='train',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)  # , num_tasks=50000)
        valid_tasks = gen_tasks(dataset, root, image_transforms=image_trans, download=download, mode='validation',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)  # , num_tasks=10000)
        test_tasks = gen_tasks(dataset, root, image_transforms=image_trans, download=download, mode='test',
                               n_ways=n_ways, k_shots=k_shots, q_shots=q_shots, num_tasks=2000)
        learner = CCVAE(in_channels=3, base_channels=32,
                        n_ways=n_ways,  args=args, latent_dim_l=args.zl, latent_dim_a=args.za)
    elif (dataset == 'fc100'):
        image_trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([84,84])])
        train_tasks = gen_tasks(dataset, root, image_transforms=image_trans, download=download, mode='train',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)  # , num_tasks=50000)
        valid_tasks = gen_tasks(dataset, root, image_transforms=image_trans, download=download, mode='validation',
                                n_ways=n_ways, k_shots=k_shots, q_shots=q_shots)  # , num_tasks=10000)
        test_tasks = gen_tasks(dataset, root, image_transforms=image_trans, download=download, mode='test',
                               n_ways=n_ways, k_shots=k_shots, q_shots=q_shots, num_tasks=2000)
        learner = CCVAE(in_channels=3, base_channels=32,
                        n_ways=n_ways,  args=args, latent_dim_l=args.zl, latent_dim_a=args.za)

    learner = learner.to(device)
    # allow_nograd=True if args.pretrained[2]=='freeze' else False
    learner = l2l.algorithms.MAML(
        learner, first_order=order, lr=inner_lr, allow_nograd=False)

    return train_tasks, valid_tasks, test_tasks, learner


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def kl_div(mus, log_vars):
    return - 0.5 * (1 + 2*log_vars - mus**2 - torch.exp(2*log_vars)).sum(dim=1)


def loss(reconst_loss: object, reconst_image, image, logits, labels, mu_a, log_var_a, mu_l, log_var_l, reconstr, wt_ce=1e2, klwt=False, rec_wt=1e-2, beta_l=1, beta_a=1):
    kl_div_s = kl_div(mu_a, log_var_a).mean()
    kl_div_l = kl_div(mu_l, log_var_l).mean()
    if klwt:
        kl_wt = mu_l.shape[-1] / (image.shape[-1] *
                                  image.shape[-2] * image.shape[-3])
    else:
        kl_wt = 1
    ce_loss = torch.nn.CrossEntropyLoss()
    classification_loss = ce_loss(logits, labels)

    rec_loss = reconst_loss(reconst_image, image)

    if reconstr == 'std':
        rec_loss = rec_loss.view(rec_loss.shape[0], -1).sum(dim=-1).mean()
    elif reconstr == 'pp':
        rec_loss = rec_loss.mean()

    L = wt_ce*classification_loss + beta_l*kl_wt*kl_div_l + \
        rec_wt*rec_loss + beta_a*kl_wt*kl_div_s  # -log p(x,y)

    losses = {'elbo': L, 'label_kl': kl_div_l, 'agnostic_kl': kl_div_s,
              'reconstruction_loss': rec_loss, 'classification_loss': classification_loss}
    return losses




def inner_adapt_trident(task, reconst_loss, learner, n_ways, k_shots, q_shots, adapt_steps, device, log_data: bool, args, extra):
    data, labels = task
    if args.dataset == 'miniimagenet':
        data, labels = data.to(device) / 255.0, labels.to(device)
    else:
        data, labels = data.to(device), labels.to(device)

    total = n_ways * (k_shots + q_shots)
    queries_index = np.zeros(total)

    # Extracting the evaluation datums from the entire task set, for the meta gradient calculation
    for offset in range(n_ways):
        queries_index[np.random.choice(
            k_shots+q_shots, q_shots, replace=False) + ((k_shots + q_shots)*offset)] = True
    support = data[np.where(queries_index == 0)]
    support_labels = labels[np.where(queries_index == 0)]
    queries = data[np.where(queries_index == 1)]
    queries_labels = labels[np.where(queries_index == 1)]

    # Logging latent spaces of queries before meta-adaptation
    if extra == "Yes":
        reconst_image, logits, mu_l_0, log_var_l_0, mu_a_0, log_var_a_0 = learner(
                torch.cat([support, queries], dim=0), 'outer')

    
    # Inner adapt step
    for _ in range(adapt_steps):

        reconst_image, logits, mu_l, log_var_l, mu_a, log_var_a = learner(
                torch.cat([support, queries], dim=0), 'inner')

        adapt_loss = loss(reconst_loss, reconst_image, support,
                          logits, support_labels, mu_a, log_var_a, mu_l, log_var_l, args.reconstr, args.wt_ce, args.klwt, args.rec_wt, args.beta_l, args.beta_a)

        learner.adapt(adapt_loss['elbo'])

        for p in learner.parameters():
            torch.clamp_(p,-1e5,1e5)



    reconst_image, logits, mu_l, log_var_l, mu_a, log_var_a = learner(
            torch.cat([support, queries], dim=0), 'outer')


    eval_loss = loss(reconst_loss, reconst_image, queries,
                     logits, queries_labels, mu_a, log_var_a, mu_l, log_var_l, args.reconstr, args.wt_ce, args.klwt, args.rec_wt, args.beta_l, args.beta_a)
    eval_acc = accuracy(F.softmax(logits, dim=1), queries_labels)
    print('eval_acc:',eval_acc.item())

    if log_data and (extra == 'Yes'):
        return eval_loss, eval_acc, reconst_image.detach().to('cpu'), queries.detach().to('cpu'), mu_l.detach().to('cpu'), log_var_l.detach().to('cpu'), mu_a.detach().to('cpu'), log_var_a.detach().to('cpu'), logits.detach().to('cpu'), queries_labels.detach().to('cpu'), mu_l_0.detach().to('cpu'), log_var_l_0.detach().to('cpu'), mu_a_0.detach().to('cpu'), log_var_a_0.detach().to('cpu')
    elif log_data and (extra == 'No'):
        return eval_loss, eval_acc, reconst_image.detach().to('cpu'), queries.detach().to('cpu'), mu_l.detach().to('cpu'), log_var_l.detach().to('cpu'), mu_a.detach().to('cpu'), log_var_a.detach().to('cpu'), logits.detach().to('cpu'), queries_labels.detach().to('cpu')
    else:
        return eval_loss, eval_acc
