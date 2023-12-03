import learn2learn as l2l

from data.loaders import CIFARFS, CUBirds200, MiniImageNet, Omniglotmix, TieredImagenet,FC100


def gen_tasks(dataname, root, image_transforms=None, target_transforms=None, download=False, **task_transforms):
    """ Generates tasks from the specified Dataset
    Arguments:- 
      root: root folder of Omniglot dataset
      image_transforms: transforms to be applied to images before loading in the dataloader
      target_transforms: transforms to be applied to target classes before loading in the dataloader
      task_transforms: specify n_ways, k_shots, q_queries, num_tasks to create and classes to sample 
                    tasks from (if Omniglot) or mode: train/valid/test split to load (if MiniImageNet)
                    or mode: train/validation/test split to load (if CIFARFS)"""

    arguments = {'n_ways': 0, 'k_shots': 0, 'q_queries': 0, 'classes': 0, 'mode': 0, 'num_tasks': -1}
    arguments.update(task_transforms)
    n_ways = arguments['n_ways']
    k_shots = arguments['k_shots']
    q_shots = arguments['q_shots']
    classes = arguments['classes']
    num_tasks = arguments['num_tasks']
    mode = arguments['mode']

    if (dataname == 'omniglot'):
        omniglot = Omniglotmix(
            root, download=download, transform=image_transforms, target_transforms=target_transforms)
        dataset = l2l.data.MetaDataset(omniglot)

        trans = [
            l2l.data.transforms.FusedNWaysKShots(dataset,
                                                 n=n_ways,
                                                 k=k_shots + q_shots,
                                                 filter_labels=classes),
            l2l.data.transforms.LoadData(dataset),
            l2l.data.transforms.RemapLabels(dataset),
            l2l.data.transforms.ConsecutiveLabels(dataset),
            l2l.vision.transforms.RandomClassRotation(
                dataset, [0.0, 90.0, 180.0, 270.0])
        ]
        tasks = l2l.data.TaskDataset(dataset, task_transforms=trans, num_tasks=num_tasks)

    elif (dataname == 'miniimagenet'):
        mini = MiniImageNet(root, mode, transform=image_transforms,
                            target_transform=target_transforms, download=download) #传递[图片，类]

        dataset = l2l.data.MetaDataset(mini)#对元学习的快速映射和采样

        trans = [
            l2l.data.transforms.FusedNWaysKShots(dataset,
                                                 n=n_ways,
                                                 k=k_shots + q_shots), # n是标签，K是每个标签取多少个图片
            l2l.data.transforms.LoadData(dataset),#在给定的数据集中加载示例
            l2l.data.transforms.RemapLabels(dataset),#对n个类进行0...n的重新排序
            l2l.data.transforms.ConsecutiveLabels(dataset)#
        ]
        tasks = l2l.data.TaskDataset(dataset, task_transforms=trans, num_tasks=num_tasks)

    elif (dataname == 'tiered'):
        tiered = TieredImagenet(root, mode, transform=image_transforms,
                            target_transform=target_transforms, download=download)

        dataset = l2l.data.MetaDataset(tiered)

        trans = [
            l2l.data.transforms.FusedNWaysKShots(dataset,
                                                 n=n_ways,
                                                 k=k_shots + q_shots),
            l2l.data.transforms.LoadData(dataset),
            l2l.data.transforms.RemapLabels(dataset),
            l2l.data.transforms.ConsecutiveLabels(dataset)
        ]
        tasks = l2l.data.TaskDataset(dataset, task_transforms=trans, num_tasks=num_tasks)
    elif (dataname == 'cifarfs'):
        cfs = CIFARFS(root, mode, transform=image_transforms,
                            target_transform=target_transforms, download=download)
        dataset = l2l.data.MetaDataset(cfs)

        trans = [
            l2l.data.transforms.FusedNWaysKShots(dataset,
                                                 n=n_ways,
                                                 k=k_shots + q_shots),
            l2l.data.transforms.LoadData(dataset),
            l2l.data.transforms.RemapLabels(dataset),
            l2l.data.transforms.ConsecutiveLabels(dataset)
        ]
        tasks = l2l.data.TaskDataset(dataset, task_transforms=trans, num_tasks=num_tasks)
    elif (dataname == 'fc100'):
        fc= FC100(root, mode, transform=image_transforms,
                            target_transform=target_transforms, download=download)
        dataset = l2l.data.MetaDataset(fc)

        trans = [
            l2l.data.transforms.FusedNWaysKShots(dataset,
                                                 n=n_ways,
                                                 k=k_shots + q_shots),
            l2l.data.transforms.LoadData(dataset),
            l2l.data.transforms.RemapLabels(dataset),
            l2l.data.transforms.ConsecutiveLabels(dataset)
        ]
        tasks = l2l.data.TaskDataset(dataset, task_transforms=trans, num_tasks=num_tasks)

    return tasks
