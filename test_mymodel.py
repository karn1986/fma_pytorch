"""
@author: KAgarwal

Adapted from one of the assignments of 
GaTech's CS 7643 Deep Learning course 
"""
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from data import audio_clips
from models import MyModel

parser = argparse.ArgumentParser(description='FMA')
parser.add_argument('--config', default='configs/config_mymodel.yaml')

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct / batch_size

    return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    """Run the trained model on the test set"""
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    model = MyModel()
    model.load_state_dict(torch.load('checkpoints/mymodel.pth'))

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    train_dataset = audio_clips(audio_dir = args.audio_dir,
                                meta_dir = args.meta_dir,
                                transform = None,
                                mode = 'train')
    enc = train_dataset.enc
    test_dataset = audio_clips(audio_dir = args.audio_dir,
                               meta_dir = args.meta_dir,
                               transform = None,
                               mode = 'test',
                               enc = enc)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    acc = AverageMeter()
    num_class = 16
    cm =torch.zeros(num_class, num_class)
    for data, target in test_loader:
        out = model(data)
        batch_acc = accuracy(out, target)
        acc.update(batch_acc, out.shape[0])
        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1
    cm = cm / cm.sum(1)
    
    print('Test Acccuracy: {:.4f}'.format(acc.avg))
    per_cls_acc = cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Test Accuracy of Class {}: {:.4f}".format(i, acc_i))

if __name__ == '__main__':
    main()