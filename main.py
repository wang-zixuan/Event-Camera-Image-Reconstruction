import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import network
from dataloader import event_loader
from utils import accuracy, preprocess, pytorch_ssim, logger


# parser
parser = argparse.ArgumentParser(description='Event camera image reconstruction')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--train_batch', default=32, type=int, help='train batch size')
parser.add_argument('--test_batch', default=16, type=int, help='test batch size')
parser.add_argument('--channels', '-c', default=1, type=int, help='train channels')
parser.add_argument('--fixed', '-f', action='store_true', help='fixed value of events')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_ssim = 0  # best test ssim
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# logs
log = logger.setup_logger(os.path.join(os.getcwd(), 'experiment_' + str(args.channels) + 'ch_'
                                       + ('fixed_' if args.fixed else '') + 'log.log'))

for key, value in sorted(vars(args).items()):
    log.info(str(key) + ':' + str(value))

# get dataset
print('==> Preparing data..')
if not os.path.exists('data/train_' + str(args.channels) + 'ch' + ('_fixed' if args.fixed else '') + '.npy'):
    print('==> Generating data now..')
    preprocess.generate_dataset(args.channels, fixed=args.fixed)
    print('Generating data done.')

# data configuration
train_data = np.load('data/train_' + str(args.channels) + 'ch' + ('_fixed' if args.fixed else '') + '.npy')
test_data = np.load('data/test_' + str(args.channels) + 'ch' + ('_fixed' if args.fixed else '') + '.npy')

train_label = np.load('data/train_label.npy')
test_label = np.load('data/test_label.npy')

# transforms
trans = transforms.ToTensor()
dataset_train = event_loader.EventDataset(train_data, train_label, trans)
dataset_test = event_loader.EventDataset(test_data, test_label, trans)

# loader
train_loader = DataLoader(dataset=dataset_train, batch_size=args.train_batch, num_workers=8, shuffle=True)
test_loader = DataLoader(dataset=dataset_test, batch_size=args.test_batch, num_workers=8, shuffle=False)

print('Preparing data done.')

# net
print('==> Building model..')
net = network.UNet(channels=args.channels)
net = net.to(device)

writer = SummaryWriter('runs/eventcamera_experiment_' + str(args.channels) + ('_fixed' if args.fixed else ''))
print('Building model done.')

test_output_image = np.zeros((len(test_label), 180, 240), dtype='float')

criterion = pytorch_ssim.SSIM(window_size=11)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.001)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('checkpoint/ckpt' + str(args.channels) + ('_fixed' if args.fixed else '') + '.pth')
    net.load_state_dict(checkpoint['net_params'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_psnr = checkpoint['psnr']
    best_ssim = checkpoint['ssim']
    start_epoch = checkpoint['epoch']
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 35], gamma=0.1, last_epoch=start_epoch)
    log.info("=> loaded checkpoint (epoch %d)" % start_epoch)

else:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 35], gamma=0.1)
    log.info("None loadmodel => will start from scratch.")

log.info('Number of model parameters: %d' % sum([p.data.nelement() for p in net.parameters()]))


def train_model(epoch):
    log.info('This is %d-th train epoch, learning rate: %f' % (epoch, scheduler.get_lr()[0]))
    net.train()
    train_loss = 0
    cur_ssim, cur_psnr = 0, 0
    for batch_idx, (input, target) in enumerate(train_loader):
        inputs, targets = input.to(device).float(), target.to(device).float()
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        cur_ssim += accuracy.accuracy_of_test(outputs.float(), targets.float())[0]
        cur_psnr += accuracy.accuracy_of_test(outputs.float(), targets.float())[1]

        if batch_idx % 2 == 0:  # every 2 mini-batches...
            # ...log the running loss
            writer.add_scalar('training_loss', loss, epoch * len(train_loader) + batch_idx)

    scheduler.step(epoch)
    log.info('train epoch: %d, average train loss: %.5f' % (epoch, train_loss / len(train_loader)))
    print('Train Loss: %.3f | SSIM: %.3f | PSNR: %.3f'
          % (train_loss / len(train_loader), cur_ssim / len(train_loader), cur_psnr / len(train_loader)))


def test_model(epoch):
    log.info('This is %d-th test epoch.' % epoch)
    global best_ssim
    global test_output_image

    net.eval()
    test_loss = 0
    cur_ssim, cur_psnr = 0, 0
    idx = 0

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(test_loader):
            inputs, targets = input.to(device).float(), target.to(device).float()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            cur_ssim += accuracy.accuracy_of_test(outputs.float(), targets.float())[0]
            cur_psnr += accuracy.accuracy_of_test(outputs.float(), targets.float())[1]

            cur_outpus = outputs.cpu().detach().numpy()
            for batch in range(cur_outpus.shape[0]):
                test_output_image[idx] = cur_outpus[batch].reshape(256, 256)[:180, :240]
                idx += 1

    cur_ssim, cur_psnr = cur_ssim / len(test_loader), cur_psnr / len(test_loader)
    print('Test Loss: %.3f | SSIM: %.3f | PSNR: %.3f'
          % (test_loss / len(test_loader), cur_ssim, cur_psnr))

    # Save checkpoint
    if cur_ssim > best_ssim:
        best_ssim = cur_ssim
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        state = {
            'net_params': net.state_dict(),
            'psnr': cur_psnr,
            'ssim': cur_ssim,
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict()
        }

        torch.save(state, 'checkpoint/ckpt' + str(args.channels) + ('_fixed' if args.fixed else '') + '.pth')
        np.save('result/test_output_image' + str(args.channels) + ('_fixed' if args.fixed else '') + '.npy',
                test_output_image)

    log.info('test epoch: %d, average test loss: %.5f' % (epoch, test_loss / len(test_loader)))


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch + 50):
        print('epoch:', epoch)
        train_model(epoch)
        test_model(epoch)
