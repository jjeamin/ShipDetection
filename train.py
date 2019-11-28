import torch
from datasets import CustomDataset, custom_collate
from augmentation import *
import utils

if torch.cuda.is_available():
    device = 'cuda'
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    #torch.set_default_tensor_type('torch.FloatTensor')

transform = Compose([Resize((1000, 1000)),
                     ToTensor()])

custom_datasets = CustomDataset(transform=transform)

custom_loader = torch.utils.data.DataLoader(
        dataset=custom_datasets,
        batch_size=2,
        shuffle=True,
        collate_fn=custom_collate)

for i, (img, targets) in enumerate(custom_loader):
    utils.save_boxing_image(img[0], targets[0])

    break



'''
net = CornerNet(classes=configs['classes']).to(device)
optimizer = optim.Adam(net.parameters(), lr=configs['lr'])
criterion = CornerNet_Loss()

if configs['mode'] == 'train':
    for e in range(configs['epoch']):
        total_loss = 0
        for i, (images, targets) in enumerate(custom_loader):
            images = images.to(device)
            # targets : [heat_tl, heat_br, embed_tl, embed_br, off_tl, off_br]
            targets = [target.to(device) for target in targets]

            optimizer.zero_grad()
            outputs = net(images)
            loss, hist = criterion(outputs,targets)
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print('[Epoch %d, Batch %d/%d] [totalLoss:%.6f] [ht_loss:%.6f, off_loss:%.6f, pull_loss:%.6f, push_loss:%.6f]'
                  % (e, i, len(custom_loader), loss.item(), hist[0], hist[1], hist[2], hist[3]))

        print('saveing model')
        torch.save({
            'epoch': e,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, './log/hour104_cornernet.ckpt')

if configs['mode'] == 'test':
    checkpoint = torch.load('./log/hour104_cornernet_1.ckpt')
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
'''