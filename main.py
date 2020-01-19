import torch
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tensorboardX import SummaryWriter
from torchvision import transforms
from dcgan import Generator, Discriminator, weight_init, nz
from torch.utils.data import DataLoader

def create_dataset(bs):
    dataset = torchvision.datasets.CIFAR10(
        './data/',
        download = True,
        train = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    train_loader = DataLoader(dataset, batch_size = bs, shuffle = True)
 
    return train_loader

def train(config):
    # Configuration
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = config.batch_size
    epoch = config.epoch
    full_gen_model_path = config.gene_model_path
    full_dis_model_path = config.disc_model_path
    summary_path = config.summary_path

    # Create networks
    netg = Generator().to(device)
    if os.path.exists(full_gen_model_path):
        netg.load_state_dict(torch.load(full_gen_model_path))
    else:
        model_par_dir = '/'.join(full_gen_model_path.split('/')[:-1])
        if not os.path.exists(model_par_dir):
            os.makedirs(model_par_dir)
        weight_init(netg)
    netg.train()
    
    netd = Discriminator().to(device)
    if os.path.exists(full_dis_model_path):
        netd.load_state_dict(torch.load(full_dis_model_path))
    else:
        model_par_dir = '/'.join(full_dis_model_path.split('/')[:-1])
        if not os.path.exists(model_par_dir):
            os.makedirs(model_par_dir)
        weight_init(netd)
    netd.train()

    # Optimizer and loss function
    optimizerg = torch.optim.Adam(netg.parameters(), lr = 1e-3, betas = [0.5, 0.999])
    optimizerd = torch.optim.Adam(netd.parameters(), lr = 1e-3, betas = [0.5, 0.999])
    loss_func = torch.nn.BCELoss()
    fixed_noise = torch.randn((batch_size, nz, 1, 1), device = device, dtype = torch.float32)
    # Dataset
    train_loader = create_dataset(batch_size)

    # # Visualize the training data
    # batch = next(iter(train_loader))
    # plt.figure(figsize = (8, 8))
    # plt.axis('off')
    # plt.title('Training data')
    # plt.imshow(np.transpose(torchvision.utils.make_grid(batch[0], padding = 2, normalize = True).cpu(), (1, 2, 0)))
    # plt.show()

    # Store results
    g_losses, d_losses = [], []
    writer = SummaryWriter(summary_path)
    total_iter = 1

    # Start!
    for e in range(1, epoch + 1):
        # Training
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            bs = label.size()[0]

            '''
                Train the discriminator: Max(log(D(x)) + log(1 - D(G(z))))
            '''
            # Train the discriminator with all-real batch
            all_real_label = torch.ones((bs, ), dtype = torch.float32, device = device)
            netd.zero_grad()
            pred_real = netd(data)
            loss_real = loss_func(pred_real, all_real_label)
            loss_real.backward()
            # Train the discriminator with all-fake batch
            noise = torch.randn((bs, nz, 1, 1), dtype = torch.float32, device = device)
            fake_img = netg(noise)
            all_fake_label = torch.zeros((bs, ), dtype = torch.float32, device = device)
            pred_fake = netd(fake_img.detach())
            loss_fake = loss_func(pred_fake, all_fake_label)
            loss_fake.backward()
            # Update the gradient
            loss_d = loss_real + loss_fake
            optimizerd.step()

            '''
                Train the generator: Max(log(D(G(z))))
            '''
            # Train the generator with fake images
            netg.zero_grad()
            pred = netd(fake_img)
            loss_g = loss_func(pred, all_real_label) # Force the generator to synthesize real images
            loss_g.backward()
            optimizerg.step()

            # Visualize the result
            g_losses.append(loss_g)
            d_losses.append(loss_d)
            writer.add_scalar('Train/loss_g', loss_g.item(), total_iter)
            writer.add_scalar('Train/loss_d', loss_d.item(), total_iter)
            print('[Epoch %d|TotalIter %d] -- LossG = %.6f, LossD = %.6f' % (e, total_iter, loss_g.item(), loss_d.item()))
            total_iter += 1

        # Visualize the fake images
        if e % 2 == 0:
            netd.eval()
            netg.eval()
            with torch.no_grad():
                test_fake_imgs = netg(fixed_noise).detach()
                test_fake_imgs = torchvision.utils.make_grid(test_fake_imgs[:64], padding = 2, normalize = True).detach().cpu().numpy()
                writer.add_image('Train/Fake_Imgs_After_{}_Epochs'.format(e), test_fake_imgs, e)
            netd.train()
            netg.train()

        # Save the model
        if e % 2 == 0:
            torch.save(netg.state_dict(), full_gen_model_path)
            torch.save(netd.state_dict(), full_dis_model_path)

    writer.close()

def fake(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    full_gen_model_path = config.gene_model_path

    netg = Generator().to(device)
    netg.load_state_dict(torch.load(full_gen_model_path))
    netg.eval()

    bs = 64
    fixed_noise = torch.randn((bs, nz, 1, 1), device = device, dtype = torch.float32)
    fake_imgs = netg(fixed_noise).detach()
    plt.figure(figsize = (8, 8))
    plt.axis('off')
    plt.title('Fake images')
    plt.imshow(np.transpose(torchvision.utils.make_grid(fake_imgs, padding = 2, normalize = True).cpu(), (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', type = bool, default = True)
    parser.add_argument('--gene_model_path', type = str, default = './model/gene.pkl')
    parser.add_argument('--disc_model_path', type = str, default = './model/disc.pkl')
    parser.add_argument('--summary_path', type = str, default = './summary/')
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--epoch', type = int, default = 30)

    config = parser.parse_args()
    if config.is_train:
        train(config)
    else:
        fake(config)
