import torch
import torch.nn as nn
import torch.nn.functional as F


class MyEncoder(torch.nn.Module):

    def __init__(self, frames_memory, size_x, size_y, embedding_size=2048):
        super().__init__()
        self.conv1_large = nn.Conv2d(frames_memory * 3, 10, 3, dilation=8, padding=8)
        self.conv1_med = nn.Conv2d(frames_memory * 3, 10, 3, dilation=4, padding=4)
        self.conv1_small = nn.Conv2d(frames_memory * 3, 10, 3, dilation=2, padding=2)
        self.conv1_tiny = nn.Conv2d(frames_memory * 3, 10, 3, dilation=1, padding=1)

        self.conv2 = nn.Conv2d(40, 20, 3, dilation=4)
        self.conv3 = nn.Conv2d(20, 20, 3, dilation=2)
        self.conv4 = nn.Conv2d(20, 20, 3, dilation=1)
        self.conv5 = nn.Conv2d(20, 20, 3, stride=2)
        self.conv6 = nn.Conv2d(20, 20, 3, stride=2)
        # self.fc1 = nn.Linear(20 * (size_x - 8) * (size_y - 8), num_actions)
        self.fc1 = nn.Linear(22500, embedding_size)

    def forward(self, x):
        x = x.float()
        # print(x.shape)
        layer1 = F.relu(
            torch.cat((self.conv1_large(x), self.conv1_med(x), self.conv1_small(x), self.conv1_tiny(x)), dim=1))
        # print(layer1.shape)
        # print(x.shape)
        x = F.relu(self.conv2(layer1))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = F.relu(self.conv4(x))
        # print(x.shape)
        x = F.relu(self.conv5(x))
        # print(x.shape)
        x = F.relu(self.conv6(x))

        # print(x.shape)
        x = x.view(x.size(0), -1)  # Flatten
        # print(x.shape)
        return self.fc1(x)


class Model(torch.nn.Module):

    def __init__(self, frames_memory, num_actions, size_x, size_y, embedding_size=2048):
        super(Model, self).__init__()
        self.encoder = MyEncoder(frames_memory, size_x, size_y, embedding_size=embedding_size)
        self.fc2 = nn.Linear(embedding_size, num_actions)

    def forward(self, x):
        return self.fc2(self.encoder(x))


class Decoder(torch.nn.Module):
    def __init__(self, embedding_size=2048):
        super().__init__()
        self.embedding_size = embedding_size
        self.fc1 = torch.nn.Linear(embedding_size, 22500)
        self.deconv1 = nn.ConvTranspose2d(20, 20, 3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(20, 20, 3, stride=2)
        self.deconv3 = torch.nn.ConvTranspose2d(20, 20, 3, 1, dilation=1)
        self.deconv4 = torch.nn.ConvTranspose2d(20, 20, 3, 1, dilation=2)
        self.deconv5 = torch.nn.ConvTranspose2d(20, 9, 3, 1, dilation=4)

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)

        x = x.view((-1, 20, 45, 25))
        # print(x.shape)
        x = self.deconv1(x)
        # print(x.shape)
        x = self.deconv2(x)
        # print(x.shape)
        x = self.deconv3(x)
        # print(x.shape)
        x = self.deconv4(x)
        # print(x.shape)
        x = self.deconv5(x)
        # print(x.shape)
        # torch.Size([2, 9, 200, 120])
        # torch.Size([2, 20, 192, 112])
        # torch.Size([2, 20, 188, 108])

        return x


class Autoencoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == '__main__':
    import numpy as np
    import torch.utils.data

    frames_memory = 3  # main.FRAMES_FEED
    data = np.load("frames_20000.npy")
    dataset_size = data.shape[0] - data.shape[0] % frames_memory
    # print(dataset_size, np.prod(data.shape))
    data = data[:dataset_size, :, :, :]
    mean = np.mean(data, axis=(0, 2, 3))
    std = np.std(data, axis=(0, 2, 3))
    data = np.subtract(data, mean[np.newaxis, :, np.newaxis, np.newaxis]) / std[np.newaxis, :, np.newaxis, np.newaxis]
    # print(mean, std)
    # print(data.shape)
    data = data.reshape((dataset_size // frames_memory, frames_memory * 3, 200, 120))
    data_torch: torch.Tensor = torch.from_numpy(data).float()
    dataset = torch.utils.data.TensorDataset(data_torch)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=40, shuffle=True)
    encoder = MyEncoder(frames_memory, None, None, embedding_size=2048)
    decoder = Decoder()
    autoencoder = Autoencoder(encoder, decoder).cuda()


    # loss_fn = torch.nn.MSELoss()
    def loss_fn(_pred, _real):
        return torch.mean((_pred - _real[:_pred.shape[0], :_pred.shape[1], :_pred.shape[2], :_pred.shape[3]]) ** 2)


    optimizer = torch.optim.Adam(autoencoder.parameters())
    for _ in range(10):
        losses = []
        for (x,) in data_loader:
            _x = x.cuda()
            pred = autoencoder(_x)
            loss = loss_fn(pred, _x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        print(np.mean(losses))
    autoencoder.cpu()
    torch.save(autoencoder.encoder.state_dict(), "encoder_state_dict.pth")
