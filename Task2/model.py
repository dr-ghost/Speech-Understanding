import torch
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as f
from tqdm import tqdm

from background import UrbanSound8KDataset, plot_waveform

class AudioClassifier(nn.Module):
    def __init__(self, in_channels : int, n_classes : int) -> None:
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = nn.Dropout(0.3)
        
        self.fc = nn.Linear(64, n_classes)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(self.device)
    
    def forward(self, x : torch.Tensor):
        x = torch.abs(torch.fft.rfft(x, dim=-1))
        x = torch.log1p(x + 1e-6)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.global_avg_pool(x)
        
        x = torch.flatten(x, 1)
        
        #x = self.dropout(x)
        
        x = self.fc(x)
        
        return x
    
    def train(self, num_epochs : int, train_dataset : Dataset, valid_dataset : Dataset, optim):
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
        
        train_loss = []
        train_acc = []
        valid_acc = []
        
        for epoch in tqdm(range(num_epochs)):
            
            train_l = 0
            train_corr = 0
            n_train = 0
            valid_corr = 0
            n_valid = 0
            
            for features, labels in train_dataloader:
                #(B, C, nW, nF)
                features = features.to(self.device)
                #(B, )
                labels = labels.to(self.device)
                
                optim.zero_grad()
                
                y_pred = self(features)
                
                loss = nn.CrossEntropyLoss()(y_pred, labels)
                
                loss.backward()
                
                optim.step()
                
                with torch.no_grad():
                    train_l += loss.cpu().item() / labels.shape[0]
                    
                    train_corr += torch.sum(torch.argmax(f.softmax(y_pred, dim=-1), dim=-1) == labels).cpu().item()
                    n_train += labels.shape[0]
                    
                    
                    
            with torch.no_grad():
                train_loss.append(train_l)
                
                train_acc.append(train_corr/n_train)
                
                for vfeatures, vlabels in valid_dataloader:
                    #(B, C, nW, nF)
                    vfeatures = vfeatures.to(self.device)
                    #(B, )
                    vlabels = vlabels.to(self.device)
                                        
                    vy_pred = self(vfeatures)
                    
                    valid_corr += torch.sum(torch.argmax(f.softmax(vy_pred, dim=-1), dim=-1) == vlabels).cpu().item()
                    n_valid += vlabels.shape[0]
                    
                valid_acc.append(valid_corr/n_valid)
                
        return train_loss, train_acc, valid_acc
                
                
                
            
            
            
        