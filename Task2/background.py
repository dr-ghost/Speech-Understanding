import os
import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import random
import torchaudio.transforms as T
from tqdm import tqdm

AUDIO_BASE_PATH = "../UrbanSound8K/audio"

def rectangular_window(audio : torch.Tensor, window_size : int, num_windows : int):
    """
    audio: (num_channels, num_frames)
    """
    num_channels, num_frames = audio.shape
    
    windows = torch.zeros(num_channels, num_windows, window_size)
    
    hop_size = (num_frames - window_size) //(num_windows - 1)
    
    for i in range(num_windows - 1):
        windows[:, i] = audio[:, i*hop_size:i*hop_size+window_size]
    
    windows[:, num_windows - 1] = audio[:, -window_size:]
    
    return windows

def hamming_window(audio : torch.Tensor, window_size : int, num_windows : int):
    windows = rectangular_window(audio, window_size, num_windows)
    
    hamming_win = torch.signal.windows.hamming(window_size)
    
    return windows * hamming_win

def hanning_window(audio : torch.Tensor, window_size : int, num_windows : int):
    windows = rectangular_window(audio, window_size, num_windows)
    
    hamming_win = torch.signal.windows.hann(window_size)
    
    return windows * hamming_win

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    
def plot_spectrogram(file_path, genre_title, n_fft=1024, hop_length=None):
    audio, sample_rate = torchaudio.load(file_path)
    
    if audio.shape[0] > 1:
        audio = audio[0, :]
        
    audio = audio[:4_00_000]

    spectrogram_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
    
    spec = spectrogram_transform(audio)
        
    db_transform = T.AmplitudeToDB(stype='power', top_db=80)
    spec_db = db_transform(spec)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(spec_db.numpy(), origin='lower', aspect='auto', 
               extent=[0, spec_db.shape[1] * (hop_length or (n_fft // 2)) / sample_rate, 0, sample_rate / 2])
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram - {genre_title}")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()
    
def plot_spectrogram_from_scratch(file_path, genre_title, n_fft=1024, hop_length=None):
    audio, sample_rate = torchaudio.load(file_path)
    
    # if audio.shape[0] > 1:
    #     audio = audio[0, :]
        
    audio = audio[:4_00_000]

    spec = torch.abs(torch.fft.rfft(hanning_window(audio, 1_000, 400), dim=-1))
    
    if spec.shape[0] > 1:
        spec = spec[0, ...]
    
    db_transform = T.AmplitudeToDB(stype='power', top_db=80)
    spec_db = db_transform(spec)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(spec_db.numpy(), origin='lower', aspect='auto', 
               extent=[0, spec_db.shape[1] * (hop_length or (n_fft // 2)) / sample_rate, 0, sample_rate / 2])
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram - {genre_title}")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()
    
def plot_spectorgram_mediate(features):
    
    spec = torch.abs(torch.fft.rfft(features, dim=-1))
    
    if spec.shape[0] > 1:
        spec = spec[0, ...]
    
    db_transform = T.AmplitudeToDB(stype='power', top_db=80)
    spec_db = db_transform(spec)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(spec_db.numpy(), origin='lower', aspect='auto', 
               extent=[0, spec_db.shape[1] * 320 / 8000, 0, 8000 / 2])
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

class UrbanSound8KDataset(Dataset):
    def __init__(self, base_path, window_function):
        self.base_path = base_path
        self.file_path_lst = []
        
        for root, sub_dir, files in os.walk(base_path):
            for file in files:
                if file.endswith(".wav"):
                    self.file_path_lst.append(os.path.join(root, file)) 
           
        if window_function.__name__ ==  "rectangular_window":     
            self.num_windows = 50
            self.window_size = 300
        else:
            self.num_windows = 50
            self.window_size = 320
        
        self.window_function = window_function
    
    def __len__(self):
        return len(self.file_path_lst)
    
    def __getitem__(self, idx):
        waveform, _ = torchaudio.load(self.file_path_lst[idx], num_frames=80_000)
        
        waveform = torchaudio.functional.resample(waveform, 44100, 8000)
        
        ln1 = waveform.shape[1]
        
        label = int(self.file_path_lst[idx].split("/")[-1].split("-")[1])
        
        #print(waveform.shape[1]) 
        
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
            
        windowed_features = self.window_function(waveform, self.window_size, self.num_windows)
        
        return windowed_features, label

if __name__ == "__main__":
    LAUDIO_BASE_PATH = "UrbanSound8K/audio"
    
    ubsd = UrbanSound8KDataset(LAUDIO_BASE_PATH, rectangular_window)
    

    