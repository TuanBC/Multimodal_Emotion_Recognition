from torch.utils.data import Dataset
import torchaudio
import torch
import os
import re
from utils import *
import random
import librosa

class multimodal_dataset(Dataset):
    def __init__(self, csv, config):
        self.csv = csv
        self.root_path = config.root_path
        self.remove_non_text = config.remove_non_text
        
    def __len__(self):
        return len(self.csv)
    
    def _load_wav(self, wav_path, duration=None, offset=0):
        wav, sr = librosa.load(wav_path, sr=None, duration=duration, offset=offset)

        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        return wav
    
    def _load_txt(self, txt_path):
        with open(txt_path, 'r') as f:
            txt = f.readlines()
        assert len(txt) == 1,  'Text line Must be 1'
        
        txt = txt[0][:-1]
        if self.remove_non_text:
            txt = re.sub('[a-zA-Z]/', '', txt).strip()
        return txt
    
    def _load_data(self, idx):
        wav_path = os.path.join(self.root_path, self.csv['segment_id'].iloc[idx]+'.wav')
        # txt_path = os.path.join(self.root_path, self.csv['segment_id'].iloc[idx]+'.txt')
        
        start = self.csv['start'].iloc[idx]
        end = self.csv['end'].iloc[idx]

        wav = self._load_wav(wav_path)
        wav = wav[int(start*16000):int(end*16000)]

        if wav.shape[-1] == 0:
            print(wav_path)

        # txt = self._load_txt(txt_path)
        txt = self.csv['text'].iloc[idx]
        
        emotion = self.csv['emotion'].iloc[idx]
        
        sample = {
            'text' : txt,
            'wav' : wav,
            'emotion': emotion2int[emotion],
        }
        return sample
    
    def __getitem__(self, idx):
        sample = self._load_data(idx)
        return sample
    

class multimodal_dataset_auxiliary_2(multimodal_dataset):
    def __init__(self, csv, config):
        self.csv = csv
        self.root_path = config.root_path
        self.remove_non_text = config.remove_non_text
        self.dict_csv_emotion = {}
        for emotion in LIST_LABEL:
            self.dict_csv_emotion[emotion] = self.csv[self.csv['emotion'] == emotion].reset_index(drop=True)

    def _load_data(self, idx):
        emotion = self.csv['emotion'].iloc[idx]

        # get the index that can be iloc

        csv_same_emotion = self.dict_csv_emotion[emotion]
        idx_random = random.choice(csv_same_emotion.index)

        if random.random() > 0.5:
            # random pick wav_path of the same emotion, but other idx
            wav_path = os.path.join(self.root_path, csv_same_emotion['segment_id'].iloc[idx_random]+'.wav')
            # txt_path = os.path.join(self.root_path, self.csv['segment_id'].iloc[idx]+'.txt')
            txt = self.csv['text'].iloc[idx]
        else:
            # random pick txt_path of the same emotion, but other idx
            wav_path = os.path.join(self.root_path, self.csv['segment_id'].iloc[idx]+'.wav')
            # txt_path = os.path.join(self.root_path, csv_same_emotion['segment_id'].iloc[idx_random]+'.txt')
            txt = csv_same_emotion['text'].iloc[idx_random]
        
        start = self.csv['start'].iloc[idx]
        end = self.csv['end'].iloc[idx]
        wav = self._load_wav(wav_path, duration=end-start, offset=start)
        # wav = self._load_wav(wav_path, duration=0)
        
        sample = {
            'text' : txt,
            'wav' : wav,
            'emotion': emotion2int[emotion],
        }
        return sample


class multimodal_dataset_inference(Dataset):
    def __init__(self, csv, config):
        self.csv = csv
        self.root_path = config.root_path
        self.remove_non_text = config.remove_non_text
        
    def __len__(self):
        return len(self.csv)
    
    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            resample = torchaudio.transforms.Resample(sr, 16000)
            wav = resample(wav)

        wav = wav.squeeze().numpy()
        return wav
    
    def _load_data(self, idx):
        wav_path = os.path.join(self.root_path, self.csv['segment_id'].iloc[idx]+'.wav')
        
        wav = self._load_wav(wav_path)
        txt = self.csv['text'].iloc[idx]        

        sample = {
            'text' : txt,
            'wav' : wav
        }
        return sample
    
    def __getitem__(self, idx):
        sample = self._load_data(idx)
        return sample


class multimodal_collator():
    def __init__(self, text_tokenizer, audio_processor, return_text=False, max_length=512):
        self.text_tokenizer = text_tokenizer
        self.audio_processor = audio_processor
        self.return_text = return_text
        self.max_length = max_length
        
    def __call__(self, batch):
        text = [d['text'] for d in batch]
        wav = [d['wav'] for d in batch]
        emotion = [d['emotion'] for d in batch]

        text_inputs = self.text_tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.max_length
        )
        
        audio_inputs = self.audio_processor(
            wav,
            sampling_rate=16000, 
            padding=True, 
            return_tensors='pt'
        )
        
        labels = {
            "emotion" : torch.LongTensor(emotion),
        }
        if self.return_text:
            labels['text'] = text
        return text_inputs, audio_inputs, labels
    

class multimodal_collator_inference():
    def __init__(self, text_tokenizer, audio_processor, return_text=False, max_length=512):
        self.text_tokenizer = text_tokenizer
        self.audio_processor = audio_processor
        self.return_text = return_text
        self.max_length = max_length
        
    def __call__(self, batch):
        text = [d['text'] for d in batch]
        wav = [d['wav'] for d in batch]
        
        text_inputs = self.text_tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.max_length
        )
        
        audio_inputs = self.audio_processor(
            wav,
            sampling_rate=16000, 
            padding=True, 
            return_tensors='pt'
        )
        
        return text_inputs, audio_inputs, None