from dataclasses import dataclass

LIST_LABEL = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise',
]

@dataclass
class HF_DataConfig():
    """
    Data Settings
    """
    root_path: str = "/mnt/mount-hdd/dataset/KEMDy20_v1_2_processed/txt_wav"
    csv_path: str = "/mnt/mount-hdd/dataset/KEMDy20_v1_2_processed/annotation.csv"
    normalized: bool = True
    remove_non_text: bool = True
    # return_text: bool = False
    
@dataclass
class HF_TrainConfig():
    lr: float = 1e-5
    label_name: str = 'emotion'
    checkpoint_path: str = './models_zoo/checkpoint/'
    log_dir: str = './models_zoo/tensorboard/'
    using_model: str = 'both'
    batch_size: int = 4
    text_encoder: str = "klue/roberta-base"
    audio_processor: str = "w11wo/wav2vec2-xls-r-300m-korean"