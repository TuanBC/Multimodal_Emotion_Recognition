from dataclasses import dataclass

LIST_LABEL = [
    'Happy',
    'Relief',
    'Surprise',
    'Neutral',
    'Worry',
    'Sadness',
    'Angry',
]

LIST_EMOTION = LIST_LABEL

@dataclass
class HF_DataConfig():
    """
    Data Settings
    """
    root_path: str = "/data/dataset/TableManager/audio/new_files4279"
    csv_path: str = "/data/nsml_backup/block-storage/Multimodal_Emotion_Recognition/data/short_audio_text_231105.csv"
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