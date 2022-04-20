from config import HF_DataConfig, HF_TrainConfig
import os
from models.pl_model_multiout_hf import PL_model
import pytorch_lightning as pl
from dataset_hf import *
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Wav2Vec2Processor
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', required=True, type=str)
    p.add_argument('--using_model', required=True, type=str)
    p.add_argument('--batch_size', required=True, type=int)
    p.add_argument('--accumulate_grad', required=True, type=int, default=1)
    p.add_argument('--clip_length', type=int, default=25)
    config = p.parse_args()

    return config


def main(args):
    pl.seed_everything(42)
    data_config = HF_DataConfig()
    train_config = HF_TrainConfig(
        batch_size=args.batch_size,
        using_model=args.using_model
    )

    # Load train and validation data
    csv = pd.read_csv(data_config.csv_path)
    csv = csv.drop_duplicates(subset=['segment_id'], ignore_index=True)
    
    csv['wav_length'] = csv['wav_end'] - csv['wav_start']
    csv = csv.query("wav_length <= %d"%args.clip_length)
    dev, _ = train_test_split(csv, test_size=0.1, random_state=1004)
    train, val = train_test_split(dev, test_size=0.1, random_state=1004)
    
    text_tokenizer = AutoTokenizer.from_pretrained(train_config.text_encoder)
    audio_processor = Wav2Vec2Processor.from_pretrained(train_config.audio_processor)
    
    train_dataset = multimodal_dataset(train, data_config)
    val_dataset = multimodal_dataset(val, data_config)
    
    train_loader = DataLoader(train_dataset, train_config.batch_size, num_workers=8,
                              collate_fn=multimodal_collator(text_tokenizer, audio_processor), pin_memory=True,
                              shuffle=True, drop_last=True)
    
    val_loader = DataLoader(val_dataset, train_config.batch_size, num_workers=8,
                              collate_fn=multimodal_collator(text_tokenizer, audio_processor), pin_memory=True, 
                              drop_last=True, shuffle=False)
        
    # Load model and configuration.
    
    model = PL_model(data_config, train_config)
    setattr(model, 'train_dataloader', lambda: train_loader)
    setattr(model, 'val_dataloader', lambda: val_loader)
        
    checkpoint_callback = plc.ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(train_config.checkpoint_path, args.exp_name),
        filename="{epoch:02d}-{val_loss:.5f}",
        save_top_k=2,
        mode="min",
    )

    # early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=False, mode="min")
    logger = TensorBoardLogger(train_config.log_dir, name=args.exp_name)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=3,
        # strategy="ddp",
        max_epochs=20,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
        precision=16,
        amp_backend="native",
        profiler="simple",
        accumulate_grad_batches=args.accumulate_grad,
        logger=logger,
        gradient_clip_val=2,
        plugins=DDPPlugin(find_unused_parameters=False)
        )
    
    trainer.fit(model)
    
if __name__ == '__main__':
    args = define_argparser()
    main(args)