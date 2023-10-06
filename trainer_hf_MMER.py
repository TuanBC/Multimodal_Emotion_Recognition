from config import HF_DataConfig, HF_TrainConfig
import os
from models.pl_model_hf import *
import pytorch_lightning as pl
from dataset_hf import *
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', required=True, type=str)
    p.add_argument('--batch_size', required=True, type=int)
    p.add_argument('--accumulate_grad', required=True, type=int, default=1)
    p.add_argument('--clip_length', type=int, default=25)
    p.add_argument('--loss', type=str, default="ce")
    config = p.parse_args()

    return config


def main(args):
    pl.seed_everything(42)
    num_gpu = torch.cuda.device_count()
    data_config = HF_DataConfig()
    train_config = HF_TrainConfig(
        batch_size=args.batch_size,
    )

    train_config.num_classes = len(LIST_LABEL)

    # Load train and validation data
    csv = pd.read_csv(data_config.csv_path)
    csv = csv.drop_duplicates(subset=['segment_id'], ignore_index=True)
    
    csv['wav_length'] = csv['wav_end'] - csv['wav_start']
    csv = csv.query("wav_length <= %d"%args.clip_length)
    # dev, _ = train_test_split(csv, test_size=0.2, random_state=1004)
    # train, val = train_test_split(dev, test_size=0.1, random_state=1004)

    train, val = train_test_split(csv, test_size=0.2, random_state=42)
    
    text_tokenizer = AutoTokenizer.from_pretrained(train_config.text_encoder)
    audio_processor = AutoProcessor.from_pretrained(train_config.audio_processor)
    
    # train_dataset = multimodal_dataset(train, data_config)
    # val_dataset = multimodal_dataset(val, data_config)

    train_dataset = multimodal_dataset_auxiliary_2(train, data_config)
    val_dataset = multimodal_dataset(val, data_config)

    
    train_loader = DataLoader(train_dataset, train_config.batch_size, num_workers=4,
                              collate_fn=multimodal_collator(text_tokenizer, audio_processor), pin_memory=True,
                              shuffle=True, drop_last=True)
    
    val_loader = DataLoader(val_dataset, train_config.batch_size, num_workers=4,
                              collate_fn=multimodal_collator(text_tokenizer, audio_processor), pin_memory=True, 
                              drop_last=True, shuffle=False)
    
    # Load model and configuration.
    
    if args.loss == "ce":
        model = PL_model_MMER(train_config)
    elif args.loss == "cs_and_ce":
        model = PL_model_MMER_multiloss(train_config)
    else:
        raise "WrongLossName"
    setattr(model, 'train_dataloader', lambda: train_loader)
    setattr(model, 'val_dataloader', lambda: val_loader)
        
    checkpoint_callback = plc.ModelCheckpoint(
        monitor="val_accuracy",
        dirpath=os.path.join(train_config.checkpoint_path, args.exp_name),
        filename="{epoch:02d}-{val_loss:.5f}",
        save_top_k=1,
        mode="max",
        save_last=True,
    )

    # early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=False, mode="min")
    tb_logger = TensorBoardLogger(train_config.log_dir, name=args.exp_name)
    wandb_logger = WandbLogger(name=args.exp_name, project="multimodal", save_dir=train_config.log_dir)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpu,
        strategy=DDPStrategy(find_unused_parameters=True),
        # strategy=FSDPStrategy(),
        max_epochs=15,
        # checkpoint_callback=True,
        callbacks=[checkpoint_callback],
        precision='16-mixed',
        profiler="simple",
        accumulate_grad_batches=args.accumulate_grad,
        logger=[tb_logger, wandb_logger],
        gradient_clip_val=2,
        # load from checkpoint
        # resume_from_checkpoint=os.path.join(train_config.checkpoint_path, args.exp_name, "last.ckpt"),
        )
    
    trainer.fit(model, ckpt_path=os.path.join(train_config.checkpoint_path, args.exp_name, "last.ckpt"))
    
if __name__ == '__main__':
    args = define_argparser()
    main(args)