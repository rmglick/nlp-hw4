import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')
    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"], help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"], help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=1, help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=10, help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=3, help="If validation performance stops improving, how many epochs should we wait before stopping?")
    parser.add_argument('--use_wandb', action='store_true', help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment', help="How should we name this experiment?")
    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_loss = float('inf')  # Track best loss instead of F1
    epochs_since_improvement = 0
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    
    args.checkpoint_dir = checkpoint_dir
    gt_sql_path = 'data/dev.sql'
    gt_record_path = 'records/ground_truth_dev.pkl'
    model_sql_path = 'results/t5_ft_dev.sql'
    model_record_path = 'records/t5_ft_dev.pkl'

    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
            gt_sql_path, model_sql_path,
            gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss': tr_loss,
                'dev/loss': eval_loss,
                'dev/record_f1': record_f1,
                'dev/record_em': record_em,
                'dev/sql_em': sql_em,
                'dev/error_rate': error_rate,
            }
            wandb.log(result_dict, step=epoch)

        save_model(checkpoint_dir, model, best=False)
        
        # Use LOSS for early stopping (not fake F1)
        if eval_loss < best_loss:
            best_loss = eval_loss
            epochs_since_improvement = 0
            save_model(checkpoint_dir, model, best=True)
            print(f"✓ New best model! Dev loss: {eval_loss:.4f}")
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= args.patience_epochs:
            print("Early stopping due to validation stagnation.")
            break


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        
        encoder_input = batch[0].to(DEVICE)
        encoder_mask = batch[1].to(DEVICE)
        labels = batch[2].to(DEVICE)
        
        outputs = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        total_tokens += 1
    
    return total_loss / max(total_tokens, 1)

def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    model.eval()
    total_loss = 0
    num_batches = 0
    generated_sqls = []
    num_syntax_errors = 0
    
    for batch in tqdm(dev_loader, desc="Evaluating"):
        encoder_input, encoder_mask, labels = batch
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        labels = labels.to(DEVICE)
        
        with torch.no_grad():
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
            
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=256,
                num_beams=1,
                do_sample=False
            )
            
            batch_sqls = []
            for ids in generated_ids:
                decoded = model.tokenizer.decode(ids, skip_special_tokens=True)
                decoded = decoded.replace("<pad>", "").replace("</s>", "").strip()
                batch_sqls.append(decoded)
            
            generated_sqls.extend(batch_sqls)
            
            for sql in batch_sqls:
                sql_upper = sql.strip().upper()
                if not sql.strip() or not (sql_upper.startswith('SELECT') or 
                        sql_upper.startswith('INSERT') or 
                        sql_upper.startswith('UPDATE') or 
                        sql_upper.startswith('DELETE') or
                        sql_upper.startswith('CREATE') or
                        sql_upper.startswith('DROP')):
                    num_syntax_errors += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    syntax_error_rate = num_syntax_errors / max(len(generated_sqls), 1)
    
    # ONLY SAVE SQL FILE (skip .pkl generation during training)
    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    with open(model_sql_path, 'w') as f:
        for sql in generated_sqls:
            f.write(sql.strip() + '\n')
    print(f"✓ Saved {len(generated_sqls)} SQL queries to {model_sql_path}")
    
    # Use ground truth records for metrics
    sql_em, record_em, record_f1, _ = compute_metrics(
        gt_path=gt_sql_pth,
        model_path=model_sql_path,
        gt_query_records=gt_record_path,
        model_query_records=gt_record_path
    )
    
    return avg_loss, record_f1, record_em, sql_em, syntax_error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    Runs inference on the test set with improved generation parameters for final submission.
    '''
    model.eval()

    # Confirm model and tokenizer classes and config
    print("=== [TEST] MODEL & TOKENIZER DEBUG ===")
    print("Model class:", type(model))
    print("Tokenizer class:", type(model.tokenizer))
    print("Tokenizer vocab size:", len(model.tokenizer))
    print("Checkpoint device:", next(model.parameters()).device)
    print("="*60)
    
    generated_sqls = []
    
    print("\n" + "="*60)
    print("Starting test set inference with beam search...")
    print("="*60)

    batch_printed = False
    for batch_id, batch in enumerate(tqdm(test_loader, desc="Test Inference")):
        encoder_input, encoder_mask = batch
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        
        if not batch_printed:
            print("Sample batch [first 10 ids]:", encoder_input[0][:10].cpu().tolist())
            print("Decoded input:", model.tokenizer.decode(encoder_input[0], skip_special_tokens=True))
            batch_printed = True
        
        with torch.no_grad():
            generatedids = model.generate(
                input_ids=encoder_input,
                attention_mask=encodermask,
                max_length=256,
                num_beams=1,
                do_sample=False
            )           

            
            batch_sqls = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
            if batch_id == 0:
                print("First decoded SQL (test):", batch_sqls[0])
            generated_sqls.extend(batch_sqls)
    
    print(f"\n✓ Generated {len(generated_sqls)} SQL queries")
    print("Now executing queries and saving records (this may take 5-10 minutes)...")
    
    save_queries_and_records(generated_sqls, model_sql_path, model_record_path)
    
    print(f"✓ Saved SQL queries to: {model_sql_path}")
    print(f"✓ Saved database records to: {model_record_path}")
    print("="*60)


def main():
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    
    # PHASE 1: TRAIN
    print("\n========== Training ==========")
    model, tokenizer = initialize_model(args)
    model.tokenizer = tokenizer
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # PHASE 2: FINAL DEV EVAL
    print("\n" + "="*60)
    print("Final evaluation on dev set with best model...")
    print("="*60)
    model, tokenizer = load_model_from_checkpoint(args, best=True)
    model.tokenizer = tokenizer
    print("[DEV EVAL] Model and tokenizer classes:", type(model), type(tokenizer))
    model.eval()
    generated_sqls = []
    for batch in tqdm(dev_loader, desc="Final Dev Eval"):
        encoder_input, encoder_mask, labels = batch
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=256,
                num_beams=1,
                do_sample=False
            )
            batch_sqls = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
            generated_sqls.extend(batch_sqls)
    gt_sql_path = 'data/dev.sql'
    gt_record_path = 'records/ground_truth_dev.pkl'
    model_sql_path = 'results/t5_ft_dev.sql'
    model_record_path = 'records/t5_ft_dev.pkl'
    print("Generating dev set records (this may take a few minutes)...")
    save_queries_and_records(generated_sqls, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, _ = compute_metrics(
        gt_path=gt_sql_path,
        model_path=model_sql_path,
        gt_query_records=gt_record_path,
        model_query_records=model_record_path)
    print(f"\n{'='*60}")
    print(f"Dev set results: Loss: N/A, Record F1: {record_f1:.4f}, Record EM: {record_em:.4f}, SQL EM: {sql_em:.4f}")
    print(f"{'='*60}\n")

    # PHASE 3: FINAL TEST INFERENCE (NEW FRESH MODEL LOAD!)
    print("\n========== Test Inference ==========")
    model, tokenizer = load_model_from_checkpoint(args, best=True)
    model.tokenizer = tokenizer
    print("[TEST EVAL] Model and tokenizer classes:", type(model), type(tokenizer))
    model_sql_path = 'results/t5_ft_test.sql'
    model_record_path = 'records/t5_ft_test.pkl'
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

    print("\n" + "="*50)
    print("Training and inference complete!")
    print(f"Best dev F1: {record_f1:.4f}")
    print(f"Test predictions saved to: {model_sql_path}")
    print("="*50)

    import sys
    sys.exit(0)

if __name__ == "__main__":
    main()
