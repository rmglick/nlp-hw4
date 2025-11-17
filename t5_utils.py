import os
import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Config, T5TokenizerFast
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb.run.name = args.run_name

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    from transformers import T5TokenizerFast
    
    if args.finetune:
        # LOAD PRETRAINED WEIGHTS
        print("Loading pretrained T5-small model...")
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        tokenizer = T5TokenizerFast.from_pretrained('t5-small')
        print("✓ Pretrained model loaded successfully")
    else:
        # Initialize from config (random weights)
        print("Initializing T5-small model from config (random weights)...")
        config = T5Config.from_pretrained('t5-small')
        model = T5ForConditionalGeneration(config)
        tokenizer = T5TokenizerFast.from_pretrained('t5-small')
        print("✓ Model initialized from scratch")
    
    model = model.to(DEVICE)
    return model, tokenizer


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    mkdir(checkpoint_dir)
    filename = os.path.join(checkpoint_dir, "model_best.pt" if best else "model_latest.pt")
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model_from_checkpoint(args, best):
    model, tokenizer = initialize_model(args)
    filename = os.path.join(args.checkpoint_dir, "model_best.pt" if best else "model_latest.pt")
    if os.path.exists(filename):
        state_dict = torch.load(filename, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Model loaded from {filename}")
    else:
        print(f"Warning: Checkpoint {filename} not found")
    model = model.to(DEVICE)
    return model, tokenizer


def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]
    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        raise NotImplementedError("Only AdamW optimizer is supported.")
    return optimizer

def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs
    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError("Unsupported scheduler type.")

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result
