"""
Train Model
===========================
Trains the Qwen model using settings from config.py.

Usage:
    python scripts/train_qwen.py
"""
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CLEANED_OUTPUT,
    MODEL_OUTPUT_DIR,
    BASE_MODEL_ID,
    MAX_LENGTH,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    BATCH_SIZE,
    EPOCHS,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    USE_BF16,
    USE_4BIT,
    QUANT_TYPE,
    LOG_DIR
)


def print_trainable_parameters(model):
    """Utility function to display how many parameters are being trained."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params} || All params: {all_params} || "
          f"Trainable%: {100 * trainable_params / all_params:.2f}%")


def main(
    train_dataset_path=CLEANED_OUTPUT,
    output_dir=MODEL_OUTPUT_DIR,
    eval_dataset_path=None,
    base_model_id=BASE_MODEL_ID,
):
    # ------------------------------
    # Quantization Configuration
    # ------------------------------
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=QUANT_TYPE,
            bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        )

    # ------------------------------
    # Load Dataset
    # ------------------------------
    print("📚 Loading dataset...")
    train_dataset = load_dataset("json", data_files=str(train_dataset_path), split="train")
    eval_dataset = load_dataset("json", data_files=str(eval_dataset_path), split="train") if eval_dataset_path else None

    # ------------------------------
    # Load Model & Tokenizer
    # ------------------------------
    print(f"🧠 Loading model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        trust_remote_code=True,
    )

    # Qwen uses <|endoftext|> as pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------
    # Tokenization Function
    # ------------------------------
    def tokenize_function(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )
        result["labels"] = [
            -100 if token == tokenizer.pad_token_id else token
            for token in result["input_ids"]
        ]
        return result

    print("🔤 Tokenizing dataset...")
    tokenized_train_dataset = train_dataset.map(tokenize_function)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function) if eval_dataset else None

    # ------------------------------
    # Prepare for LoRA Fine-tuning
    # ------------------------------
    print("⚙️ Preparing model for k-bit training...")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        lora_dropout=LORA_DROPOUT,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # ------------------------------
    # Trainer Setup
    # ------------------------------
    print("🚀 Starting training...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        warmup_steps=50,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        bf16=USE_BF16,
        optim="paged_adamw_8bit",
        logging_steps=50,
        logging_dir=str(LOG_DIR),
        save_safetensors=True,
        save_strategy="epoch" if eval_dataset_path else "no",
        report_to="none",
        max_grad_norm=0.3,
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Disable caching for training
    model.config.use_cache = False

    # ------------------------------
    # Train & Save
    # ------------------------------
    trainer.train()
    trainer.save_model()
    print(f"✅ Fine-tuning complete! Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
