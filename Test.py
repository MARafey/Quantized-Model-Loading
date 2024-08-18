import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import os
from PyPDF2 import PdfReader
from peft import LoraConfig, get_peft_model, TaskType
from transformers import DataCollatorForLanguageModeling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Step 1: Prepare the data
def pdf_to_text(pdf_dir):
    texts = []
    for file in os.listdir(pdf_dir):
        if file.endswith('.pdf'):
            reader = PdfReader(os.path.join(pdf_dir, file))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            texts.append(text)
    return texts

pdf_dir = "PDFS"
texts = pdf_to_text(pdf_dir)

# Save texts to a file
with open("dataset.txt", "w", encoding="utf-8") as f:
    for text in texts:
        f.write(text + "\n")

# Step 2: Load the model and tokenizer
model_name = "unsloth/mistral-7b-bnb-4bit"  # Choose the appropriate size
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 3: Prepare the dataset
dataset = load_dataset("text", data_files="dataset.txt")
tokenized_dataset = dataset.map(
    lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512),
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# Step 4: Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)
model.to(device)

# Step 5: Set up the trainer
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=1e-4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Step 6: Train the model
trainer.train()

# Step 7: Save the model
trainer.save_model("./finetuned_llama2")

# Optionally, you can save the adapter separately
model.save_pretrained("./lora_adapter")