
```markdown
# 🧬 AI Model Evolution Lab

**Experimenting with evolutionary "mutations" of language models via parameter-efficient fine-tuning in Google Colab.**

---

## 🚀 Overview

This project builds a practical pipeline to create a family of AI model variants by "mutating" a base LLM using **LoRA fine-tuning (PEFT)** in a realistic, runnable environment like Google Colab.

Moving away from simulation to real training of quantized models on Colab GPUs, this enables experimentation with AI evolution by directed fine-tuning on various datasets.

---

## 📂 Project Structure

```
├── README.md                  # This file
├── colab_notebook.ipynb       # Core Colab notebook for mutation experiments
└── adapters/                  # (Generated) Stores saved LoRA adapters
```

---

## ⚙️ Setup Instructions

1. Clone this repository:
   ```
   git clone https://github.com/your-username/ai-model-evolution-lab.git
   cd ai-model-evolution-lab
   ```

2. Open `colab_notebook.ipynb` in [Google Colab](https://colab.research.google.com).

3. Run the notebook cells sequentially to:
   - Set up the environment
   - Load a quantized base model
   - Define mutation (fine-tuning) function with LoRA
   - Prepare sample dataset
   - Perform mutations to create model variants

---

## 🔬 Usage

- Customize datasets inside the notebook to create mutations tuned for different tasks.
- Save and reload LoRA adapters for inference or further mutations.
- Extend orchestration for batch mutation and evolutionary experiments.

---

## 🛠 Development & Contributions

Welcome contributions! Ideas include  
- Multi-variant orchestration  
- Integrated fitness evaluation  
- Dataset diversity  

---

## ⚠️ Disclaimer

For research, experimentation, and education only. Not production-ready.

---

Have fun with your AI evolution sandbox!
```

***

## 2. colab_notebook.ipynb (Markdown/Code for your notebook)

```markdown
# 🧬 AI Model Evolution Lab - Colab Notebook

## 1. Setup Environment

```
!pip install -q -U transformers accelerate peft bitsandbytes datasets
```

## 2. Check GPU

```
import torch
print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
```

## 3. Load Model & Tokenizer

```
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "HuggingFaceH4/zephyr-7b-beta"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
print("Model loaded successfully!")
```

## 4. Define Mutation Function

```
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
import torch

def mutate_model(base_model, dataset, mutation_name="variant_1"):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

    model_to_mutate = get_peft_model(base_model, peft_config)
    model_to_mutate.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=f"./{mutation_name}_output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_dir=f'./{mutation_name}_logs',
        report_to="none"
    )

    trainer = Trainer(
        model=model_to_mutate,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda d: {
            'input_ids': torch.stack([x['input_ids'] for x in d]),
            'attention_mask': torch.stack([x['attention_mask'] for x in d]),
            'labels': torch.stack([x['input_ids'] for x in d])
        }
    )

    print(f"🧬 Mutating: {mutation_name} ...")
    trainer.train()
    trainer.model.save_pretrained(f"./{mutation_name}_adapter")
    print(f"✅ Mutation complete: {mutation_name}")
    return trainer.model
```

## 5. Prepare Example Dataset

```
from datasets import Dataset

texts = ["The secret of life is mutation."] * 100
dataset = Dataset.from_dict({"text": texts})

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
```

## 6. Run Mutation

```
variant_1 = mutate_model(model, tokenized_dataset, mutation_name="philosophical_variant")
```

## 7. Next Steps

- Load saved adapters for inference  
- Experiment with other datasets  
- Create orchestration for multi-variant evolution  
```

***

## 3. How to Clone and Use

From your terminal or laptop (including on your bike!):

```bash
git clone https://github.com/your-username/ai-model-evolution-lab.git
cd ai-model-evolution-lab
```

Then open `colab_notebook.ipynb` in Google Colab and run through the cells.

***

If you want, I can prepare the exact `.ipynb` JSON file ready to upload to GitHub, plus the `README.md`.

Would you like me to generate those GitHub-ready files?

