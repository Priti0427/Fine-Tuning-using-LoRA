# Fine-Tuning-using-Lora

Fine-Tuning of a model involves training a pre-trained model like GPT, LLaMA, Qwen etc. for more specific tasks efficiently. Fine-tuning of LLMs is computationally expensive as it involves billions of parameters. LoRA(Low Rank Adaptation) is a Parameter Efficient Fine Tuning technique(PEFT), which reduces the computational requirements. It involves addition of small set of task-specific weights while keeping most of the base model’s parameters frozen.

LoRA learns low-rank matrices that approximate the changes needed to fine-tune the model.This reduces the number of trainable parameters, making the fine-tuning process cheaper, faster, and more memory efficient.
