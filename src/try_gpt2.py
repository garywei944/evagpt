from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("gpt2")

print("Config:", config)
print()
