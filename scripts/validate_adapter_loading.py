#!/usr/bin/env python3.11
"""
Validation script to prove that the custom adapter is actually being loaded and used.
"""
import os
import sys
from pathlib import Path
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import hashlib

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Set CUDA device to 0 (the available A100 GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_base_model_only(base_model_name: str):
    """Load only the base model without any adapter."""
    print(f"Loading base model only: {base_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        use_fast=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )
    
    model.eval()
    return model, tokenizer

def load_model_with_adapter(base_model_name: str, adapter_path: str):
    """Load model with adapter."""
    print(f"Loading model with adapter from: {adapter_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        use_fast=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )
    
    # Load adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    return model, tokenizer

def inspect_adapter_files(adapter_path: str):
    """Inspect the adapter files to show what's being loaded."""
    print("\n" + "=" * 60)
    print("ADAPTER FILES INSPECTION")
    print("=" * 60)
    
    adapter_path = Path(adapter_path)
    
    # Check adapter config
    config_file = adapter_path / "adapter_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"✅ Adapter config found:")
        print(f"   - Base model: {config.get('base_model_name_or_path', 'N/A')}")
        print(f"   - LoRA rank (r): {config.get('r', 'N/A')}")
        print(f"   - LoRA alpha: {config.get('lora_alpha', 'N/A')}")
        print(f"   - Target modules: {config.get('target_modules', 'N/A')}")
        print(f"   - Task type: {config.get('task_type', 'N/A')}")
    else:
        print("❌ No adapter_config.json found")
    
    # Check adapter weights
    weights_file = adapter_path / "adapter_model.safetensors"
    if not weights_file.exists():
        weights_file = adapter_path / "adapter_model.bin"
    
    if weights_file.exists():
        file_size = weights_file.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ Adapter weights found: {weights_file.name} ({file_size:.1f} MB)")
        
        # Calculate file hash to prove it's unique
        with open(weights_file, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:16]
        print(f"   - File hash (first 16 chars): {file_hash}")
    else:
        print("❌ No adapter weights file found")
    
    # List all files in adapter directory
    print(f"\n📁 All files in {adapter_path}:")
    for file in sorted(adapter_path.rglob("*")):
        if file.is_file():
            size = file.stat().st_size
            print(f"   - {file.relative_to(adapter_path)} ({size} bytes)")

def get_model_parameters_info(model, model_name: str):
    """Get information about model parameters."""
    print(f"\n📊 {model_name} Parameters:")
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Trainable %: {100 * trainable_params / total_params:.4f}%")
    
    # Check for LoRA-specific parameters
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_params.append(name)
    
    if lora_params:
        print(f"   - LoRA parameters found: {len(lora_params)}")
        print(f"   - Sample LoRA params: {lora_params[:3]}...")
    else:
        print("   - No LoRA parameters found")
    
    return total_params, trainable_params, len(lora_params)

def compare_model_outputs(base_model, adapter_model, tokenizer, test_prompt: str):
    """Compare outputs between base model and adapter model."""
    print(f"\n🔍 Comparing outputs for: '{test_prompt[:50]}...'")
    
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
    
    # Generate with base model
    with torch.no_grad():
        base_outputs = base_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    base_response = tokenizer.decode(base_outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    
    # Generate with adapter model
    with torch.no_grad():
        adapter_outputs = adapter_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    adapter_response = tokenizer.decode(adapter_outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    
    print(f"\n📝 Base model response:")
    print(f"   {base_response}")
    print(f"\n🎯 Adapter model response:")
    print(f"   {adapter_response}")
    
    # Check if responses are different
    if base_response != adapter_response:
        print(f"\n✅ RESPONSES ARE DIFFERENT - Adapter is working!")
        return True
    else:
        print(f"\n⚠️  RESPONSES ARE IDENTICAL - Adapter might not be working")
        return False

def validate_adapter_loading():
    """Main validation function."""
    base_model_name = "t-tech/T-lite-it-1.0"
    adapter_path = "src/ml/models/finetuned/t_lite"
    
    print("=" * 60)
    print("ADAPTER LOADING VALIDATION")
    print("=" * 60)
    
    # 1. Inspect adapter files
    inspect_adapter_files(adapter_path)
    
    # 2. Load base model
    print(f"\n🔄 Loading base model...")
    base_model, tokenizer = load_base_model_only(base_model_name)
    base_total, base_trainable, base_lora = get_model_parameters_info(base_model, "Base Model")
    
    # 3. Load model with adapter
    print(f"\n🔄 Loading model with adapter...")
    adapter_model, _ = load_model_with_adapter(base_model_name, adapter_path)
    adapter_total, adapter_trainable, adapter_lora = get_model_parameters_info(adapter_model, "Adapter Model")
    
    # 4. Compare parameter counts
    print(f"\n📊 PARAMETER COMPARISON:")
    print(f"   - Base model LoRA params: {base_lora}")
    print(f"   - Adapter model LoRA params: {adapter_lora}")
    
    if adapter_lora > base_lora:
        print(f"   ✅ Adapter has MORE LoRA parameters - Adapter is loaded!")
    else:
        print(f"   ⚠️  No additional LoRA parameters found")
    
    # 5. Test with fashion-specific prompts
    system_prompt = """Ты — персональный стилист, модный эксперт. Отвечай на вопрос так, как если бы давал совет клиенту. Делай рекомендации точными и стилистически осмысленными. Объясняй, почему тот или иной приём работает. Не пиши очевидного (например, «наденьте топ»). Говори про цвет, настроение, пропорции, фактуры. Учитывай сезон,функциональность и случай, если указано. Пиши по-русски. Ответ должен быть коротким и лаконичным, но содержательным."""
    
    test_question = "Как правильно сочетать кожаную куртку в стиле гранж?"
    test_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{test_question}<|im_end|>\n<|im_start|>assistant\n"
    
    print(f"\n🧪 RESPONSE COMPARISON TEST:")
    responses_different = compare_model_outputs(base_model, adapter_model, tokenizer, test_prompt)
    
    # 6. Final verdict
    print(f"\n" + "=" * 60)
    print("FINAL VALIDATION RESULTS")
    print("=" * 60)
    
    if adapter_lora > 0 and responses_different:
        print("🎉 SUCCESS: Your custom adapter is definitely loaded and working!")
        print("   - LoRA parameters detected in adapter model")
        print("   - Responses differ between base and adapter models")
        print("   - Adapter is modifying model behavior")
    elif adapter_lora > 0:
        print("⚠️  PARTIAL: Adapter is loaded but responses are similar")
        print("   - LoRA parameters detected")
        print("   - But responses are identical (might need different test)")
    else:
        print("❌ ISSUE: Adapter might not be properly loaded")
        print("   - No LoRA parameters detected")
        print("   - Check adapter files and loading process")

if __name__ == "__main__":
    validate_adapter_loading() 