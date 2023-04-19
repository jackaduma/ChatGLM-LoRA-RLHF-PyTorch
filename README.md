# **ChatGLM-LoRA-RLHF-PyTorch**

---
## **Table of Contents**
- [**ChatGLM-LoRA-RLHF-PyTorch**](#chatglm-lora-rlhf-pytorch)
  - [**Table of Contents**](#table-of-contents)
  - [**Environment Setup**](#environment-setup)
  - [**Todo List**](#todo-list)
  - [**Run**](#run)
    - [**Data Process**](#data-process)
    - [**Supervised Finetune**](#supervised-finetune)
    - [**Merge PEFT adapter into Model**](#merge-peft-adapter-into-model)
  - [**Topics**](#topics)
  - [**Reference**](#reference)
---

## **Environment Setup**
```
穷人卡：2080Ti 12G
torch==2.0.0
cuda==11.8
```

---
## **Todo List**

- [x] SFT: Supervised Finetune
- [x] Merge Adapter into Model
- [ ] RLHF
  - [ ] train reward model
  - [ ] tuning with RL

## **Run**
---

### **Data Process**

转化alpaca数据集为jsonl

```bash
python cover_alpaca2jsonl.py --data_path data/alpaca_data.json --save_path data/alpaca_data.jsonl
```

tokenization

```bash
python tokenize_dataset_rows.py --jsonl_path data/alpaca_data.jsonl --save_path data/alpaca --max_seq_length 200 --skip_overlength True
```

### **Supervised Finetune**

must use latest peft version
```
pip uninstall peft -y
pip install git+https://github.com/huggingface/peft.git  # 最新版本 >=0.3.0.dev0
```

```bash
python supervised_finetune.py --dataset_path data/alpaca --lora_rank 8 --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --save_steps 200 --save_total_limit 3  --learning_rate 1e-4 --fp16 --remove_unused_columns false --logging_steps 10 --output_dir output
```

### **Merge PEFT adapter into Model**

```bash
pip uninstall peft -y
pip install peft==0.2.0  # 0.3.0.dev0 raise many errors
python merge_peft_adapter.py
```

---

## **Topics**
1. PEFT的版本，目前从git上安装的是 0.3.0.dev0 版本，在merge_peft_adapter的时候有问题，需要切换到peft==0.2.0 (0.3.0.dev0 没有 _get_submodules()这个函数)
2. 因为huggingface的transformer暂时不支持ChatGLM的封装接口，需要自己从ChatGLM的hub上下载代码放到本地目录 models 下面，供后续使用
3. 同样，ChatGLM的model代码是自己的，和huggingface没合并，所以在调用加载的时候，都主要加上参数 trust_remote_code=True
4. 训练 Reward Model 需要执行 SeqCLS 这个Task： huggingface 的 transformer 提供 "AutoModelForSequenceClassification" 这个类。但是 ChatGLM 只有 "ChatGLMForConditionalGeneration" 这个类。所以暂时没法训练 Reward model，等ChatGLM自己放出代码，或者huggingface 集成 ChatGLM吧

## **Reference**
data preprocess: [cover_alpaca2jsonl.py](./cover_alpaca2jsonl.py) 和 [tokenize_dataset_rows.py](./tokenize_dataset_rows.py) 来自项目 [ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning)

requirements 主要是按照 [alpaca-lora](https://github.com/tloen/alpaca-lora) 来配环境。

* [https://github.com/tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)
* [https://github.com/mymusise/ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning)
* [https://github.com/lvwerra/trl](https://github.com/lvwerra/trl)
* [https://github.com/jasonvanf/llama-trl](https://github.com/jasonvanf/llama-trl)