# tune.online.py

```python
import torch
import gc

from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from transformers import Trainer,TrainingArguments
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed
import os,random,sys,json,re
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import TextStreamer
from torch.nn.utils import clip_grad_norm_

random.seed()
def get_truly_random_seed_through_os():
    """
    Usually the best random sample you could get in any programming language is generated through the operating system.
    In Python, you can use the os module.

    source: https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
    """
    RAND_SIZE = 4
    random_data = os.urandom(
        RAND_SIZE
    )  # Return a string of size random bytes suitable for cryptographic use.
    random_seed = int.from_bytes(random_data, byteorder="big")
    return random_seed

seed = get_truly_random_seed_through_os()
set_seed(seed)

json_file = sys.argv[1]

with open(json_file,"r") as jf:
    config = json.load(jf)

cleanse = False
MODEL = config["MODEL"]
TRAIN_FILE = config["TRAIN_FILE"]
OUTPUT_DIR = config["OUTPUT_DIR"]
OVERWRITE = bool(config["OVERWRITE"])
BATCH_SIZE = int(config['BATCH_SIZE'])
EPOCHS = int(config["EPOCHS"])
LRATE = float(config["LRATE"])
STEPS = int(config["STEPS"])
LOAD_4BIT = config["LOAD_4BIT"].lower() == "true"
LOAD_8BIT = config["LOAD_8BIT"].lower() == "true"
FULLTUNE = config["FULLTUNE"].lower() == "true"
OPTIMIZER = config["OPTIM"]
MAXSEQ= int(config["MAXSEQ"])
BF16 = config["BF16"].lower() == "true"
FP16 = config["FP16"].lower() == "true"
if(BF16 == True):
    FP16 = False
if("PERCENT" in config):
    PERCENT = int(config["PERCENT"])
else:
    PERCENT = 100
if("NUM_SAMPLES" in config):
    NUM_SAMPLES = int(config["NUM_SAMPLES"])
else:
    NUM_SAMPLES=0
if("SELECT_OUTPUT" in config):
    SELECT_OUTPUT = config["SELECT_OUTPUT"]
else:
    SELECT_OUTPUT = "output"
if("SHUFFLE" in config):
    os.system("python " + config["SHUFFLE"])


print("-----------------------------------------------------")
print("Configuration")
print("-----------------------------------------------------")
print("MODEL",MODEL)
print("TRAIN_FILE",TRAIN_FILE)
print("OUTPUT_DIR",OUTPUT_DIR)
print("BATCH_SIZE","AUTO")
print("EPOCHS",EPOCHS)
print("LRATE",LRATE)
print("STEPS",STEPS)
print("LOAD_4BIT",LOAD_4BIT)
print("LOAD_8BIT",LOAD_8BIT)
print("FULLTUNE",FULLTUNE)
print("MAXSEQ",MAXSEQ)
print("BF16",BF16)
print("FP16",FP16)
print("-----------------------------------------------------")

# -----------------------
# Load model + tokenizer
# -----------------------


model_name = OUTPUT_DIR
if(os.path.isdir(OUTPUT_DIR)):
    base_model = OUTPUT_DIR
else:
    base_model = MODEL
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if(BF16 == True):
    dtype = "bfloat16"
elif(FP16 == True):
    dtype = "float16"
else:
    dtype = "float32"

model = AutoModelForCausalLM.from_pretrained(
    base_model, 
    device_map="auto",   
    dtype=dtype)
optimizer = AdamW(model.parameters(), lr=LRATE)
model.gradient_checkpointing_enable()

# Custom stopping criteria to stop when the <|endoftext|> token is generated
class StopOnEndOfText(StoppingCriteria):
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if the last token generated is the eos_token_id
        return input_ids[0, -1] == self.eos_token_id

# Create an instance of the stopping criteria with the model's EOS token
eos_token_id = tokenizer.eos_token_id
stopping_criteria = StoppingCriteriaList([StopOnEndOfText(eos_token_id)])
textstreamer = TextStreamer(tokenizer, skip_prompt = True)


# -----------------------
# Load dataset
# -----------------------
with open(TRAIN_FILE, "r") as jf:
    db = json.load(jf)

random.shuffle(db)


def append_eos(encodings, tokenizer):
    """
    Append EOS token to input_ids and attention_mask in a batch of encodings.

    Args:
        encodings (dict): output from tokenizer(...) with "input_ids" and "attention_mask".
        tokenizer: Hugging Face tokenizer (with eos_token_id defined).

    Returns:
        dict: encodings with EOS appended to input_ids and attention_mask.
    """
    batch_size = encodings["input_ids"].size(0)

    # [batch_size, 1] EOS token column
    eos_tokens = torch.full(
        (batch_size, 1),
        tokenizer.eos_token_id,
        device=encodings["input_ids"].device
    )

    # append EOS to input_ids
    encodings["input_ids"] = torch.cat([encodings["input_ids"], eos_tokens], dim=1)

    # append mask = 1 for EOS
    encodings["attention_mask"] = torch.cat(
        [encodings["attention_mask"], torch.ones_like(eos_tokens)], dim=1
    )

    return encodings

# -----------------------
# Training loop
# -----------------------
for step, ex in enumerate(db, start=1):
    # Build prompt (everything except "output")
    prompt = "### Prompt:\n\n```"
    for k, v in ex.items():
        if k == "output":
            continue
        if isinstance(v, list):
            prompt += f"{k}: {','.join(v)}\n"
        else:
            prompt += f"{k}: {v}\n"
    prompt = prompt.strip() + "```\n\n### Response:\n\n```"
    print(prompt)
    # Target (first output string)
    target = '\n\n'.join(ex["output"]) + "```"

    # Encode prompt + target
    enc = tokenizer(prompt + target, truncation=True,padding="do_not_pad", max_length=MAXSEQ,return_tensors="pt").to("cuda")    
    enc = append_eos(enc,tokenizer)
    labels = enc["input_ids"].clone()
    
    # Mask out prompt tokens (so loss is only on target)
    prompt_len = len(tokenizer(prompt)["input_ids"])
    labels[0, :prompt_len] = -100

    # Forward
    outputs = model(**enc, labels=labels)
    loss = outputs.loss
    if not torch.isfinite(loss):
        print(f"⚠️ Invalid loss detected: {loss.item()}")
        continue 

    # Backprop
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    print(f"\rStep {step} | Loss: {loss.item():.4f}                 ",end='')
        
    if step % 1 == 0:
        inputs = tokenizer(prompt, truncation=True, padding="do_not_pad", max_length=MAXSEQ, return_tensors="pt").to("cuda")        
        try:
            outputs = model.generate(
                **inputs,
                streamer = textstreamer,
                max_new_tokens=50,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                no_repeat_ngram_size=3,                    # ✨ prevents “In your world…” loops
                repetition_penalty=1.15,                   # mild anti-repeat
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        except:
            print("SUPER ERROR DETECTED")    


    if(cleanse):
        torch.cuda.empty_cache()   # frees up unused cached memory    
        torch.cuda.ipc_collect()   # cleans up inter-process memory
        gc.collect()
        torch.cuda.empty_cache()

    # Save checkpoint every 200 steps
    if step % 200 == 0:
        save_dir = model_name
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"✅ Saved checkpoint at {save_dir}")

```
