import sys

sys.path.append('/home/trl/trl')

# available_gpus = ['0', '1','2']

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(available_gpus)

import numpy as np
import torch
import re
import random
import transformers
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer,
)

from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
# from utils import download_url, load_jsonl
import argparse

from utils.text_generation import AgentManager
from gsm8k_train.gsm8k_utils import is_correct, clean_answer

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "The answer is"


def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name_or_path,
        quantization_config=nf4_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    tokenizer = LlamaTokenizer.from_pretrained('/home/trl/trl/hf_hub/models/meta-llama-Llama-2-7b-chat-hf')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer.pad_token)
    print(tokenizer.bos_token)
    print(tokenizer.unk_token)
    print(tokenizer.eos_token)
    print(tokenizer.truncation_side)
    print(tokenizer.padding_side)

    model.eval()

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k_shot",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='/home/trl/trl/hf_hub/models/meta-llama-Llama-2-7b-chat-hf',
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="The root folder of the data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--load", type=str, default=None, help="load quantized model")

    args = parser.parse_args()
    return args


def build_dataset(config, query_dataset, input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained('/home/trl/trl/hf_hub/models/meta-llama-Llama-2-7b-chat-hf')
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(query_dataset, 'main', split="test")

    # ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    # input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        # sample["query"] = B_INST + sample["prompt"] + E_INST
        sample["query"] = sample["question"]
        sample["response"] = sample["answer"]
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


@torch.no_grad()
def main():
    args = parse_args()
    k_shot = args.k_shot

    ppo_config = PPOConfig(
        # model_name="/home/trl/trl/hf_hub/models/llama2-7b-chat-dtse/05022051",
        # model_name="/home/trl/trl/hf_hub/models/llama2-7b-chat-ppo/05052208",
        model_name='/home/trl/trl/hf_hub/models/meta-llama-Llama-2-7b-chat-hf',
        query_dataset="/home/trl/trl/hf_hub/datasets/gsm8k",
        ppo_epochs=1,
        learning_rate=1e-4,
        log_with=None,
        mini_batch_size=1,
        batch_size=32,
        gradient_accumulation_steps=16,
        early_stopping=True,
        target_kl=6.0,
        kl_penalty="full",
        seed=123,
        use_score_scaling=False,
        use_score_norm=False,
        # score_clip=50.0,
        init_kl_coef=0.1,
        adap_kl_ctrl=False,
        # lam=0.95,
        # cliprange_value=5.0,
        # optimize_cuda_cache=True,
        # optimize_device_cache=True,
        # world_size=4
    )

    seed_everything(args.seed)

    dataset = build_dataset(ppo_config, ppo_config.query_dataset)

    model, tokenizer = load(ppo_config.model_name)
    ppo_trainer = PPOTrainer(ppo_config, model, tokenizer=tokenizer, dataset=dataset, data_collator=collator)

    generate_kwargs = {
        "top_k": 10,
        "top_p": 0.7,
        "temperature": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": -1,
        "repetition_penalty": 1.1,
        "max_new_tokens": 256,
    }

    agent = AgentManager(ppo_trainer, tokenizer, generate_kwargs, system_prompt='Question: Shawn has five toys. For '
                                                                                'Christmas, he got two toys each from'
                                                                                'his mom and dad. How many toys does he '
                                                                                'have now?\nAnswer: Shawn started with '
                                                                                '5 toys. If he got 2 toys each from his '
                                                                                'mom and dad, then that is 4 more toys. '
                                                                                '5 + 4 = 9.\nQuestion: {}\n Answer:')

    # TODO: add pass@k method
    answers = []
    for batch in tqdm(ppo_trainer.dataloader):
        cor_stack = []
        for _ in range(k_shot):
            response_tensors = agent.get_response(batch["query"])
            responses = [tokenizer.decode(r.squeeze(),
                                          skip_special_tokens=True,
                                          ignore_tokenization_space=True, )
                         for r in response_tensors]
            is_cor = [float(is_correct(clean_answer(r), a)) for r, a in zip(responses, batch["response"])]
            cor_stack.append(is_cor)
        cor_stack = np.vstack(cor_stack)
        answers.extend(np.any(cor_stack, axis=0))
        # if DEBUG:
        #     print(f"Full input_text:\n{input_text}\n\n")
        # print(
        #     # f'Question: {sample["instruction"]}\n\n'
        #     # f'Answers: {extract_answer_from_output(sample["output"])}\n\n'
        #     # f"Model Answers: {model_answer}\n\n"
        #     # f"Model Completion: {model_completion}\n\n"
        #     f"Is correct: {is_cor}\n\n"
        # )

        print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {float(sum(answers)) / len(answers)}."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        for answer in answers:
            print(answer, file=f)

    with open(os.path.join(args.output_dir, "scores.txt"), "w") as f:
        print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {float(sum(answers)) / len(answers)}.",
            file=f,
        )


if __name__ == "__main__":
    main()
