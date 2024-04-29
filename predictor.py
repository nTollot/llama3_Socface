# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional
from tqdm import tqdm
import fire

from llama import Dialog, Llama

from load_data import generate_data, tags, random_elements
from analyze_data import create_df


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Cette fonction classifie les informations des différents individus. 

    La fenêtre contextuelle des modèles llama3 est de 8192 tokens, donc `max_seq_len` doit être <= 8192.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    backup_save = 10
    outputs = []    
    for idx in tqdm(random_elements):
        content_sys, content_user = generate_data(idx)
        dialogs: List[Dialog] = [[
            {"role": "system", "content": content_sys},
            {"role": "user", "content": content_user}
            ]] 
        results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        outputs.append(results[0]['generation']['content'])
        if idx%backup_save==0:
            df = create_df(outputs, list(tags.values()), [str(i) for i, _ in enumerate(outputs)])
            df.to_csv("test.csv")
    df = create_df(outputs, list(tags.values()), [str(i) for i, _ in enumerate(outputs)])
    df.to_csv("test.csv")
    print(df)    

if __name__ == "__main__":
    fire.Fire(main)
