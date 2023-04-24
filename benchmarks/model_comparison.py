import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def gen_payload(datum):
    if len(datum['instances']) == 0:
        prompt = f"<human>: {datum['instruction']}"
    else:
        prompt = f"<human>: {datum['instruction']}\n{datum['instances'][0]['input']}"
    res = {
        "prompt": prompt+"\n<bot>:",
        "max_tokens": 256,
        "temperature": 0.9,
        "top_p": 0.9,
        "top_k": 50,
        "n": 1,
    }
    return res

def load_model(path: str):
    model = AutoModelForCausalLM.from_pretrained(path)
    model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

def generate(model, tokenizer, payload):
    inputs = tokenizer(payload['prompt'], padding=True, truncation=True, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=payload['max_tokens'],
        do_sample=True,
        top_k=payload['top_k'],
        top_p=payload['top_p'],
        temperature=payload['temperature'],
        return_dict_in_generate=True,
        output_scores=False, # return logit score
        output_hidden_states=True,
    )
    token = outputs.sequences[0, input_length:]
    output = tokenizer.decode(token)
    return output

def mix(models, weights):
    # multiplies the weights by the logits
    for i, model in enumerate(models):
        for param in model.parameters():
            param.data *= weights[i]
    
    for i, model in enumerate(models):
        if i>0:
            for param0, param in zip(models[0].parameters(), model.parameters()):
                param0.data += param.data
    return models[0]

def main():
    with open("./.cache/datasets/human_eval.jsonl", "r") as fp:
        data = [json.loads(line) for line in fp]

    model_0, tokenizer = load_model("./.cache/models/pythia-gpt4all-6000/")
    model_1, _ = load_model("./.cache/models/pythia-sharegpt-6000/")
    model_2, _ = load_model("./.cache/models/pythia-sharegpt_gpt4all-12000/")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    mix_model = mix([model_0, model_1], [0.5, 0.5])

    models_kw = {
        "pythia-gpt4all-6000": model_0,
        "pythia-sharegpt-6000": model_1,
        "pythia-sharegpt-gpt4all_12000": model_2,
        "mixed_model": mix_model
    }
    results = []
    for k, v in models_kw.items():
        v.to(device)
        for datum in data:
            payload = gen_payload(datum)
            output = generate(v, tokenizer, payload)
            results.append({
                "model": k,
                "input": payload['prompt'],
                "output": output,
                "payload": payload,
                "datum": datum,
            })
    with open("results.jsonl", "w") as fp:
        for line in results:
            fp.write(json.dumps(line)+"\n")
    
if __name__ == "__main__":
    main()