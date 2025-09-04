import os

def generate_with_openai(prompt: str, max_tokens: int = 256, model: str = "gpt-4") -> str:
    if openai is None:
        raise RuntimeError("get some OpenAI package.")
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("find some key")
    openai.api_key = os.environ["API_KEY_HEREEEEEEEEEEE_ARGGGGGGGGGGGGGGGGGGGGGGGGGG"]
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are an expert in thermodynamics."},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return resp["choices"][0]["message"]["content"].strip()

# HF generator wrapper 
def generate_with_hf(prompt: str, model_name: str = "databricks/dolly-v2-3b", max_new_tokens: int = 256) -> str:
    if pipeline is None:
        raise RuntimeError("transformers pipeline not available. Install transformers + torch.")
    # BE CAREFUL: large models may require GPU and lots of RAM.
    gen = pipeline("text-generation", model=model_name, device=0 if torch.cuda.is_available() else -1)
    out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]["generated_text"]
