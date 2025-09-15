from openai import OpenAI
from environment.config.config import config

client = OpenAI(api_key='<KEY>')

llm_api_key = config['llm']['api_key']
llm_base_url = config['llm']['base_url']
client.api_key = llm_api_key
client.base_url = llm_base_url


def deepseek(model="deepseek-v3", system=None, user=None):
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    if user is not None:
        messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response


def claude(model="claude-3-7-sonnet-20250219", system=None, user=None):
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    if user is not None:
        messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response


def gpt(model="gpt-4o-mini", system=None, user=None, messages=None):
    if messages is not None:
        pass
    else:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        if user is not None:
            messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response
