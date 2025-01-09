import time, tiktoken
from openai import OpenAI
import openai
import os, anthropic, json

TOKENS_IN = dict()
TOKENS_OUT = dict()

encoding = tiktoken.get_encoding("cl100k_base")

def curr_cost_est() -> float:
    """
    Calculate the current cost estimate based on token usage.
    
    Returns:
        float: Total estimated cost in dollars
    """
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
    }
    costmap_out = {
        "gpt-4o": 10.00/ 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet": 12.00 / 1000000,
    }
    return sum([costmap_in[_]*TOKENS_IN[_] for _ in TOKENS_IN]) + sum([costmap_out[_]*TOKENS_OUT[_] for _ in TOKENS_OUT])

def setup_api_clients(openai_api_key: str | None, anthropic_api_key: str | None) -> OpenAI | None:
    """
    Set up API clients for OpenAI and Anthropic.
    
    Args:
        openai_api_key: OpenAI API key
        anthropic_api_key: Anthropic API key
        
    Returns:
        OpenAI: OpenAI client if OpenAI key is provided, None otherwise
        
    Raises:
        Exception: If no API keys are provided
    """
    preloaded_api = os.getenv('OPENAI_API_KEY')
    if openai_api_key is None and preloaded_api is not None:
        openai_api_key = preloaded_api
    if openai_api_key is None and anthropic_api_key is None:
        raise Exception("No API key provided")
    if openai_api_key:
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    return OpenAI() if openai_api_key else None

def query_gpt4o_mini(client: OpenAI, prompt: str, system_prompt: str, version: str, temp: float | None = None) -> str:
    """
    Query the GPT-4O Mini model.
    
    Args:
        client: OpenAI client instance
        prompt: User input prompt
        system_prompt: System context prompt
        version: API version
        temp: Temperature parameter for response randomness
        
    Returns:
        str: Model's response text
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
    model = "gpt-4o-mini-2024-07-18"
    if version == "0.28":
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini", messages=messages, temperature=temp)
    else:
        completion = client.chat.completions.create(
            model=model, messages=messages, temperature=temp)
    return completion.choices[0].message.content

def query_claude(prompt: str, system_prompt: str) -> str:
    """
    Query the Claude model.
    
    Args:
        prompt: User input prompt
        system_prompt: System context prompt
        
    Returns:
        str: Model's response text
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-3-5-sonnet-latest",
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}])
    return json.loads(message.to_json())["content"][0]["text"]

def query_gpt4o(client: OpenAI, prompt: str, system_prompt: str, version: str, temp: float | None = None) -> str:
    """
    Query the GPT-4O model.
    
    Args:
        client: OpenAI client instance
        prompt: User input prompt
        system_prompt: System context prompt
        version: API version
        temp: Temperature parameter for response randomness
        
    Returns:
        str: Model's response text
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
    model = "gpt-4o-2024-08-06"
    if version == "0.28":
        completion = openai.ChatCompletion.create(
            model="gpt-4o", messages=messages, temperature=temp)
    else:
        completion = client.chat.completions.create(
            model=model, messages=messages, temperature=temp)
    return completion.choices[0].message.content

def query_o1(client: OpenAI, prompt: str, system_prompt: str, version: str, model_str: str) -> str:
    """
    Query O1 models (mini or preview).
    
    Args:
        client: OpenAI client instance
        prompt: User input prompt
        system_prompt: System context prompt
        version: API version
        model_str: Model identifier ('o1-mini' or 'o1-preview')
        
    Returns:
        str: Model's response text
    """
    messages = [{"role": "user", "content": system_prompt + prompt}]
    model = "o1-mini-2024-09-12" if model_str == "o1-mini" else "o1-preview"
    if version == "0.28":
        completion = openai.ChatCompletion.create(
            model=model_str, messages=messages)
    else:
        completion = client.chat.completions.create(
            model=model, messages=messages)
    return completion.choices[0].message.content

def update_token_counts(model_str: str, system_prompt: str, prompt: str, answer: str) -> None:
    """
    Update token count trackers for input and output.
    
    Args:
        model_str: Model identifier
        system_prompt: System context prompt
        prompt: User input prompt
        answer: Model's response
    """
    if model_str in ["o1-preview", "o1-mini", "claude-3.5-sonnet"]:
        encoding = tiktoken.encoding_for_model("gpt-4o")
    else:
        encoding = tiktoken.encoding_for_model(model_str)
    if model_str not in TOKENS_IN:
        TOKENS_IN[model_str] = 0
        TOKENS_OUT[model_str] = 0
    TOKENS_IN[model_str] += len(encoding.encode(system_prompt + prompt))
    TOKENS_OUT[model_str] += len(encoding.encode(answer))

def query_model(
    model_str: str,
    prompt: str,
    system_prompt: str,
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    tries: int = 5,
    timeout: float = 5.0,
    temp: float | None = None,
    print_cost: bool = True,
    version: str = "1.5"
) -> str:
    """
    Main function to query various AI models with retry mechanism.
    
    Args:
        model_str: Model identifier
        prompt: User input prompt
        system_prompt: System context prompt
        openai_api_key: OpenAI API key
        anthropic_api_key: Anthropic API key
        tries: Number of retry attempts
        timeout: Timeout between retries in seconds
        temp: Temperature parameter for response randomness
        print_cost: Whether to print cost estimates
        version: API version
        
    Returns:
        str: Model's response text
        
    Raises:
        Exception: If max retries are exceeded
    """
    client = setup_api_clients(openai_api_key, anthropic_api_key)
    
    for _ in range(tries):
        try:
            if model_str in ["gpt-4o-mini", "gpt4omini", "gpt-4omini", "gpt4o-mini"]:
                answer = query_gpt4o_mini(client, prompt, system_prompt, version, temp)
            elif model_str == "claude-3.5-sonnet":
                answer = query_claude(prompt, system_prompt)
            elif model_str in ["gpt4o", "gpt-4o"]:
                answer = query_gpt4o(client, prompt, system_prompt, version, temp)
            elif model_str in ["o1-mini", "o1-preview"]:
                answer = query_o1(client, prompt, system_prompt, version, model_str)
            
            update_token_counts(model_str, system_prompt, prompt, answer)
            if print_cost:
                print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
            return answer
            
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")