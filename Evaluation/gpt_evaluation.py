# -*- coding: gbk -*-
import sys
import os
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_folder_path)

import argparse
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
import time


# Load environment variables from .env file
load_dotenv()


'''
Initialize Azure OpenAI client based on model name
Returns: AzureOpenAI client instance
'''
def init_azure_client(model_name):
    # Determine which endpoint and API key to use based on model name
    if model_name.startswith('gpt-4'):
        endpoint = os.getenv('GPT4_ENDPOINT')
        api_key = os.getenv('GPT4_API_KEY')
    elif model_name.startswith('gpt-5'):
        endpoint = os.getenv('GPT5_ENDPOINT')
        api_key = os.getenv('GPT5_API_KEY')
    else:
        # Default to GPT4 for other models (or you can add more conditions)
        endpoint = os.getenv('GPT4_ENDPOINT')
        api_key = os.getenv('GPT4_API_KEY')
    
    api_version = os.getenv('API_VERSION')
    
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )
    
    return client


'''
Call GPT to generate answer
Returns: (answer, usage_dict) where usage_dict contains token usage information
'''
def gpt_answer(prompt, model_name, client):
    max_try = 2
    
    # Try several times in case of accident like internet disconnection.
    for attempt in range(max_try):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            # Request successful - extract token usage
            usage = response.usage
            # Extract cached tokens from prompt_tokens_details
            cached_tokens = 0
            if usage and hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
            
            usage_dict = {
                'prompt_tokens': usage.prompt_tokens if usage else 0,
                'completion_tokens': usage.completion_tokens if usage else 0,
                'total_tokens': usage.total_tokens if usage else 0,
                'prompt_cache_hit_tokens': cached_tokens
            }
            return response.choices[0].message.content, usage_dict
        except Exception as e:
            print(f"[!] Exception occurred: {e} at attempt {attempt+1}")
        if attempt < max_try - 1:
            print("[ ] Waiting 5 seconds before retrying...")
            time.sleep(5)
    # Return error with zero usage
    error_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'prompt_cache_hit_tokens': 0}
    return "Error", error_usage


# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate answers using a selected model and trick.')
    parser.add_argument('--selected_model', default='gpt-3.5-turbo', type=str, required=True, help='The model to use for generation.')
    parser.add_argument('--trick', type=str, default='zero-shot', required=False, help='The trick to apply for generation.')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for the dataset and output files.')
    
    args = parser.parse_args()
    selected_model = args.selected_model
    trick = args.trick
    dataset_name = 'mini-StatQA for ' + trick
    suffix = args.suffix

    # Path setting
    input_csv_path = 'Data/Integrated Dataset/Dataset with Prompt/Test Set/' + dataset_name + f'{suffix}.csv'
    output_csv_path = f'Model Answer/Origin Answer/{selected_model}_{trick}{suffix}.csv'
    if os.path.exists(output_csv_path):
        df_existing = pd.read_csv(output_csv_path)
        last_processed_row = len(df_existing)
    else:
        df_existing = pd.DataFrame()
        last_processed_row = 0

    df = pd.read_csv(input_csv_path)
    start_row = max(last_processed_row, 0)
    print(f"Starting from row: {start_row}")
    answers = []
    start_time = time.time()
    
    # Initialize Azure OpenAI client once (reused for all requests to benefit from caching)
    print(f"Initializing Azure OpenAI client for model: {selected_model}")
    azure_client = init_azure_client(selected_model)
    
    # Token usage tracking
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0
    total_tokens = 0

    # Generate answer
    for index, row in df.iterrows():
        if index < start_row:
            continue
        prompt = row['prompt']
        answer, usage = gpt_answer(prompt, selected_model, azure_client)
        answers.append(answer)
        
        # Accumulate token usage
        total_prompt_tokens += usage['prompt_tokens']
        total_completion_tokens += usage['completion_tokens']
        total_cached_tokens += usage['prompt_cache_hit_tokens']
        total_tokens += usage['total_tokens']
        
        if (index + 1) % 5 == 0 or index == len(df) - 1:
            answer_slice = pd.Series(answers)
            slice_start = index + 1 - len(answers)
            slice_end = index + 1
            df.loc[df.index[slice_start:slice_end], 'model_answer'] = answer_slice.values
            df_existing = pd.concat([df_existing, df.loc[df.index[slice_start:slice_end]]])
            df_existing.to_csv(output_csv_path, index=False)
            answers = []
            print(f'[+] Model: {selected_model}. Processed and saved up to row: {index}')
            print(f'    Token usage - Input: {total_prompt_tokens}, Cached: {total_cached_tokens}, Output: {total_completion_tokens}, Total: {total_tokens}')

    end_time = time.time()
    time_consumption = end_time - start_time
    print('------------------------------------------------')
    print(f"Finished, output path: {output_csv_path}")
    print(f"Time consumption: {time_consumption} seconds")
    print(f"Token usage:")
    print(f"  Total input tokens: {total_prompt_tokens}")
    print(f"  Total cached tokens: {total_cached_tokens}")
    print(f"  Total output tokens: {total_completion_tokens}")
    print(f"  Total tokens: {total_tokens}")
    print('------------------------------------------------')