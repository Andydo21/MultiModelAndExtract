"""Portable adapter for the OpenAI-compatible Kaggle Qwen gateway."""
import json
import os
import re
import requests

HEADERS = {'ngrok-skip-browser-warning': 'true'}


def base_url():
    return os.getenv('VLLM_BASE_URL', '').rstrip('/')


def is_connected(legacy_url=''):
    try:
        if base_url():
            response = requests.get(base_url() + '/models', headers=HEADERS, timeout=5)
            data = response.json()
            model = os.getenv('QWEN_MODEL_ID', 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ')
            return response.ok and any(item.get('id') == model for item in data.get('data', []))
        if legacy_url:
            response = requests.get(legacy_url.rstrip('/') + '/health', headers=HEADERS, timeout=5)
            return response.ok and response.json().get('status') in ('healthy', 'ok', 'ready')
    except (requests.RequestException, ValueError, AttributeError, TypeError):
        pass
    return False


def chat(system, text, max_tokens=900, json_output=False):
    if not base_url():
        raise RuntimeError('VLLM_BASE_URL is not configured')
    body = {'model': os.getenv('QWEN_MODEL_ID', 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ'),
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': text}],
            'max_tokens': max_tokens, 'temperature': 0.1}
    if json_output:
        body['response_format'] = {'type': 'json_object'}
    response = requests.post(base_url() + '/chat/completions', json=body, headers=HEADERS, timeout=300)
    response.raise_for_status()
    content = response.json()['choices'][0]['message']['content'].strip()
    if not json_output:
        return content
    fenced = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
    if fenced:
        content = fenced.group(1)
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError('Qwen must return a JSON object')
    return data
