"""Exercise the installed tokenizer and API request converters offline, without a GPU."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sysconfig

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--expected-stage', choices=('r1', 'r2'), default='r2',
                    help='R1 has no R2 speed layer; all other installed-runtime checks still run.')
args = parser.parse_args()
from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.utils import construct_input_messages, construct_tool_dicts
from fastapi import FastAPI
from fastapi.testclient import TestClient
from dsv41_guard import OfflineGuard

spec = importlib.util.spec_from_file_location('official_1017', args.model / 'encoding/encoding.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
tokenizer = DeepseekV41Tokenizer.from_pretrained(str(args.model), local_files_only=True)
checked = 0
for file in sorted((args.model / 'encoding/tests').glob('test_input_*.json')):
    for case in reference.load_cases(str(file)):
        for mode in ('chat', 'thinking'):
            for effort in ('low', 'high', 'max', 42):
                messages = copy.deepcopy(case['messages'])
                expected = reference.encode_messages(copy.deepcopy(messages), thinking_mode=mode, reasoning_effort=effort)
                actual = tokenizer.apply_chat_template(messages, tokenize=False, thinking=mode == 'thinking', reasoning_effort=effort)
                assert actual == expected, (file, mode, effort)
                assert tokenizer.apply_chat_template(messages, tokenize=True, thinking=mode == 'thinking', reasoning_effort=effort) == tokenizer.encode(expected, add_special_tokens=False)
                checked += 1

config = DeepseekV41Config.from_pretrained(str(args.model), local_files_only=True)
assert config.vision_max_n_token == 1024
assert config.dspark_block_size == 5 and config.is_mm_prefix_lm
request = ResponsesRequest.model_validate({
    'model': 'DeepSeek-Flash', 'instructions': 'Be precise',
    'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'Call read_file to read test.txt'}]}],
    'tools': [{'type': 'function', 'name': 'read_file', 'parameters': {
        'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}],
})
messages = construct_input_messages(request_input=request.input, request_instructions=request.instructions)
tools = construct_tool_dicts(request.tools, request.tool_choice)
encoded = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, thinking=True, reasoning_effort='high')
assert 'read_file' in encoded and 'Be precise' in encoded and 'test.txt' in encoded

app = FastAPI()
@app.post('/v1/chat/completions')
async def endpoint(payload: dict):
    return payload
app.add_middleware(OfflineGuard)
with TestClient(app) as client:
    payload = {'model': 'DeepSeek-Flash', 'messages': [{'role': 'user', 'content': 'test'}]}
    assert client.post('/v1/chat/completions', json=payload).json() == payload
    assert client.post('/v1/chat/completions', json={**payload, 'model': 'DeepSeek-V4.1-Flash'}).status_code == 400
    assert client.post('/v1/chat/completions', json={**payload, 'model': 'DeepSeek-V4.1-Flash-1017'}).status_code == 400
    payload['messages'][0]['content'] = [{'type': 'image_url', 'image_url': {'url': 'https://invalid.example/x.png'}}]
    assert client.post('/v1/chat/completions', json=payload).status_code == 400
    payload['messages'][0]['content'][0]['image_url']['url'] = 'data:image/png;base64,AAAA'
    assert client.post('/v1/chat/completions', json=payload).status_code == 200

site = Path(sysconfig.get_paths()['purelib'])
manifest = json.loads(Path('/opt/dsv411017/image/encoding/manifest.json').read_text())
for row in manifest['files']:
    assert hashlib.sha256((site / row['path']).read_bytes()).hexdigest() == row['patched_sha256']
if args.expected_stage == 'r2':
    speed = json.loads(Path('/opt/dsv411017/image/speed/manifest.json').read_text())
    for row in speed['files']:
        assert hashlib.sha256((site / row['path']).read_bytes()).hexdigest() == row['patched_sha256']
print(json.dumps({'status': 'PASS', 'official_prompt_and_token_id_cases': checked,
                  'vision_budget': config.vision_max_n_token, 'dspark_block_size': config.dspark_block_size,
                  'Responses_conversion': True, 'guard_model_and_inline_images': True,
                  'served_model_name': 'DeepSeek-Flash', 'installed_encoding_hashes': True,
                  'installed_speed_overlay_hashes': True if args.expected_stage == 'r2' else None,
                  'expected_stage': args.expected_stage,
                  'r2_speed_layer_check': 'PASS' if args.expected_stage == 'r2' else 'NOT_APPLICABLE_R1',
                  'gpu_executed': False}))
