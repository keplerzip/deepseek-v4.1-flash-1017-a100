"""CPU-only checks against actual installed native vLLM protocol modules.

Run in the pinned R2 image with the five reviewed overlays read-only mounted.
This checks request conversion, image decoding and prompt-prefix preservation;
it never creates an engine, loads weights, or measures actual cache hits.
"""
import argparse
import asyncio
import base64
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


class Checker:
    def __init__(self, args):
        self.args = args
        self.report = {'schema': 1, 'status': 'RUNNING', 'suite': 'r39-native-protocol-compat',
                       'gpu_executed': False, 'weights_loaded': False, 'network_used': False,
                       'real_inference_executed': False, 'newapi_billing_end_to_end': 'NOT_RUN',
                       'scope': 'Installed native Pydantic, ASGI middleware, protocol translation, image bytes and encoder CPU checks only; prompt prefix characters are not cache-hit tokens.',
                       'checks': [], 'files': []}
        self.project = args.project
        self.fixture = self.project / 'tests/r39-assets'
        self.provenance = json.loads((self.project / 'manifests/r39-overlay-provenance.json').read_text())

    def check(self, name, function):
        row = {'name': name, 'status': 'FAIL'}
        try:
            row['details'] = function()
            row['status'] = 'PASS'
        except Exception as error:
            row.update(error=(type(error).__name__ + ': ' + str(error))[:6000],
                       traceback=traceback.format_exc()[-12000:])
        self.report['checks'].append(row)
        atomic_json(self.args.output, self.report)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def verify_hashes(self):
        rows = []
        for row in self.provenance['files']:
            installed = Path(row['site_packages_path'])
            actual = hashlib.sha256(installed.read_bytes()).hexdigest()
            result = dict(row, installed_sha256=actual)
            self.report['files'].append(result)
            rows.append(result)
        require(len(rows) == 5, 'Exactly five reviewed overlays are required')
        require(all(row['installed_sha256'] == row['patched_sha256'] for row in rows),
                'Installed files are not the exact reviewed protocol overlays')
        return {'installed_hashes_verified': len(rows)}

    def imports(self):
        import vllm.platforms
        from vllm.platforms.cpu import CpuPlatform
        vllm.platforms._current_platform = CpuPlatform()
        self.torch = importlib.import_module('torch')
        self.serving = importlib.import_module('vllm.entrypoints.anthropic.serving').AnthropicServingMessages
        anthropic = importlib.import_module('vllm.entrypoints.anthropic.protocol')
        self.messages_request = anthropic.AnthropicMessagesRequest
        self.count_request = anthropic.AnthropicCountTokensRequest
        self.responses_request = importlib.import_module('vllm.entrypoints.openai.responses.protocol').ResponsesRequest
        self.responses_utils = importlib.import_module('vllm.entrypoints.openai.responses.utils')
        self.chat_utils = importlib.import_module('vllm.entrypoints.chat_utils')
        self.tokenizer_module = importlib.import_module('vllm.tokenizers.deepseek_v41')
        self.encoder = importlib.import_module('vllm.tokenizers.deepseek_v41_encoding')
        self.guard = importlib.import_module('dsv41_guard')
        self.image_io = importlib.import_module('vllm.multimodal.media.image').ImageMediaIO
        require(not self.torch.cuda.is_initialized(), 'CPU checks must not create a CUDA context')
        return {'platform': 'CPU', 'cuda_context_initialized': False}

    def wrapped_tokenizer(self):
        class StubTokenizer:
            """No vocabulary/weight reads: request the actual string encoder."""
            def get_added_vocab(self):
                return {}
        return self.tokenizer_module.get_deepseek_v41_tokenizer(StubTokenizer())

    def tokenizer_capability(self):
        wrapped = self.wrapped_tokenizer()
        require(getattr(wrapped, 'supports_inline_system_messages', False) is True,
                'Native 1017 tokenizer must advertise inline system support')
        require(self.serving._detect_merge_inline_system(None, wrapped) is False,
                '1017 encoder-based tokenizer must not merge inline system turns')
        require(self.serving._detect_merge_inline_system(None, object()) is True,
                'Unknown tokenizer must retain conservative system-first fallback')
        return {'inline_system_capability': True, 'unknown_tokenizer_fallback_preserved': True}

    def tail_prefix(self):
        wrapped = self.wrapped_tokenizer()
        merge = self.serving._detect_merge_inline_system(None, wrapped)
        long_text = 'Stable earlier conversation data, no audit budget here.\n' * 256
        cold = {'model': 'DeepSeek-Flash', 'system': 'Stable global instruction.', 'max_tokens': 64,
                'messages': [{'role': 'user', 'content': long_text},
                             {'role': 'system', 'content': 'Remaining audit budget: 99.'}]}
        followup = copy.deepcopy(cold)
        followup['messages'].extend([{'role': 'assistant', 'content': '{"check":73019}'},
                                     {'role': 'user', 'content': 'Return the same check JSON.'},
                                     {'role': 'system', 'content': 'Remaining audit budget: 98.'}])
        converted = [self.serving.to_chat_completion_request(self.messages_request(**body),
                     merge_inline_system=merge) for body in (cold, followup)]
        require([item['role'] for item in converted[0].messages] == ['system', 'user', 'system'],
                'Trailing system turn moved away from its original position')
        require(all(item.messages[0]['content'] == cold['system'] for item in converted),
                'Changing trailing system content contaminated the global prefix')
        prompts = [wrapped.apply_chat_template(item.messages, tokenize=False, thinking=False) for item in converted]
        prefix = os.path.commonprefix(prompts)
        require(long_text in prefix, 'Native official encoder lost the reusable earlier conversation prefix')
        count_body = {key: value for key, value in followup.items() if key != 'max_tokens'}
        count_converted = self.serving.to_chat_completion_request(self.count_request(**count_body), merge_inline_system=merge)
        require(count_converted.messages == converted[1].messages, 'Messages count_tokens and generation conversion differ')
        return {'cold_roles': [item['role'] for item in converted[0].messages],
                'followup_roles': [item['role'] for item in converted[1].messages],
                'common_prefix_characters': len(prefix), 'earlier_user_characters': len(long_text),
                'generation_count_conversion_equal': True,
                'cache_hit_tokens_measured': False}

    def image_source(self, filename):
        return {'type': 'base64', 'media_type': 'image/png',
                'data': base64.b64encode((self.fixture / filename).read_bytes()).decode()}

    def tool_images(self):
        body = {'model': 'DeepSeek-Flash', 'max_tokens': 64,
                'messages': [{'role': 'user', 'content': 'Read local images.'},
                    {'role': 'assistant', 'content': [
                        {'type': 'tool_use', 'id': ident, 'name': 'view_image', 'input': {'label': ident}}
                        for ident in ('call_a', 'call_b')]},
                    {'role': 'user', 'content': [
                        {'type': 'tool_result', 'tool_use_id': ident, 'content': [
                            {'type': 'text', 'text': ident},
                            {'type': 'image', 'source': self.image_source(filename)}]}
                        for ident, filename in [('call_b', 'video_frames/03.png'), ('call_a', 'video_frames/01.png')]]}]}
        converted = self.serving.to_chat_completion_request(self.messages_request(**body), merge_inline_system=False)
        tools = [item for item in converted.messages if item['role'] == 'tool']
        require(len(tools) == 2, 'Expected two original tool-result slots')
        require([item['tool_call_id'] for item in tools] == ['call_b', 'call_a'], 'Incoming tool-result IDs changed')
        for item, filename in zip(tools, ['video_frames/03.png', 'video_frames/01.png']):
            require(type(item['content']) is list, 'Tool image detached into an unassociated user turn')
            require([part['type'] for part in item['content']] == ['text', 'image_url'], 'Tool text/image order changed')
            expected = 'data:image/png;base64,' + self.image_source(filename)['data']
            require(item['content'][1]['image_url']['url'] == expected, 'Tool image/call-ID binding changed')
        require(len([item for item in converted.messages if item['role'] == 'user']) == 1,
                'Tool images must not become standalone user messages')
        ordered = self.encoder.sort_tool_messages_for_media(converted.messages)
        require([item['tool_call_id'] for item in ordered if item['role'] == 'tool'] == ['call_a', 'call_b'],
                'Native 1017 renderer media-order helper did not follow assistant call order')
        prompt = self.wrapped_tokenizer().apply_chat_template(ordered, tokenize=False, thinking=False)
        require(prompt.count(self.encoder.IMAGE_PLACEHOLDER) == 2, 'Official encoding lost image placeholders')
        require(prompt.index('call_a') < prompt.index('call_b'), 'Tool text order no longer follows call IDs')
        return {'wire_order': ['call_b', 'call_a'], 'renderer_order': ['call_a', 'call_b'],
                'images_preserved_in_originating_tool': True, 'image_placeholders': 2}

    def responses_details(self):
        url = 'data:image/png;base64,' + self.image_source('video_frames/01.png')['data']
        for location in ('user', 'function_call_output', 'custom_tool_call_output'):
            part = {'type': 'input_image', 'image_url': url}
            item = {'role': 'user', 'content': [part]} if location == 'user' else {
                'type': location, 'call_id': 'call_a', 'output': [part]}
            original = copy.deepcopy(item)
            req = self.responses_request(model='DeepSeek-Flash', max_output_tokens=64, input=[item])
            require(item == original, 'Responses validator mutated caller-owned input')
            normalized = req.input[0]
            if not isinstance(normalized, dict):
                normalized = normalized.model_dump()
            image = list(normalized['content' if location == 'user' else 'output'])[0]
            require(image.get('detail') == 'auto', 'Omitted Responses input_image.detail was not defaulted')
            self.chat_utils._parse_chat_message_content_mm_part(image)
        # Both direct messages and tool outputs reject explicitly invalid detail.
        for location in ('user', 'function_call_output'):
            part = {'type': 'input_image', 'image_url': url, 'detail': 'ultra-invalid'}
            item = {'role': 'user', 'content': [part]} if location == 'user' else {
                'type': location, 'call_id': 'call_a', 'output': [part]}
            try:
                self.responses_request(model='DeepSeek-Flash', max_output_tokens=64, input=[item])
            except ValueError:
                continue
            raise AssertionError('Explicit invalid detail was accepted')
        return {'default_detail': 'auto', 'locations': ['user', 'function_call_output', 'custom_tool_call_output'],
                'explicit_invalid_detail_rejected': True}

    def tool_business_json(self):
        for value in ({'type': 'image', 'filename': 'photo.png'}, {'type': 'file', 'path': 'business-value'}):
            body = {'model': 'DeepSeek-Flash', 'messages': [{'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'call1', 'name': 'classify', 'input': copy.deepcopy(value)}]}],
                'tools': [{'name': 'classify', 'input_schema': {'type': 'object', 'example': copy.deepcopy(value)}}]}
            self.guard.check_media(body)
            self.guard.check_media({'tools': [{'type': 'function', 'function': {'name': 'classify',
                'parameters': {'type': 'object', 'example': copy.deepcopy(value)}}}]})
        # Genuine images nested in tool results/Responses outputs must still be
        # checked; business-JSON exemption is not a blanket skip of tool turns.
        rejected = 0
        for image in ({'type': 'image_url', 'image_url': {'url': 'https://example.invalid/image.png'}},
                      {'type': 'input_image', 'image_url': 'file:///etc/passwd'}):
            for payload in ({'messages': [{'role': 'tool', 'tool_call_id': 'call1', 'content': [image]}]},
                            {'input': [{'type': 'function_call_output', 'call_id': 'call1', 'output': [image]}]}):
                try:
                    self.guard.check_media(payload)
                except ValueError:
                    rejected += 1
                else:
                    raise AssertionError('Real tool-output media bypassed offline validation')
        return {'tool_use_input_business_json_allowed': True, 'json_schema_examples_allowed': True,
                'real_tool_output_external_media_rejections': rejected}

    async def through_guard(self, body):
        captured, sent = {}, []
        async def app(scope, receive, send):
            event = await receive()
            payload = json.loads(event['body'])
            req = self.responses_request(**payload)
            messages = self.responses_utils.construct_input_messages(request_input=req.input)
            images = [part for message in messages for part in message.get('content', [])
                      if isinstance(part, dict) and part.get('type') == 'input_image']
            for image in images:
                _, url = self.chat_utils._parse_chat_message_content_mm_part(image)
                self.image_io().load_base64('application/octet-stream', url.split(',', 1)[1])
            captured.update(messages=messages, image_count=len(images))
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'CPU_PROTOCOL_PARSED'})
        delivered = False
        async def receive():
            nonlocal delivered
            require(not delivered, 'Fixture application read body more than once')
            delivered = True
            return {'type': 'http.request', 'body': json.dumps(body).encode(), 'more_body': False}
        async def send(event):
            sent.append(event)
        scope = {'type': 'http', 'method': 'POST', 'path': '/v1/responses', 'headers': []}
        # No authentication tokens are needed or inspected in this CPU fixture.
        previous = os.environ.pop('VLLM_API_KEY', None)
        try:
            await self.guard.OfflineGuard(app)(scope, receive, send)
        finally:
            if previous is not None:
                os.environ['VLLM_API_KEY'] = previous
        status = next(event['status'] for event in sent if event['type'] == 'http.response.start')
        return status, captured

    def guard_pipeline(self):
        data = self.image_source('video_frames/01.png')['data']
        image = {'type': 'input_image', 'image_url': 'data:application/octet-stream;base64,' + data}
        body = {'model': 'DeepSeek-Flash', 'max_output_tokens': 64, 'input': [
            {'type': 'function_call', 'name': 'view_image', 'call_id': 'call_a', 'arguments': '{}'},
            {'type': 'function_call_output', 'call_id': 'call_a', 'output': [image]}]}
        status, captured = asyncio.run(self.through_guard(body))
        require(status == 200 and captured.get('image_count') == 1,
                'Valid Codex octet-stream PNG did not pass guard -> native protocol -> actual image decoding')
        tool = next(item for item in captured['messages'] if item.get('role') == 'tool')
        require(tool['tool_call_id'] == 'call_a' and tool['content'][0]['detail'] == 'auto',
                'view_image tool call ID/default detail changed in middleware pipeline')
        for url in ('https://example.invalid/a.png', 'file:///etc/passwd',
                    'data:application/octet-stream;base64,bm90LWFuLWltYWdl',
                    'data:application/octet-stream;base64,invalid!'):
            invalid = copy.deepcopy(body)
            invalid['input'][1]['output'][0]['image_url'] = url
            rejected, _ = asyncio.run(self.through_guard(invalid))
            require(rejected == 400, 'Offline guard accepted unsupported media source: ' + url[:60])
        return {'valid_png_octet_stream_status': status, 'actual_image_decode': True,
                'tool_call_id': 'call_a', 'remote_file_nonimage_invalid_base64_rejected': True,
                'inference_performed': False}

    def run(self):
        self.check('installed-overlay-sha256', self.verify_hashes)
        self.check('installed-cpu-imports-no-cuda', self.imports)
        if self.report['checks'][-1]['status'] == 'PASS':
            for name, function in (
                ('v41-inline-system-capability-and-fallback', self.tokenizer_capability),
                ('messages-tail-system-multiturn-prefix-and-count', self.tail_prefix),
                ('messages-tool-result-image-call-id-order', self.tool_images),
                ('responses-image-detail-native-pydantic', self.responses_details),
                ('tool-business-json-schema-not-media', self.tool_business_json),
                ('codex-octet-stream-full-guard-protocol-image-pipeline', self.guard_pipeline),
            ):
                self.check(name, function)
            self.check('no-cuda-context-after-checks', lambda: (
                require(not self.torch.cuda.is_initialized(), 'Unexpected CUDA initialization') or {'cuda_context_initialized': False}))
        self.report['status'] = 'PASS' if len(self.report['checks']) == 9 and all(
            row['status'] == 'PASS' for row in self.report['checks']) else 'FAIL'
        atomic_json(self.args.output, self.report)
        print(json.dumps({'status': self.report['status'], 'gpu_executed': False,
                          'report': str(self.args.output), 'checks': len(self.report['checks'])}), flush=True)
        return 0 if self.report['status'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        return Checker(args).run()
    except Exception as error:
        atomic_json(args.output, {'schema': 1, 'suite': 'r39-native-protocol-compat', 'status': 'FAIL',
                    'gpu_executed': False, 'weights_loaded': False, 'real_inference_executed': False,
                    'error': type(error).__name__ + ': ' + str(error), 'traceback': traceback.format_exc()})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
