"""Exercise the installed native Responses generator with synthetic engine deltas.

This is protocol validation, not inference or a model-quality/performance test.
"""
import argparse
import asyncio
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import socket

import vllm.platforms
from vllm.platforms.cpu import CpuPlatform
vllm.platforms._current_platform = CpuPlatform()
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.logprobs import Logprob as SampleLogprob
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.openai.responses.api_router import _convert_stream_to_sse_events
from vllm.parser.deepseek_v41 import DeepSeekV41Parser


class StubTokenizer:
    """Minimal tokenizer for native grammar parsing; no weights/vocab are read."""
    def get_vocab(self):
        return {}

    def decode(self, token_ids, **kwargs):
        return ''


FUNCTION = {'type':'function', 'name':'spawn_agent', 'parameters':{
    'type':'object', 'properties':{'task':{'type':'string'}}, 'required':['task']}}
NAMESPACE = {'type':'namespace', 'name':'functions', 'description':'Delegation fixtures', 'tools':[FUNCTION]}


def invocation(name, task):
    return (f'<｜DSML｜ invoke name="{name}">'
            f'<｜DSML｜ parameter name="task" string="true">{task}</｜DSML｜ parameter>'
            '</｜DSML｜ invoke>')


def protocol_signature(items):
    """Compare identities, visible content and tool semantics, not final metadata."""
    result = []
    for item in items:
        row = {'type':item['type'], 'id':item['id']}
        if item['type'] == 'function_call':
            row.update({key:item.get(key) for key in ['call_id','name','arguments','namespace']})
        else:
            row['content'] = [{'type':part['type'], 'text':part.get('text','')}
                              for part in item.get('content', [])]
        result.append(row)
    return result


def without_identity(items):
    result = copy.deepcopy(items)
    for item in result:
        item.pop('id', None)
        if item['type'] == 'function_call':
            item.pop('call_id', None)
    return result


async def run_case(case):
    chunks = case['chunks']
    thinking = case.get('thinking', False)
    include_reasoning = case.get('include_reasoning', True)
    native_parser = case.get('native_parser', True)
    finish_reason = case.get('finish_reason', 'stop')
    with_logprobs = case.get('logprobs', False)
    extra = {'include':['message.output_text.logprobs'], 'top_logprobs':2} if with_logprobs else {}
    request = ResponsesRequest(model='DeepSeek-Flash', input='SYNTHETIC greeting fixture',
                               stream=True, store=True, include_reasoning=include_reasoning,
                               tools=case.get('tools', []), **extra)
    serving = object.__new__(OpenAIServingResponses)
    serving.use_harmony = False
    serving.parser = DeepSeekV41Parser if native_parser else None
    serving.enable_log_outputs = False
    serving.request_logger = None
    serving.enable_auto_tools = True
    serving.tool_server = None
    serving.return_tokens_as_token_ids = False
    serving.response_store_lock = asyncio.Lock()
    serving.response_store = {}
    serving.event_store = {}
    tokenizer = StubTokenizer()
    def make_context(context_request):
        return SimpleContext(parser_cls=DeepSeekV41Parser if native_parser else None,
                             tokenizer=tokenizer, request=context_request,
                             chat_template_kwargs={'thinking':thinking, 'reasoning_effort':'high' if thinking else 'none'})
    context = make_context(request)
    params = SamplingParams(max_tokens=32, logprobs=2 if with_logprobs else None)

    async def results(target_context=context):
        for index, text in enumerate(chunks):
            logprobs = [{100+index:SampleLogprob(logprob=-0.25-index, rank=1, decoded_token=text),
                         500+index:SampleLogprob(logprob=-2.25-index, rank=2, decoded_token='ALT')}]
            output = CompletionOutput(index=0, text=text, token_ids=[100+index],
                                      cumulative_logprob=0.0, logprobs=logprobs if with_logprobs else None,
                                      finish_reason=finish_reason if index == len(chunks)-1 else None)
            piece = RequestOutput(request_id=request.request_id, prompt='SYNTHETIC',
                                  prompt_token_ids=[1, 2], prompt_logprobs=None,
                                  outputs=[output], finished=index == len(chunks)-1,
                                  num_cached_tokens=1)
            target_context.append_output(piece)
            yield target_context

    raw_events = []
    async def native_events():
        async for event in serving.responses_stream_generator(request, params, results(), context,
                                                              'DeepSeek-Flash', tokenizer, None, created_time=1):
            raw_events.append(event)
            yield event

    # Use the installed native HTTP router's actual SSE serializer too.
    wire = ''.join([chunk async for chunk in _convert_stream_to_sse_events(native_events())])
    wire_events = []
    for frame in wire.strip().split('\n\n'):
        event_line, data_line = frame.split('\n', 1)
        event_type = event_line.removeprefix('event: ')
        data = json.loads(data_line.removeprefix('data: '))
        assert data['type'] == event_type
        wire_events.append(data)
    events = [event.model_dump(mode='json', by_alias=True) for event in raw_events]
    assert events == wire_events, 'Native SSE serializer changed event data'
    assert [event['sequence_number'] for event in events] == list(range(len(events)))
    added = [event['item'] for event in events if event['type']=='response.output_item.added']
    done = [event['item'] for event in events if event['type']=='response.output_item.done']
    if case.get('expected_shape_error'):
        assert events[-1]['type'] == 'error', 'Output shape mismatch was not reported as an error event'
        assert events[-1]['code'] == 'server_error'
        assert events[-1]['message'] == 'Stream and final response output items disagree'
        assert not any(event['type'] in ['response.completed','response.incomplete'] for event in events)
        stored = serving.response_store[request.request_id].model_dump(mode='json', by_alias=True)
        assert stored['status'] == 'failed', 'Stored response must not claim completion after shape failure'
        assert stored['error'] == {'code':'server_error','message':events[-1]['message']}
        signal = asyncio.Event()
        serving.event_store[request.request_id] = (deque(raw_events), signal)
        async def failed_replay():
            return [event.model_dump(mode='json', by_alias=True) async for event in serving.responses_background_stream_generator(request.request_id)]
        assert await asyncio.wait_for(failed_replay(), timeout=1.0) == events
        return {'name':case['name'], 'status':'PASS', 'source':'synthetic CompletionOutput; not model output',
                'terminal_event_type':'error', 'response_status':'failed',
                'shape_mismatch_reported_as_native_error_event':True,
                'background_replay_terminated':True, 'stored_status_failed':True,
                'event_types':[event['type'] for event in events]}
    final = events[-1]['response']
    expected_terminal = 'response.incomplete' if finish_reason == 'length' else 'response.completed'
    expected_status = 'incomplete' if finish_reason == 'length' else 'completed'
    assert final['status'] == expected_status
    assert final['usage']['input_tokens'] == 2
    assert final['usage']['input_tokens_details']['cached_tokens'] == 1
    assert final['usage']['output_tokens'] == len(chunks)
    assert final['usage']['total_tokens'] == 2+len(chunks)
    stored = serving.response_store[request.request_id].model_dump(mode='json', by_alias=True)
    assert stored['usage'] == final['usage']
    assert stored['output'] == final['output'], 'Store did not retain serialized stream identities'
    assert events[-1]['type'] == expected_terminal, 'Terminal event type does not match response status'
    assert protocol_signature(done) == protocol_signature(final['output']), 'Final output identity or tool/content semantics differ from emitted done items'
    assert [item['id'] for item in added] == [item['id'] for item in done], 'Item identity changed while streaming'
    if finish_reason == 'length':
        assert final['incomplete_details']['reason'] == 'max_output_tokens'
    delta_text = ''.join(event['delta'] for event in events if event['type']=='response.output_text.delta')
    final_text = ''.join(part['text'] for item in final['output'] if item['type']=='message'
                         for part in item['content'] if part['type']=='output_text')
    assert delta_text == final_text
    if 'expected_text' in case:
        assert final_text == case['expected_text']
    if not include_reasoning:
        assert all(item['type'] != 'reasoning' for item in done + final['output'])
    calls = [item for item in done if item['type']=='function_call']
    for call, task in zip(calls, case.get('expected_tasks', [])):
        assert call['name'] == 'spawn_agent'
        assert json.loads(call['arguments']) == {'task':task}
        assert call.get('namespace') == case.get('expected_namespace')
    assert len(calls) == len(case.get('expected_tasks', []))

    # Native full generation is the metadata reference. Only identities may differ.
    full_request = request.model_copy(update={'stream':False, 'store':False})
    full_context = make_context(full_request)
    reference = await serving.responses_full_generator(full_request, params, results(full_context), full_context,
                                                      'DeepSeek-Flash', tokenizer, None, created_time=1)
    reference = reference.model_dump(mode='json', by_alias=True)
    assert without_identity(final['output']) == without_identity(reference['output']), 'Native nonidentity output metadata changed'
    assert final['usage'] == reference['usage'], 'Native usage or cache metadata changed'
    assert final['status'] == reference['status'] and final['incomplete_details'] == reference['incomplete_details']
    assert serving.response_store[request.request_id].model_dump(mode='json', by_alias=True)['output'] == final['output'], 'Nonstored reference run modified the stored streamed response'
    final_logprobs = [part.get('logprobs') for item in final['output'] if item['type']=='message' for part in item['content']]
    if with_logprobs:
        assert len(final_logprobs) == 1 and len(final_logprobs[0]) == len(chunks)
        for index, row in enumerate(final_logprobs[0]):
            assert row['token'] == chunks[index] and row['logprob'] == -0.25-index
            assert row['bytes'] == list(chunks[index].encode('utf-8'))
            assert len(row['top_logprobs']) == 2 and row['top_logprobs'][1]['token'] == 'ALT'
        assert any(event.get('logprobs') for event in events if event['type']=='response.output_text.delta'), 'Native stream logprobs are missing'

    # An incomplete response must also terminate the background replay stream.
    signal = asyncio.Event()
    serving.event_store[request.request_id] = (deque(raw_events), signal)
    async def replay_background():
        return [event.model_dump(mode='json', by_alias=True) async for event in serving.responses_background_stream_generator(request.request_id)]
    replayed = await asyncio.wait_for(replay_background(), timeout=1.0)
    assert replayed == events, 'Background replay changed serialized events'

    return {'name':case['name'], 'status':'PASS', 'source':'synthetic CompletionOutput; not model output',
            'native_parser':native_parser, 'thinking':thinking, 'include_reasoning':include_reasoning,
            'engine_finish_reason':finish_reason, 'event_count':len(events),
            'terminal_event_type':events[-1]['type'], 'response_status':final['status'],
            'incomplete_details':final['incomplete_details'], 'usage':final['usage'],
            'added_ids':[item['id'] for item in added], 'done_ids':[item['id'] for item in done],
            'completed_ids':[item['id'] for item in final['output']],
            'done_call_ids':[item['call_id'] for item in calls],
            'completed_call_ids':[item['call_id'] for item in final['output'] if item['type']=='function_call'],
            'done_final_protocol_identity_and_semantics_match':protocol_signature(done)==protocol_signature(final['output']),
            'native_nonidentity_output_metadata_preserved':without_identity(final['output'])==without_identity(reference['output']),
            'requested_logprobs_preserved':with_logprobs,
            'final_logprob_token_count':sum(len(value or []) for value in final_logprobs),
            'terminal_matches_status':events[-1]['type']==expected_terminal,
            'background_replay_terminated':True,
            'stored_output_equals_final_output':stored['output']==final['output'],
            'visible_text_characters':len(final_text), 'visible_text_sha256':hashlib.sha256(final_text.encode()).hexdigest(),
            'wire_sha256':hashlib.sha256(wire.encode()).hexdigest(),
            'wire_utf8_valid':wire.encode('utf-8').decode('utf-8')==wire,
            'event_types':[event['type'] for event in events]}


def verify_runtime_sources(project):
    project = project.resolve(strict=True)
    manifest_path = project/'manifests/r39-overlay-provenance.json'
    manifest = json.loads(manifest_path.read_text())
    required = {
        'vllm/tokenizers/deepseek_v41.py',
        'vllm/entrypoints/anthropic/serving.py',
        'vllm/entrypoints/openai/responses/protocol.py',
        'vllm/entrypoints/openai/responses/serving.py',
        'dsv41_guard.py',
    }
    rows = manifest['files']
    assert len(rows) == len(required), 'Expected five reviewed protocol source overlays'
    found = set()
    result = []
    for row in rows:
        relative = str(row['overlay_path']).removeprefix('source/r39-overlay/')
        assert relative in required and relative not in found, 'Unexpected or repeated protocol source path'
        found.add(relative)
        installed = Path(row['site_packages_path'])
        expected_path = Path('/usr/local/lib/python3.12/dist-packages')/relative
        assert installed == expected_path, 'Unexpected installed protocol source path'
        actual = hashlib.sha256(installed.read_bytes()).hexdigest()
        assert actual == row['patched_sha256'], 'Installed protocol source hash mismatch: '+relative
        assert hashlib.sha256((project/row['overlay_path']).read_bytes()).hexdigest() == actual, 'Reviewed and installed sources differ: '+relative
        result.append({'relative_path':relative,'installed_sha256':actual})
    assert found == required
    interfaces = [name for _,name in socket.if_nameindex()]
    assert interfaces == ['lo'], 'CPU protocol checks require Docker --network none'
    assert not vllm.platforms.current_platform.is_cuda(), 'CPU protocol checks must not select a CUDA platform'
    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    report = {'scope':'Real pinned R2 Responses generator, DeepSeekV41Parser and native SSE router serialization with V4 overlays; synthetic engine deltas only',
              'status':'RUNNING', 'gpu_executed':False, 'network_used':False, 'weights_loaded':False,
              'real_inference_executed':False, 'app_subagent_end_to_end':'NOT_RUN',
              'current_app_failure_root_cause':'NOT_PROVEN', 'assertion_mode':'STRICT_FIXED_PROTOCOL',
              'raw_model_or_reasoning_content_saved':False, 'checks':[]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report['runtime_sources'] = verify_runtime_sources(args.project)
    except Exception as error:
        report.update(status='FAIL', error=type(error).__name__+': '+str(error))
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({'status':'FAIL', 'stage':'runtime-source-verification', 'error':report['error']}),flush=True)
        return 1
    greeting = '你好，很高兴见到你！'
    cases = [
        {'name':'plain-UTF8-greeting','chunks':['你好','，很高兴见到你！'],'native_parser':False,'expected_text':greeting},
        {'name':'single-quote-pass-through','chunks':['“'],'native_parser':False,'expected_text':'“'},
        {'name':'native-off-UTF8-greeting','chunks':['你好','，很高兴见到你！'],'expected_text':greeting},
        {'name':'native-high-reasoning-then-greeting','chunks':['先想一下','</think>',greeting],'thinking':True,'expected_text':greeting},
        {'name':'native-single-function','chunks':['<｜DSML｜ calls>'+invocation('spawn_agent','Greet')+'</｜DSML｜ calls>'],
         'tools':[FUNCTION],'expected_tasks':['Greet']},
        {'name':'native-high-reasoning-hidden','chunks':['先想一下','</think>',greeting],'thinking':True,'include_reasoning':False,'expected_text':greeting},
        {'name':'native-reasoning-token-limit','chunks':['先想一下，预算尚未足够生成最终回答'],'thinking':True,'finish_reason':'length','expected_text':''},
        {'name':'native-two-functions-namespace','chunks':['<｜DSML｜ calls>',invocation('functions__spawn_agent','First'),invocation('functions__spawn_agent','Second'),'</｜DSML｜ calls>'],
         'tools':[NAMESPACE],'expected_tasks':['First','Second'],'expected_namespace':'functions'},
        {'name':'native-content-token-limit','chunks':['你好','，很高兴见到你！'],'finish_reason':'length','expected_text':greeting},
        {'name':'native-text-before-function-preserves-order','chunks':['我来分派。','<｜DSML｜ calls>',invocation('spawn_agent','Greet'),'</｜DSML｜ calls>'],
         'tools':[FUNCTION],'expected_tasks':['Greet'],'expected_text':'我来分派。'},
        {'name':'native-requested-output-logprobs','chunks':['你好','，很高兴见到你！'],'expected_text':greeting,'logprobs':True},
        {'name':'native-fallback-empty-shape-error','chunks':[''],'native_parser':False,'expected_shape_error':True},
    ]
    for case in cases:
        try:
            row = await run_case(case)
        except Exception as error:
            row = {'name':case['name'],'status':'FAIL','error':type(error).__name__+': '+str(error)}
        report['checks'].append(row)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({key:row[key] for key in ['name','status','done_final_protocol_identity_and_semantics_match','native_nonidentity_output_metadata_preserved','terminal_matches_status'] if key in row},ensure_ascii=False),flush=True)
    report['status'] = 'PASS' if all(row['status']=='PASS' for row in report['checks']) else 'FAIL'
    report['failed_cases'] = sum(row['status']!='PASS' for row in report['checks'])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':len(report['checks']),'failed_cases':report['failed_cases']}),flush=True)
    return 0 if report['status']=='PASS' else 1


raise SystemExit(asyncio.run(main()))
