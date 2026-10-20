"""Offline CPU protocol checks using real R2 parser/generator/router code.

All engine output is fixed synthetic fixture data; no weights, GPU or network.
This is not a model-quality, performance or App subagent acceptance test.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import socket
import traceback

import vllm.platforms
from vllm.platforms.cpu import CpuPlatform
vllm.platforms._current_platform = CpuPlatform()
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.logprobs import Logprob as SampleLogprob
from vllm.sampling_params import SamplingParams, RequestOutputKind
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.openai.responses.api_router import _convert_stream_to_sse_events
from vllm.parser.deepseek_v41 import DeepSeekV41Parser
from vllm.exceptions import GenerationError
from vllm.entrypoints.serve.engine.protocol import ErrorResponse,ErrorInfo


class StubTokenizer:
    def get_vocab(self): return {}
    def decode(self, token_ids, **kwargs): return ''


class FixtureTokenizer:
    """Publisher-tokenizer-derived fixed frames, not an alternate tokenizer."""
    def __init__(self, fixture):
        self.fixture = fixture
        self.all_special_tokens = fixture['all_special_tokens']
        self.all_special_ids = fixture['all_special_ids']
    def get_vocab(self): return self.fixture['vocab']
    def decode(self, token_ids, **kwargs):
        return ''.join(self.fixture['decoded_tokens'][str(value)] for value in token_ids)


FUNCTION = {'type':'function', 'name':'compatibility_probe_never_execute', 'parameters':{
    'type':'object', 'properties':{'task':{'type':'string'}}, 'required':['task']}}
NAMESPACE = {'type':'namespace', 'name':'functions', 'description':'Synthetic protocol fixture', 'tools':[FUNCTION]}


def invocation(task='Greet', namespace=False):
    name = ('functions__' if namespace else '') + FUNCTION['name']
    return (f'<｜DSML｜ invoke name="{name}">'
            f'<｜DSML｜ parameter name="task" string="true">{task}</｜DSML｜ parameter>'
            '</｜DSML｜ invoke>')


def tool(task='Greet', namespace=False):
    return '<｜DSML｜ calls>' + invocation(task, namespace) + '</｜DSML｜ calls>'


def semantics(items):
    result={'message':'', 'reasoning':'', 'function_call':[]}
    for item in items:
        if item['type']=='function_call':
            result['function_call'].append({key:item.get(key) for key in ['name','namespace','arguments']})
        else:
            assert item['type'] in ['message','reasoning']
            result[item['type']]+=''.join(part['text'] for part in item.get('content',[]))
    return result


def identities(items):
    return [(item['type'],item['id'],item.get('call_id')) for item in items]


def nonidentity(items):
    result=copy.deepcopy(items)
    for item in result:
        item.pop('id',None)
        if item['type']=='function_call': item.pop('call_id',None)
    return result


def make_fixture(case, *, streaming_guard=True):
    thinking=case.get('thinking',False)
    with_probs=case.get('logprobs',False)
    request=ResponsesRequest(
        model='DeepSeek-Flash', input='SYNTHETIC protocol fixture',
        stream=True, store=True, include_reasoning=case.get('include_reasoning',True),
        tools=case.get('tools',[]), max_output_tokens=4096,
        **({'include':['message.output_text.logprobs'],'top_logprobs':2} if with_probs else {}),
    )
    serving=object.__new__(OpenAIServingResponses)
    serving.use_harmony=False
    serving.parser=DeepSeekV41Parser if case.get('native_parser',True) else None
    serving.enable_log_outputs=False
    serving.request_logger=None
    serving.enable_auto_tools=True
    serving.tool_server=None
    serving.return_tokens_as_token_ids=False
    serving.response_store_lock=asyncio.Lock()
    serving.response_store={}
    serving.event_store={}
    if case.get('final_error_fixture'):
        async def failed_full_generator(*args,**kwargs):
            return ErrorResponse(error=ErrorInfo(message='Synthetic final response error',type='server_error',code=500))
        serving.responses_full_generator=failed_full_generator
    if streaming_guard:
        def forbid_full_reparse(*args, **kwargs):
            raise AssertionError('Streaming must not invoke the independent full-output parser')
        serving._make_response_output_items = forbid_full_reparse
    tokenizer=FixtureTokenizer(case['_tokenizer_fixture']) if '_tokenizer_fixture' in case else StubTokenizer()
    context=SimpleContext(parser_cls=serving.parser,tokenizer=tokenizer,request=request,
                          chat_template_kwargs={'thinking':thinking,'reasoning_effort':'high' if thinking else 'none'})
    params=request.to_sampling_params(4096,{})
    assert params.output_kind == RequestOutputKind.DELTA
    async def results():
        if case.get('error_before_output'):
            raise GenerationError('Synthetic engine failure before first output')
        for index,text in enumerate(case['chunks']):
            probs=[{100+index:SampleLogprob(logprob=-0.25-index,rank=1,decoded_token=text),
                    500+index:SampleLogprob(logprob=-2.25-index,rank=2,decoded_token='ALT')}]
            frame_ids=case.get('chunk_token_ids',[[100+i] for i in range(len(case['chunks']))])[index]
            output=CompletionOutput(
                index=0,text=text,token_ids=frame_ids,cumulative_logprob=0.0,
                logprobs=probs if with_probs else None,
                finish_reason=case.get('finish_reason','stop') if index==len(case['chunks'])-1 else None,
            )
            piece=RequestOutput(
                request_id=request.request_id,prompt='SYNTHETIC',prompt_token_ids=case.get('prompt_token_ids',[1,2]),
                prompt_logprobs=None,outputs=[output],finished=index==len(case['chunks'])-1,
                num_cached_tokens=1,
            )
            context.append_output(piece)
            yield context
    return request,serving,context,params,tokenizer,results


async def run_case(case, wires):
    request,serving,context,params,tokenizer,results=make_fixture(case)
    raw=[]
    async def native_events():
        async for event in serving.responses_stream_generator(
            request,params,results(),context,'DeepSeek-Flash',tokenizer,None,created_time=1,
        ):
            raw.append(event)
            yield event
    wire=''.join([chunk async for chunk in _convert_stream_to_sse_events(native_events())])
    wires[case['name']]=wire
    events=[event.model_dump(mode='json',by_alias=True) for event in raw]
    wire_events=[]
    for frame in wire.strip().split('\n\n'):
        event_line,data_line=frame.split('\n',1)
        data=json.loads(data_line.removeprefix('data: '))
        assert event_line=='event: '+data['type']
        wire_events.append(data)
    assert wire_events==events, 'Native SSE serializer changed event data'
    assert [event['sequence_number'] for event in events]==list(range(len(events)))
    added=[event['item'] for event in events if event['type']=='response.output_item.added']
    done=[event['item'] for event in events if event['type']=='response.output_item.done']
    terminal=events[-1]
    final=terminal['response']
    expected_status='failed' if case.get('failure_code') else 'incomplete' if case.get('finish_reason')=='length' else 'completed'
    assert terminal['type']=='response.'+expected_status, 'Incorrect terminal event: '+terminal['type']
    assert final['status']==expected_status
    expected_input=0 if case.get('error_before_output') else len(case.get('prompt_token_ids',[1,2]))
    expected_generated=0 if case.get('error_before_output') else sum(len(ids) for ids in case.get('chunk_token_ids',[[100+i] for i in range(len(case['chunks']))]))
    assert final['usage']['input_tokens']==expected_input
    assert final['usage']['input_tokens_details']['cached_tokens']==(0 if case.get('error_before_output') else 1)
    assert final['usage']['output_tokens']==expected_generated
    assert final['usage']['total_tokens']==expected_input+expected_generated
    stored=serving.response_store[request.request_id].model_dump(mode='json',by_alias=True)
    assert stored==final, 'Stored response differs from emitted terminal response'
    assert [event['output_index'] for event in events if event['type']=='response.output_item.added']==list(range(len(added)))
    assert [item['id'] for item in done]==[item['id'] for item in added[:len(done)]]
    assert identities(done)==identities(final['output']), 'Done/terminal identities differ'
    if not case.get('failure_code'):
        assert len(added)==len(done)
    assert len(set(item['id'] for item in added))==len(added)
    metadata_preserved=None
    if case.get('failure_code'):
        assert final['error']['code']=='server_error'
        assert final['error']['message'], 'Failure reason must not be empty'
        if case['failure_code'] not in ['generation_error','final_generation_error']:
            assert case['failure_code'] in final['error']['message']
        assert not any(event['type']=='response.completed' for event in events)
    else:
        assert identities(done)==identities(final['output']), 'Done/terminal identities differ'
        assert done==final['output'], 'Done/terminal output fields differ'
        if done:
            assert done[-1]['status']==('incomplete' if expected_status=='incomplete' else 'completed')
        delta_text=''.join(event['delta'] for event in events if event['type']=='response.output_text.delta')
        assert delta_text==semantics(final['output'])['message']
        if 'expected_text' in case: assert delta_text==case['expected_text']
        delta_reasoning=''.join(event['delta'] for event in events if event['type']=='response.reasoning_text.delta')
        assert delta_reasoning==semantics(done)['reasoning'], 'Reasoning delta/done mismatch'
        if 'expected_reasoning' in case: assert delta_reasoning==case['expected_reasoning']
        if not case.get('include_reasoning',True):
            assert all(item['type']!='reasoning' for item in done+final['output'])
        if expected_status=='incomplete':
            assert final['incomplete_details']['reason']=='max_output_tokens'

        # The separate JSON path remains native. Its segmentation/whitespace
        # is deliberately not a streaming acceptance oracle (upstream #59307).
        ref_request,ref_serving,ref_context,ref_params,ref_tokenizer,ref_results=make_fixture(case,streaming_guard=False)
        ref=await ref_serving.responses_full_generator(
            ref_request,ref_params,ref_results(),ref_context,'DeepSeek-Flash',ref_tokenizer,None,created_time=1,
        )
        reference=ref.model_dump(mode='json',by_alias=True)
        for key in ['input_tokens','output_tokens','total_tokens','input_tokens_details']:
            assert final['usage'][key]==reference['usage'][key], 'Native token/cache accounting changed'
        assert final['status']==reference['status']
        assert final['incomplete_details']==reference['incomplete_details']
        # Check observed per-message token probabilities, including bytes.
        for item in done:
            if item['type']!='message': continue
            observed=[prob for event in events if event['type']=='response.output_text.delta'
                      and event['item_id']==item['id'] for prob in event.get('logprobs',[])]
            actual=[prob for part in item.get('content',[]) for prob in (part.get('logprobs') or [])]
            assert len(actual)==len(observed), 'Lost or duplicated streamed logprobs'
            for full,delta in zip(actual,observed):
                assert full['token']==delta['token'] and full['logprob']==delta['logprob']
                assert full['bytes']==list(delta['token'].encode('utf-8',errors='replace'))
                assert len(full['top_logprobs'])==len(delta.get('top_logprobs') or [])
                for full_top,delta_top in zip(full['top_logprobs'],delta.get('top_logprobs') or []):
                    assert full_top['token']==delta_top['token'] and full_top['logprob']==delta_top['logprob']
                    assert full_top['bytes']==list(delta_top['token'].encode('utf-8',errors='replace'))
        if case.get('logprobs') and not case.get('include_reasoning',True):
            assert not any(event.get('logprobs') for event in events)
        metadata_preserved=True
    calls=[item for item in final['output'] if item['type']=='function_call']
    if not case.get('failure_code'):
        arguments=case.get('expected_arguments',[{'task':task} for task in case.get('expected_tasks',[])])
        assert len(calls)==len(arguments)
        for call,expected in zip(calls,arguments):
            assert call['name']==case.get('function_name',FUNCTION['name'])
            assert call.get('namespace')==case.get('expected_namespace')
            assert json.loads(call['arguments'])==expected
            assert ''.join(event['delta'] for event in events if event['type']=='response.function_call_arguments.delta' and event['item_id']==call['id'])==call['arguments']
            args_done=[event for event in events if event['type']=='response.function_call_arguments.done' and event['item_id']==call['id']]
            assert len(args_done)==1 and args_done[0]['arguments']==call['arguments']
        if calls and context.response_parser is not None:
            native_ids=[slot.id for slot in context.response_parser._tool_slots if slot.name_sent]
            assert [call['call_id'] for call in calls]==native_ids, 'Native parser call_id was replaced'

    # Run the real background event producer, then replay its terminal event.
    bg_request,bg_serving,bg_context,bg_params,bg_tokenizer,bg_results=make_fixture(case)
    await asyncio.wait_for(bg_serving._run_background_request_stream(
        bg_request,bg_params,bg_results(),bg_context,'DeepSeek-Flash',bg_tokenizer,None,created_time=1,
    ),timeout=2.0)
    bg_events=await asyncio.wait_for(_collect_background(bg_serving,bg_request.request_id),timeout=1.0)
    assert bg_events[-1]['type']==terminal['type']
    assert bg_events[-1]['response']['status']==expected_status
    assert bg_events[-1]['response']['usage']==final['usage']
    assert bg_serving.response_store[bg_request.request_id].model_dump(mode='json',by_alias=True)==bg_events[-1]['response']
    return {'name':case['name'],'status':'PASS','fixture':('fault-injected native final ErrorResponse; native streaming parser/generator/router'
            if case.get('final_error_fixture') else 'synthetic engine deltas; native R2 parser/generator/router'),
            'terminal_event_type':terminal['type'],'response_status':expected_status,
            'usage':final['usage'],'error':final.get('error'),'event_count':len(events),
            'output_types':[item['type'] for item in final['output']],
            'stream_item_metadata_preserved':metadata_preserved,'second_full_parse_forbidden':not case.get('final_error_fixture'),
            'fixture_tokenizer_sha256':case.get('_tokenizer_fixture',{}).get('tokenizer_sha256'),
            'stored_response_matches_terminal':True,'background_producer_and_replay_terminated':True,
            'wire_sha256':hashlib.sha256(wire.encode()).hexdigest()}


async def _collect_background(serving,request_id):
    return [event.model_dump(mode='json',by_alias=True)
            async for event in serving.responses_background_stream_generator(request_id)]


def verify_sources(project):
    manifest=json.loads((project/'manifests/r39-overlay-provenance.json').read_text())
    required={'vllm/tokenizers/deepseek_v41.py','vllm/entrypoints/anthropic/serving.py',
              'vllm/entrypoints/openai/responses/protocol.py','vllm/entrypoints/openai/responses/serving.py',
              'vllm/entrypoints/openai/responses/context.py','vllm/entrypoints/openai/responses/streaming_events.py','dsv41_guard.py'}
    assert len(manifest['files'])==7
    found=set()
    sources=[]
    for row in manifest['files']:
        relative=str(row['overlay_path']).removeprefix('source/r39-overlay/')
        assert relative in required and relative not in found
        found.add(relative)
        installed=Path('/usr/local/lib/python3.12/dist-packages')/relative
        assert str(installed)==row['site_packages_path']
        sha=hashlib.sha256(installed.read_bytes()).hexdigest()
        assert sha==row['patched_sha256']==hashlib.sha256((project/row['overlay_path']).read_bytes()).hexdigest()
        sources.append({'path':relative,'sha256':sha})
    assert found==required
    assert [name for _,name in socket.if_nameindex()]==['lo'], 'Requires Docker --network none'
    assert not vllm.platforms.current_platform.is_cuda()
    return sources


def processor_zero_delta_fixture():
    from vllm.entrypoints.generate.base.protocol import DeltaToolCall, DeltaFunctionCall
    from vllm.entrypoints.openai.responses.streaming_events import SimpleStreamingEventProcessor, _StateType
    processor=SimpleStreamingEventProcessor(tools=[])
    call=DeltaToolCall(index=0,id='call_fixed_zero_args',type='function',
                       function=DeltaFunctionCall(name='zero_args'))
    opened=processor.open(_StateType.TOOL_CALL,call)
    closed=processor.close_current()
    done=[event for event in closed if event.type=='response.function_call_arguments.done']
    assert len(done)==1 and done[0].arguments==''
    assert closed[-1].type=='response.output_item.done'
    assert closed[-1].item.call_id==call.id
    assert opened[0].item.id==closed[-1].item.id
    assert processor.output_items==[closed[-1].item]
    return {'name':'zero-argument-done-without-delta','status':'PASS',
            'fixture':'Native SimpleStreamingEventProcessor objects only; no engine/model inference',
            'zero_argument_done_always_emitted':True,'native_call_id_preserved':True}


async def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--wire-output',type=Path,help='Optional fixed synthetic fixtures only; never actual model data')
    args=ap.parse_args()
    report={'scope':'Fixed R2 native parser/generator/SSE router CPU checks with synthetic engine deltas; explicitly identified metadata unit fixtures and final-error fault injection',
            'status':'RUNNING','gpu_executed':False,'network_used':False,'weights_loaded':False,
            'real_inference_executed':False,'app_subagent_end_to_end':'NOT_RUN','current_target_app_recovery':'NOT_RUN',
            'raw_model_or_reasoning_content_saved':False,'checks':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try: report['runtime_sources']=verify_sources(args.project)
    except Exception as e:
        report.update(status='FAIL',error=type(e).__name__+': '+str(e))
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');return 1
    greeting='你好，很高兴见到你！'
    repeated=['想一下','</think>','你好','<think>','再想','</think>','再见']
    cases=[
      {'name':'plain-UTF8-greeting','chunks':['你好','，很高兴见到你！'],'native_parser':False,'expected_text':greeting},
      {'name':'single-quote-pass-through','chunks':['“'],'native_parser':False,'expected_text':'“'},
      {'name':'native-off-UTF8-greeting','chunks':['你好','，很高兴见到你！'],'expected_text':greeting},
      {'name':'native-high-reasoning-then-greeting','chunks':['先想一下','</think>',greeting],'thinking':True,'expected_text':greeting},
      {'name':'native-single-function','chunks':[tool()],'tools':[FUNCTION],'expected_tasks':['Greet']},
      {'name':'native-zero-argument-function','chunks':['<｜DSML｜ calls><｜DSML｜ invoke name="'+FUNCTION['name']+'"></｜DSML｜ invoke></｜DSML｜ calls>'],
       'tools':[dict(FUNCTION,parameters={'type':'object','properties':{}})],'expected_arguments':[{}]},
      {'name':'native-high-reasoning-hidden','chunks':['先想一下','</think>',greeting],'thinking':True,'include_reasoning':False,'expected_text':greeting},
      {'name':'native-reasoning-token-limit','chunks':['先想一下'],'thinking':True,'finish_reason':'length','expected_text':''},
      {'name':'native-two-functions-namespace','chunks':['<｜DSML｜ calls>',invocation('First',True),invocation('Second',True),'</｜DSML｜ calls>'],
       'tools':[NAMESPACE],'expected_tasks':['First','Second'],'expected_namespace':'functions'},
      {'name':'native-content-token-limit','chunks':['你好','，很高兴见到你！'],'finish_reason':'length','expected_text':greeting},
      {'name':'native-text-before-function','chunks':['我来分派。',tool()],'tools':[FUNCTION],'expected_tasks':['Greet'],'expected_text':'我来分派。'},
      {'name':'native-requested-output-logprobs','chunks':['你好','，很高兴见到你！'],'logprobs':True,'expected_text':greeting},
      {'name':'fallback-empty-output','chunks':[''],'native_parser':False,'expected_text':''},
      {'name':'repeat-thinking-segments','chunks':repeated,'thinking':True,'expected_text':'你好再见'},
      {'name':'repeat-thinking-hidden','chunks':repeated,'thinking':True,'include_reasoning':False,'expected_text':'你好再见'},
      {'name':'repeat-thinking-token-limit','chunks':repeated,'thinking':True,'finish_reason':'length','expected_text':'你好再见'},
      {'name':'repeat-thinking-before-namespace-functions','chunks':repeated+['<｜DSML｜ calls>',invocation('First',True),invocation('Second',True),'</｜DSML｜ calls>'],
       'thinking':True,'tools':[NAMESPACE],'expected_tasks':['First','Second'],'expected_namespace':'functions','expected_text':'你好再见'},
      {'name':'native-generation-error','chunks':['不能完成'],'finish_reason':'error','failure_code':'generation_error'},
      {'name':'native-generation-error-before-output','chunks':[],'error_before_output':True,'failure_code':'generation_error'},
      {'name':'native-generation-error-after-partial-output','chunks':['想一下','</think>','可见部分','错误结束'],'thinking':True,'finish_reason':'error','failure_code':'generation_error'},
      {'name':'final-error-response-fault-injection','chunks':['固定输出'],'final_error_fixture':True,'failure_code':'final_generation_error'},
      {'name':'segmented-logprobs-stream-source','chunks':repeated,'thinking':True,'logprobs':True,'expected_text':'你好再见'},
      {'name':'segmented-logprobs-hidden','chunks':repeated,'thinking':True,'logprobs':True,'include_reasoning':False,'expected_text':'你好再见'},
    ]
    for size in [1,7]:
        text=''.join(repeated)+tool()
        cases.append({'name':f'repeat-thinking-tool-fragment-{size}','chunks':[text[i:i+size] for i in range(0,len(text),size)],
                      'thinking':True,'tools':[FUNCTION],'expected_tasks':['Greet'],'expected_text':'你好再见'})
    fixture=json.loads((args.project/'tests/fixtures/deepseek-v41-stream-token-cases.json').read_text())
    assert fixture['schema']==1 and len(fixture['cases'])>=30
    report['publisher_tokenizer_sha256']=fixture['tokenizer_sha256']
    for case in fixture['cases']:
        case['_tokenizer_fixture']=fixture
        cases.append(case)
    wires={}
    for case in cases:
        try: row=await run_case(case,wires)
        except Exception as e:
            row={'name':case['name'],'status':'FAIL','error':type(e).__name__+': '+str(e)}
        report['checks'].append(row)
        print(json.dumps({key:row[key] for key in ['name','status','terminal_event_type','error'] if key in row},ensure_ascii=False),flush=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    try: report['checks'].append(processor_zero_delta_fixture())
    except Exception as e: report['checks'].append({'name':'zero-argument-done-without-delta','status':'FAIL','error':type(e).__name__+': '+str(e)})
    report['failed_cases']=sum(row['status']!='PASS' for row in report['checks'])
    report['status']='PASS' if not report['failed_cases'] else 'FAIL'
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if args.wire_output:
        args.wire_output.write_text(json.dumps({'scope':'Fixed synthetic fixtures only; no model data','wires':wires},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':len(report['checks']),'failed_cases':report['failed_cases']}),flush=True)
    return 0 if report['status']=='PASS' else 1


raise SystemExit(asyncio.run(main()))
