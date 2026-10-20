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
from vllm.sampling_params import SamplingParams
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


def make_fixture(case):
    thinking=case.get('thinking',False)
    with_probs=case.get('logprobs',False)
    request=ResponsesRequest(
        model='DeepSeek-Flash', input='SYNTHETIC protocol fixture',
        stream=True, store=True, include_reasoning=case.get('include_reasoning',True),
        tools=case.get('tools',[]),
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
    tokenizer=StubTokenizer()
    context=SimpleContext(parser_cls=serving.parser,tokenizer=tokenizer,request=request,
                          chat_template_kwargs={'thinking':thinking,'reasoning_effort':'high' if thinking else 'none'})
    params=SamplingParams(max_tokens=32,logprobs=2 if with_probs else None)
    async def results():
        if case.get('error_before_output'):
            raise GenerationError('Synthetic engine failure before first output')
        for index,text in enumerate(case['chunks']):
            probs=[{100+index:SampleLogprob(logprob=-0.25-index,rank=1,decoded_token=text),
                    500+index:SampleLogprob(logprob=-2.25-index,rank=2,decoded_token='ALT')}]
            output=CompletionOutput(
                index=0,text=text,token_ids=[100+index],cumulative_logprob=0.0,
                logprobs=probs if with_probs else None,
                finish_reason=case.get('finish_reason','stop') if index==len(case['chunks'])-1 else None,
            )
            piece=RequestOutput(
                request_id=request.request_id,prompt='SYNTHETIC',prompt_token_ids=[1,2],
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
    expected_input=0 if case.get('error_before_output') else 2
    expected_generated=0 if case.get('error_before_output') else len(case['chunks'])
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
        assert semantics(done)==semantics(final['output']), 'Done/terminal text or tool semantics differ'
        delta_text=''.join(event['delta'] for event in events if event['type']=='response.output_text.delta')
        assert delta_text==semantics(final['output'])['message']
        if 'expected_text' in case: assert delta_text==case['expected_text']
        if not case.get('include_reasoning',True):
            assert all(item['type']!='reasoning' for item in done+final['output'])
        if expected_status=='incomplete':
            assert final['incomplete_details']['reason']=='max_output_tokens'

        # Independent nonstream native response provides usage/metadata reference.
        ref_request,ref_serving,ref_context,ref_params,ref_tokenizer,ref_results=make_fixture(case)
        ref=await ref_serving.responses_full_generator(
            ref_request,ref_params,ref_results(),ref_context,'DeepSeek-Flash',ref_tokenizer,None,created_time=1,
        )
        reference=ref.model_dump(mode='json',by_alias=True)
        assert final['usage']==reference['usage']
        assert final['status']==reference['status']
        assert final['incomplete_details']==reference['incomplete_details']
        assert semantics(final['output'])==semantics(reference['output'])
        if [item['type'] for item in final['output']]==[item['type'] for item in reference['output']]:
            assert nonidentity(final['output'])==nonidentity(reference['output']), 'Native metadata changed on ordinary path'
        else:
            function_index=0
            for item in final['output']:
                templates=[source for source in reference['output'] if source['type']==item['type']]
                template=templates[function_index] if item['type']=='function_call' else templates[0]
                if item['type']=='function_call': function_index+=1
                wanted=copy.deepcopy(template)
                wanted['id']=item['id']
                if item['type']=='function_call':
                    wanted['call_id']=item['call_id']
                else:
                    wanted['content'][0]['text']=item['content'][0]['text']
                assert item==wanted, 'Segmented output discarded native metadata'
        metadata_preserved=True
    calls=[item for item in final['output'] if item['type']=='function_call']
    if not case.get('failure_code'):
        assert len(calls)==len(case.get('expected_tasks',[]))
        for call,task in zip(calls,case.get('expected_tasks',[])):
            assert call['name']==FUNCTION['name']
            assert call.get('namespace')==case.get('expected_namespace')
            assert json.loads(call['arguments'])=={'task':task}

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
            'native_nonidentity_metadata_preserved':metadata_preserved,
            'stored_response_matches_terminal':True,'background_producer_and_replay_terminated':True,
            'wire_sha256':hashlib.sha256(wire.encode()).hexdigest()}


async def _collect_background(serving,request_id):
    return [event.model_dump(mode='json',by_alias=True)
            async for event in serving.responses_background_stream_generator(request_id)]


def verify_sources(project):
    manifest=json.loads((project/'manifests/r39-overlay-provenance.json').read_text())
    required={'vllm/tokenizers/deepseek_v41.py','vllm/entrypoints/anthropic/serving.py',
              'vllm/entrypoints/openai/responses/protocol.py','vllm/entrypoints/openai/responses/serving.py','dsv41_guard.py'}
    assert len(manifest['files'])==5
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


def logprob_segmentation_fixture():
    """Unit-level metadata fixture complements native generator logprob cases."""
    from openai.types.responses import ResponseOutputMessage,ResponseOutputText,ResponseReasoningItem
    from openai.types.responses.response_reasoning_item import Content
    from openai.types.responses.response_output_text import Logprob,LogprobTopLogprob
    from vllm.entrypoints.openai.responses.serving import _reconcile_streamed_output
    def probability(token,value):
        return Logprob(token=token,logprob=value,bytes=list(token.encode()),top_logprobs=[
            LogprobTopLogprob(token=token,logprob=value,bytes=list(token.encode()))])
    probs=[probability('你好',-.5),probability('再见',-.7)]
    native=[ResponseReasoningItem(id='native-r',type='reasoning',summary=[],status=None,
                                  content=[Content(type='reasoning_text',text='一次二次')]),
            ResponseOutputMessage(id='native-m',type='message',role='assistant',status='completed',
                                  content=[ResponseOutputText(type='output_text',text='你好再见',annotations=[],logprobs=probs)])]
    streamed=[ResponseReasoningItem(id='r1',type='reasoning',summary=[],status='completed',content=[Content(type='reasoning_text',text='一次')]),
              ResponseOutputMessage(id='m1',type='message',role='assistant',status='completed',content=[ResponseOutputText(type='output_text',text='你好',annotations=[])]),
              ResponseReasoningItem(id='r2',type='reasoning',summary=[],status='completed',content=[Content(type='reasoning_text',text='二次')]),
              ResponseOutputMessage(id='m2',type='message',role='assistant',status='completed',content=[ResponseOutputText(type='output_text',text='再见',annotations=[])])]
    before=[p.model_dump() for p in probs]
    result=_reconcile_streamed_output(native,streamed,{'m1':[probs[0]],'m2':[probs[1]]})
    messages=[item for item in result if item.type=='message']
    after=[p.model_dump() for item in messages for p in item.content[0].logprobs]
    assert before==after, 'Segmenting metadata duplicated or lost a native logprob'
    assert [len(item.content[0].logprobs) for item in messages]==[1,1]
    assert [item.id for item in result]==['r1','m1','r2','m2']
    assert [item.status for item in result if item.type=='reasoning']==[None,None]
    assert [p.model_dump() for p in native[1].content[0].logprobs]==before
    # This is a unit fixture, not synthetic engine inference or App acceptance.
    return {'name':'segmented-logprob-metadata-unit-fixture','status':'PASS','fixture':'metadata objects only',
            'native_logprobs_preserved_exactly_once':True,'segment_counts':[1,1]}


def reconciliation_guard_fixtures():
    """Prove unsupported/ambiguous metadata returns a typed failure, not EOF."""
    from openai.types.responses import ResponseOutputMessage,ResponseOutputText,ResponseOutputRefusal
    from vllm.entrypoints.openai.responses.serving import _reconcile_streamed_output,_StreamReconciliationError
    def message(identity,text=None):
        return ResponseOutputMessage(id=identity,type='message',role='assistant',status='completed',content=(
            [] if text is None else [ResponseOutputText(type='output_text',text=text,annotations=[])]))
    refusals=ResponseOutputMessage(id='refusal',type='message',role='assistant',status='completed',
                                  content=[ResponseOutputRefusal(type='refusal',refusal='Synthetic refusal')])
    fixtures=[
        ('missing-native-part',[message('n')],[message('s1',''),message('s2','')], 'missing_native_text_content_part'),
        ('unsupported-refusal-part',[refusals],[message('s1','')], 'unsupported_text_content_part'),
        ('same-types-different-boundaries',[message('n1','a'),message('n2','bc')],
         [message('s1','ab'),message('s2','c')], 'ambiguous_native_text_segments'),
        ('actual-text-mismatch',[message('n1','first')],[message('s1','second')], 'stream_final_semantic_mismatch'),
    ]
    checks=[]
    for name,native,streamed,expected in fixtures:
        try: _reconcile_streamed_output(native,streamed,{})
        except _StreamReconciliationError as error:
            assert error.code==expected
        else: raise AssertionError('Invalid metadata was accepted: '+name)
        checks.append({'name':name,'error_code':expected})
    return {'name':'reconciliation-failure-guards-unit-fixtures','status':'PASS','fixture':'metadata objects only','guards':checks}


async def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--wire-output',type=Path,help='Optional fixed synthetic fixtures only; never actual model data')
    args=ap.parse_args()
    report={'scope':'Fixed R2 native parser/generator/SSE router CPU checks with synthetic engine deltas; explicitly identified metadata unit fixtures and final-error fault injection',
            'status':'RUNNING','gpu_executed':False,'network_used':False,'weights_loaded':False,
            'real_inference_executed':False,'app_subagent_end_to_end':'NOT_RUN','current_target_failure_root_cause':'NOT_PROVEN',
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
      {'name':'segmented-logprobs-unprovable-native','chunks':repeated,'thinking':True,'logprobs':True,'failure_code':'segmented_logprobs_not_losslessly_alignable'},
    ]
    for size in [1,7]:
        text=''.join(repeated)+tool()
        cases.append({'name':f'repeat-thinking-tool-fragment-{size}','chunks':[text[i:i+size] for i in range(0,len(text),size)],
                      'thinking':True,'tools':[FUNCTION],'expected_tasks':['Greet'],'expected_text':'你好再见'})
    wires={}
    for case in cases:
        try: row=await run_case(case,wires)
        except Exception as e:
            row={'name':case['name'],'status':'FAIL','error':type(e).__name__+': '+str(e)}
        report['checks'].append(row)
        print(json.dumps({key:row[key] for key in ['name','status','terminal_event_type','error'] if key in row},ensure_ascii=False),flush=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    try: report['checks'].append(logprob_segmentation_fixture())
    except Exception as e: report['checks'].append({'name':'segmented-logprob-metadata-unit-fixture','status':'FAIL','error':type(e).__name__+': '+str(e)})
    try: report['checks'].append(reconciliation_guard_fixtures())
    except Exception as e: report['checks'].append({'name':'reconciliation-failure-guards-unit-fixtures','status':'FAIL','error':type(e).__name__+': '+str(e)})
    report['failed_cases']=sum(row['status']!='PASS' for row in report['checks'])
    report['status']='PASS' if not report['failed_cases'] else 'FAIL'
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if args.wire_output:
        args.wire_output.write_text(json.dumps({'scope':'Fixed synthetic fixtures only; no model data','wires':wires},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':len(report['checks']),'failed_cases':report['failed_cases']}),flush=True)
    return 0 if report['status']=='PASS' else 1


raise SystemExit(asyncio.run(main()))
