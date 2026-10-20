"""Controlled protocol fixtures only; no model/GPU/live service acceptance."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

PROBE_SOURCE = Path(os.environ.get('TOOL_CHAIN_PROBE',str(Path(__file__).resolve().parents[1]/'tests/tool_chain_live.py')))
DEPENDENCIES = Path(os.environ.get('TOOL_CHAIN_DEPENDENCIES',str(PROBE_SOURCE.parent)))
sys.path.insert(0,str(DEPENDENCIES))
spec=importlib.util.spec_from_file_location('tool_chain_live_candidate',PROBE_SOURCE)
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def call(task,count,identity,call_id,namespaced=True):
 return {'type':'function_call','id':identity,'call_id':call_id,'name':probe.NAME,
         'namespace':'functions' if namespaced else None,'status':'completed',
         'arguments':json.dumps({'task':task,'count':count})}


def calls(namespaced=True,parallel=True):
 out=[call('V7_FIRST',2,'fc_first','call_first',namespaced)]
 if parallel:out.append(call('V7_SECOND',3,'fc_second','call_second',namespaced))
 return out


def text_item(text):
 return {'type':'message','id':'msg_result','role':'assistant','status':'completed',
         'content':[{'type':'output_text','text':text,'annotations':[],'logprobs':None}]}


def final(output):
 return {'id':'resp_fixed','model':probe.MODEL,'status':'completed','error':None,
         'output':copy.deepcopy(output),'usage':{'input_tokens':8,'output_tokens':5,'total_tokens':13,
         'input_tokens_details':{'cached_tokens':2}}}


def answer(parallel=True):
 rows=[{'ack':'ACK_V7_FIRST','count':2}]
 if parallel:rows.append({'ack':'ACK_V7_SECOND','count':3})
 return json.dumps(rows)


def events_for(response):
 events=[{'type':'response.created','response':{'id':response['id'],'status':'in_progress'}},
         {'type':'response.in_progress','response':{'id':response['id'],'status':'in_progress'}}]
 for index,item in enumerate(response['output']):
  added=copy.deepcopy(item);added['status']='in_progress'
  if item['type']=='function_call':added['arguments']=''
  else:added['content']=[]
  events.append({'type':'response.output_item.added','output_index':index,'item':added})
  if item['type']=='function_call':
   events += [{'type':'response.function_call_arguments.delta','output_index':index,'item_id':item['id'],'delta':item['arguments']},
              {'type':'response.function_call_arguments.done','output_index':index,'item_id':item['id'],'arguments':item['arguments'],'name':item['name']}]
  else:
   text=item['content'][0]['text']
   event_prefix='response.reasoning_text' if item['type']=='reasoning' else 'response.output_text'
   events += [{'type':event_prefix+'.delta','output_index':index,'content_index':0,'item_id':item['id'],'delta':text},
              {'type':event_prefix+'.done','output_index':index,'content_index':0,'item_id':item['id'],'text':text}]
  events.append({'type':'response.output_item.done','output_index':index,'item':copy.deepcopy(item)})
 events.append({'type':'response.completed','response':copy.deepcopy(response)})
 return resequence(events)


def resequence(events):
 for i,event in enumerate(events):event['sequence_number']=i
 return events


def wire(events):
 return ''.join('event: '+e['type']+'\ndata: '+json.dumps(e,ensure_ascii=False)+'\n\n' for e in events).encode()


class Reply(io.BytesIO):
 def __init__(self,response=None,stream=False,body=None,status=200,content_type=None):
  super().__init__(body if body is not None else wire(events_for(response)) if stream else json.dumps(response).encode())
  self.status=status;self.content_type=content_type or ('text/event-stream' if stream else 'application/json')
 def getheader(self,name):return self.content_type if name.lower()=='content-type' else None


class HTTP:
 plans=[];sent=[];closed=[]
 def __init__(self,path,timeout):self.path=path;self.timeout=timeout
 def request(self,method,path,body,headers):
  self.__class__.sent.append({'method':method,'path':path,'body':json.loads(body),'headers':headers})
 def getresponse(self):return self.__class__.plans.pop(0)
 def close(self):self.__class__.closed.append(True)


class ProbeTests(unittest.TestCase):
 def setUp(self):
  HTTP.plans=[];HTTP.sent=[];HTTP.closed=[]
  self.patch=patch.object(probe,'UnixHTTP',HTTP);self.patch.start();self.addCleanup(self.patch.stop)
 def do_case(self,*,stream=False,namespaced=True,parallel=True,first=None,second=None):
  first=first if first is not None else final(calls(namespaced,parallel))
  second=second if second is not None else final([text_item(answer(parallel))])
  HTTP.plans=[Reply(first,stream),Reply(second,stream)]
  return probe.one_case(Path('/nonexistent-fixed-fixture.sock'),2,stream=stream,effort='medium',namespaced=namespaced,parallel=parallel)
 def audit(self,events):
  obj=probe.ToolAudit()
  for event in events:obj.feed(event)
  return obj.checked_contract()
 def test_good_json_and_sse_roundtrips_preserve_reverse_bindings(self):
  for stream in [False,True]:
   for namespaced,parallel in [(False,False),(True,True)]:
    with self.subTest(stream=stream,namespaced=namespaced):
     row=self.do_case(stream=stream,namespaced=namespaced,parallel=parallel)
     self.assertEqual(row['status'],'PASS',row)
     first,second=[x['body'] for x in HTTP.sent[-2:]]
     outputs=[x for x in second['input'] if x.get('type')=='function_call_output']
     self.assertEqual([x['call_id'] for x in outputs],['call_second','call_first'] if parallel else ['call_first'])
     self.assertEqual(second['tool_choice'],'none');self.assertEqual(first['tools'],second['tools'])
     for result in outputs:
      data=json.loads(result['output']);expected=3 if result['call_id']=='call_second' else 2
      self.assertEqual(data['count'],expected)
     self.assertEqual(row['actual_codex_subagent'],'NOT_RUN')
 def test_wrong_ack_count_pairing_is_not_accepted(self):
  wrong=json.loads(answer());wrong[0]['count'],wrong[1]['count']=3,2
  row=self.do_case(second=final([text_item(json.dumps(wrong))]))
  self.assertEqual(row['status'],'FAIL',row)
 def test_boolean_ack_count_is_not_integer(self):
  wrong=json.loads(answer());wrong[0]['count']=True
  row=self.do_case(second=final([text_item(json.dumps(wrong))]))
  self.assertEqual(row['status'],'FAIL',row)
 def test_model_does_not_need_to_echo_opaque_api_call_ids(self):
  row=self.do_case()
  self.assertEqual(row['status'],'PASS',row)
  self.assertFalse(row['model_call_id_echo_required'])
  self.assertTrue(row['api_call_id_contract_retained'])
  request=HTTP.sent[-1]['body']
  results=[item for item in request['input'] if item.get('type')=='function_call_output']
  self.assertTrue(all(item.get('call_id') for item in results))
  self.assertTrue(all(set(json.loads(item['output']))=={'ack','count'} for item in results))
  self.assertNotIn('call_id',request['input'][-1]['content'])

 def test_wrong_marker_is_rejected(self):
  row=self.do_case(second=final([text_item('ACK_V7_FIRST only')]))
  self.assertEqual(row['status'],'FAIL')
 def test_second_turn_cannot_add_tool_call(self):
  row=self.do_case(second=final(calls()))
  self.assertEqual(row['status'],'FAIL')
 def test_typed_arguments_reject_float_and_bool(self):
  for value in [2.0,True]:
   with self.subTest(value=value):
    items=calls();items[0]['arguments']=json.dumps({'task':'V7_FIRST','count':value})
    with self.assertRaises(ValueError):probe.validate_calls(items,namespaced=True,parallel=True)
 def test_duplicate_call_ids_are_rejected(self):
  items=calls();items[1]['call_id']=items[0]['call_id']
  with self.assertRaises(ValueError):probe.validate_calls(items,namespaced=True,parallel=True)
 def test_duplicate_item_ids_are_rejected(self):
  items=calls();items[1]['id']=items[0]['id']
  with self.assertRaises(ValueError):probe.validate_calls(items,namespaced=True,parallel=True)
 def test_nonstring_item_or_call_ids_are_rejected(self):
  for key in ['id','call_id']:
   with self.subTest(key=key):
    items=calls();items[0][key]=12
    with self.assertRaises(ValueError):probe.validate_calls(items,namespaced=True,parallel=True)
 def test_namespace_mismatch_is_rejected(self):
  with self.assertRaises(ValueError):probe.validate_calls(calls(False),namespaced=True,parallel=True)
 def test_wrong_model_incomplete_or_error_is_rejected(self):
  for change in [{'model':'wrong'},{'status':'incomplete'},{'error':{'code':'server_error','message':'fixed synthetic error'}}]:
   with self.subTest(change=change):
    response=final(calls());response.update(change)
    self.assertEqual(self.do_case(first=response)['status'],'FAIL')
 def test_invalid_usage_is_rejected(self):
  for usage in [{'input_tokens':8,'output_tokens':True,'total_tokens':9},
                {'input_tokens':8,'output_tokens':0,'total_tokens':8},
                {'input_tokens':-1,'output_tokens':5,'total_tokens':4},
                {'input_tokens':8,'output_tokens':5,'total_tokens':99},
                {'input_tokens':8,'output_tokens':5,'total_tokens':13,'input_tokens_details':{'cached_tokens':9}}]:
   with self.subTest(usage=usage):
    response=final(calls());response['usage']=usage
    self.assertEqual(self.do_case(first=response)['status'],'FAIL')
 def test_json_response_id_is_required_string(self):
  for value in [None,12,'']:
   with self.subTest(value=value):
    response=final(calls());response['id']=value
    self.assertEqual(self.do_case(first=response)['status'],'FAIL')
 def test_sse_lifecycle_response_id_is_consistent(self):
  for value in [None,12,'','resp_changed']:
   with self.subTest(value=value):
    events=events_for(final(calls()));events[-1]['response']['id']=value
    with self.assertRaises(ValueError):self.audit(events)
 def test_sse_missing_lifecycle_prefix_is_rejected(self):
  events=events_for(final(calls()))[2:]
  with self.assertRaises(ValueError):self.audit(resequence(events))
 def test_sse_orphan_argument_event_is_rejected(self):
  for kind in ['response.function_call_arguments.delta','response.function_call_arguments.done']:
   with self.subTest(kind=kind):
    events=events_for(final(calls()));event={'type':kind,'output_index':2,'item_id':'fc_orphan','name':'ghost'}
    event['delta' if kind.endswith('.delta') else 'arguments']='{}'
    events.insert(-1,event)
    with self.assertRaises(ValueError):self.audit(resequence(events))
 def test_sse_delta_mismatch_is_rejected(self):
  events=events_for(final(calls()));next(e for e in events if e['type'].endswith('arguments.delta'))['delta']='{}'
  with self.assertRaises(ValueError):self.audit(events)
 def test_sse_missing_arguments_done_is_rejected(self):
  events=[e for e in events_for(final(calls())) if e['type']!='response.function_call_arguments.done']
  with self.assertRaises(ValueError):self.audit(resequence(events))
 def test_sse_done_final_identity_mismatch_is_rejected(self):
  events=events_for(final(calls()));events[-1]['response']['output'][0]['id']='changed'
  with self.assertRaises(ValueError):self.audit(events)
 def test_sse_duplicate_added_or_done_is_rejected(self):
  for kind in ['response.output_item.added','response.output_item.done']:
   with self.subTest(kind=kind):
    events=events_for(final(calls()));i=next(i for i,e in enumerate(events) if e['type']==kind);events.insert(i,copy.deepcopy(events[i]))
    with self.assertRaises(ValueError):self.audit(resequence(events))
 def test_sse_unclosed_added_is_rejected(self):
  events=events_for(final(calls()));events.insert(-1,{'type':'response.output_item.added','output_index':2,'item':text_item('unclosed')})
  with self.assertRaises(ValueError):self.audit(resequence(events))
 def test_sse_event_after_terminal_is_rejected(self):
  events=events_for(final(calls()));events.append({'type':'response.output_text.delta','output_index':3,'item_id':'new','content_index':0,'delta':'extra'})
  with self.assertRaises(ValueError):self.audit(resequence(events))
 def test_sse_missing_terminal_or_nonconsecutive_sequence_is_rejected(self):
  for events in [events_for(final(calls()))[:-1],events_for(final(calls()))]:
   if events[-1]['type']=='response.completed':events[-1]['sequence_number']+=1
   with self.assertRaises(ValueError):self.audit(events)
 def test_sse_total_wire_limit_is_bounded(self):
  body=wire(events_for(final(calls())))+(b':'+b'x'*1000+b'\n')*(probe.LIMIT//1000+1)
  HTTP.plans=[Reply(body=body,stream=True)]
  with self.assertRaisesRegex(ValueError,'[Bb]ound|limit'):
   probe.request(Path('/nonexistent-fixed-fixture.sock'),{'stream':True},2)
 def test_http_error_body_is_not_exposed(self):
  secret='synthetic response body must not enter reports'
  HTTP.plans=[Reply(body=secret.encode(),status=500)]
  with self.assertRaises(ValueError) as error:probe.request(Path('/fixed.sock'),{'stream':False},2)
  self.assertEqual(str(error.exception),'HTTP 500');self.assertNotIn(secret,str(error.exception))
 def test_stream_report_contains_hashes_without_plaintext_or_hex(self):
  row=self.do_case(stream=True)
  self.assertEqual(row['status'],'PASS')
  for hop in ['first','second']:
   metadata=row[hop]['stream_contract'].get('stream_text',{})
   self.assertFalse({'text','repr','utf8_hex'}&set(metadata))


@unittest.skipUnless(os.environ.get('NATIVE_FIXED_R2')=='1','Only enabled inside pinned offline R2 image')
class NativeInputTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import socket
  assert [name for _,name in socket.if_nameindex()]==['lo'], 'Native fixture requires Docker --network none'
  import vllm.platforms
  from vllm.platforms.cpu import CpuPlatform
  vllm.platforms._current_platform=CpuPlatform()
 def run_native_case(self,namespaced,stream):
  from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
  from vllm.entrypoints.openai.responses import utils
  from vllm.tool_parsers.utils import iter_response_function_tool_dicts
  from vllm.tokenizers import deepseek_v41_encoding as encoder
  reasoning={'type':'reasoning','id':'rs_fixed','status':'completed','summary':[],
             'content':[{'type':'reasoning_text','text':'Fixed synthetic reasoning; no model output.'}]}
  first=final([reasoning]+calls(namespaced,True))
  second=final([text_item(answer(True))])
  HTTP.plans=[Reply(first,stream),Reply(second,stream)];HTTP.sent=[];HTTP.closed=[]
  with patch.object(probe,'UnixHTTP',HTTP):
   row=probe.one_case(Path('/nonexistent-native-fixture.sock'),2,stream=stream,effort='medium',namespaced=namespaced,parallel=True)
  self.assertEqual(row['status'],'PASS',row)
  body=HTTP.sent[-1]['body'];request=ResponsesRequest.model_validate(body)
  self.assertEqual(request.tool_choice,'none')
  messages=utils.construct_chat_messages_with_tool_call(request.input)
  generated_calls=[call for message in messages for call in message.get('tool_calls',[])]
  expected_name=('functions__' if namespaced else '')+probe.NAME
  self.assertEqual([call['function']['name'] for call in generated_calls],[expected_name,expected_name])
  self.assertEqual([call['id'] for call in generated_calls],['call_first','call_second'])
  results=[message for message in messages if message.get('role')=='tool']
  self.assertEqual([message['tool_call_id'] for message in results],['call_second','call_first'])
  self.assertEqual([json.loads(message['content'])['count'] for message in results],[3,2])
  tool_dicts=iter_response_function_tool_dicts(request.tools)
  self.assertEqual(tool_dicts[0]['name'],expected_name)
  self.assertTrue(any('Fixed synthetic reasoning' in str(message) for message in messages))
  ordered=encoder.sort_tool_messages_for_media(messages)
  self.assertEqual([message['tool_call_id'] for message in ordered if message.get('role')=='tool'],['call_first','call_second'])
 def test_ordinary_second_turn_original_output_input_schema_and_native_conversion(self):
  self.run_native_case(False,False)
 def test_namespace_sse_second_turn_original_output_input_schema_and_native_conversion(self):
  self.run_native_case(True,True)


if __name__=='__main__':unittest.main(verbosity=2)
