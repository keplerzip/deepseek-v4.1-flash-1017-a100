"""Client reasoning controls against the unchanged official prompt encoder."""
import ast
import copy
import importlib.util
from pathlib import Path
from typing import Any
import unittest


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / 'source/r39-overlay'
spec = importlib.util.spec_from_file_location(
    'reasoning_test_encoder',
    ROOT / 'build/image/encoding/patched/vllm/tokenizers/deepseek_v41_encoding.py')
encoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(encoder)


def functions(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names
             or isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id in names
                     for target in node.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


namespace = functions(OVERLAY / 'vllm/tokenizers/deepseek_v41.py', {
    'COMPAT_REASONING_EFFORT_MAPPINGS', 'normalize_reasoning_kwargs',
    '_normalize_messages', 'get_deepseek_v41_tokenizer'}, {
        'copy': copy, 'Any': Any, 'HfTokenizer': object,
        'ChatCompletionMessageParam': dict,
        'IMAGE_PLACEHOLDER': encoder.IMAGE_PLACEHOLDER,
        'REASONING_EFFORT_MAPPINGS': encoder.REASONING_EFFORT_MAPPINGS,
        'encode_messages': encoder.encode_messages})
budget_guard = functions(OVERLAY / 'dsv41_guard.py', {'validate_thinking_budget'}, {})[
    'validate_thinking_budget']


class StubTokenizer:
    def get_added_vocab(self):
        return {}


class ReasoningEffortTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = namespace['get_deepseek_v41_tokenizer'](StubTokenizer())
        self.messages = [{'role': 'user', 'content': 'Return two plus two.'}]

    def prompt(self, **kwargs):
        return self.tokenizer.apply_chat_template(self.messages, tokenize=False, **kwargs)

    def test_all_efforts_render_distinct_expected_strengths(self):
        for effort, numeric in [('minimal', 25), ('low', 50), ('medium', 63),
                                ('high', 75), ('xhigh', 90), ('max', 100),
                                (1, 1), (61, 61), (100, 100)]:
            with self.subTest(effort=effort):
                self.assertEqual(self.prompt(reasoning_effort=effort), encoder.encode_messages(
                    self.messages, thinking_mode='thinking', reasoning_effort=numeric))

    def test_none_overrides_enabled_thinking(self):
        self.assertEqual(self.prompt(thinking=True, enable_thinking=True, reasoning_effort='none'),
                         encoder.encode_messages(self.messages, thinking_mode='chat'))

    def test_explicit_off_wins_over_automatic_enable(self):
        for effort in ('low', 'medium', 'xhigh', 'max'):
            with self.subTest(effort=effort):
                self.assertNotIn('Reasoning Effort:', self.prompt(
                    thinking=False, enable_thinking=True, reasoning_effort=effort))

    def test_enable_thinking_off_is_honored(self):
        self.assertNotIn('Reasoning Effort:', self.prompt(
            enable_thinking=False, reasoning_effort='medium'))

    def test_default_and_native_names_keep_official_prompts(self):
        self.assertEqual(self.prompt(), encoder.encode_messages(self.messages, thinking_mode='thinking'))
        for effort in ('low', 'high', 'max', 1, 100):
            with self.subTest(effort=effort):
                self.assertEqual(self.prompt(reasoning_effort=effort), encoder.encode_messages(
                    self.messages, thinking_mode='thinking', reasoning_effort=effort))
        self.assertEqual(encoder.REASONING_EFFORT_MAPPINGS, {'low': 50, 'high': 75, 'max': 100})

    def test_invalid_effort_is_not_silently_defaulted(self):
        for effort in (True, False, 0, -1, 101, 50.0, '63', 'unknown', '', [], {}):
            with self.subTest(effort=effort):
                with self.assertRaisesRegex(ValueError, 'reasoning_effort'):
                    self.prompt(reasoning_effort=effort)

    def test_input_kwargs_and_messages_remain_unchanged(self):
        kwargs = {'reasoning_effort': 'medium', 'thinking': True}
        before = copy.deepcopy((kwargs, self.messages))
        self.prompt(**kwargs)
        self.assertEqual((kwargs, self.messages), before)


class ClaudeBudgetTests(unittest.TestCase):
    def test_enabled_budget_requires_real_positive_integer_below_output_limit(self):
        for budget in (True, False, 0, -1, 128.0, '128', None, 256, 512):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, 'budget_tokens'):
                    budget_guard({'thinking': {'type': 'enabled', 'budget_tokens': budget},
                                  'max_tokens': 256}, '/v1/messages')

    def test_budget_guard_does_not_mutate_valid_requests(self):
        for path in ('/v1/messages', '/v1/messages/count_tokens'):
            payload = {'thinking': {'type': 'enabled', 'budget_tokens': 128}, 'max_tokens': 256}
            before = copy.deepcopy(payload)
            budget_guard(payload, path)
            self.assertEqual(payload, before)

    def test_output_limit_cannot_be_coerced_past_budget_validation(self):
        for maximum in ('256', 256.0, True, False, None, 0, -1):
            with self.subTest(maximum=maximum):
                with self.assertRaisesRegex(ValueError, 'max_tokens'):
                    budget_guard({'thinking': {'type': 'enabled', 'budget_tokens': 128},
                                  'max_tokens': maximum}, '/v1/messages')

    def test_count_tokens_does_not_require_generation_output_limit(self):
        budget_guard({'thinking': {'type': 'enabled', 'budget_tokens': 128}},
                     '/v1/messages/count_tokens')

    def test_adaptive_without_budget_and_business_fields_are_untouched(self):
        budget_guard({'thinking': {'type': 'adaptive'}}, '/v1/messages')
        budget_guard({'thinking': {'budget_tokens': 'business data'}}, '/v1/chat/completions')
        budget_guard({'tools': [{'input_schema': {'thinking': {'budget_tokens': 'schema'}}}]}, '/v1/messages')


if __name__ == '__main__':
    unittest.main(verbosity=2)
