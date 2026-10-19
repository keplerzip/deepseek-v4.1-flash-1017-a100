"""1017 official-format parity and renderer media-order checks; no GPU needed."""
from __future__ import annotations

import ast
import asyncio
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "build/image/encoding"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


encoder = load("encoder_1017", PATCH / "patched/vllm/tokenizers/deepseek_v41_encoding.py")
reference = load("official_1017", PATCH / "source-reference.py")
installer = load("installer_1017", PATCH / "install.py")


def calls():
    return [{"id": "first", "type": "function", "function": {
                "name": "inspect", "arguments": '{"label":"FIRST"}'}},
            {"id": "second", "type": "function", "function": {
                "name": "inspect", "arguments": '{"label":"SECOND"}'}}]


def history(with_images=False):
    def content(label):
        parts = [{"type": "text", "text": label}]
        if with_images:
            parts.append({"type": "image_url", "image_url": {"url": label + ".png"}})
        return parts
    return [{"role": "user", "content": "Inspect the results."},
            {"role": "assistant", "content": "", "reasoning_content": "keep this thought",
             "tool_calls": calls()},
            {"role": "tool", "tool_call_id": "second", "content": content("SECOND")},
            {"role": "tool", "tool_call_id": "first", "content": content("FIRST")}]


def normalizer():
    """Execute the unchanged R1.9 wrapper function without transformers/vLLM."""
    source = ast.parse((PATCH / "source-wrapper.py").read_text())
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_normalize_messages")
    namespace = {"copy": copy, "Any": object, "ChatCompletionMessageParam": dict,
                 "IMAGE_PLACEHOLDER": encoder.IMAGE_PLACEHOLDER}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "source-wrapper.py", "exec"), namespace)
    return namespace["_normalize_messages"]


normalize = normalizer()


class OfficialParityTests(unittest.TestCase):
    def test_reference_is_pinned(self):
        manifest = json.loads((PATCH / "manifest.json").read_text())
        self.assertEqual(hashlib.sha256((PATCH / "source-reference.py").read_bytes()).hexdigest(),
                         manifest["reference"]["sha256"])
        self.assertEqual(hashlib.sha256((PATCH / "source-wrapper.py").read_bytes()).hexdigest(),
                         "ec2645dc38f2ba5a2997ab8e41df4295cf9f50a8a32106fc3f82597f51ec2fff")

    def test_text_and_tool_histories_match_official(self):
        examples = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "2+2"}],
            [{"role": "user", "content": [{"type": "text", "text": "alpha"},
                                           {"type": "text", "text": "beta"}]}],
            history(),
            history() + [{"role": "assistant", "content": "Done", "reasoning_content": "Result"}],
            history() + [{"role": "latest_reminder", "content": "Check carefully."}],
            [{"role": "user", "content": "First"}, {"role": "assistant", "content": "OK"},
             {"role": "system", "content": "A new instruction"}],
            [{"role": "tool", "content_blocks": [{"type": "text", "text": "Structured"}]}],
        ]
        for index, messages in enumerate(examples):
            before = copy.deepcopy(messages)
            for mode in ("chat", "thinking"):
                for effort in ("low", "high", "max", 61):
                    with self.subTest(example=index, mode=mode, effort=effort):
                        kwargs = {"thinking_mode": mode, "reasoning_effort": effort}
                        self.assertEqual(encoder.encode_messages(messages, **kwargs),
                                         reference.encode_messages(messages, **kwargs))
            self.assertEqual(messages, before)

    def test_tool_continuation_preserves_reasoning(self):
        prompt = encoder.encode_messages(history(), thinking_mode="thinking")
        self.assertIn("keep this thought</think>", prompt)
        self.assertLess(prompt.index("<tool_result>FIRST"), prompt.index("<tool_result>SECOND"))

    def test_context_is_merged_before_sorted_slice(self):
        full = history()
        context = full[:3]
        messages = full[3:]
        for mode in ("chat", "thinking"):
            self.assertEqual(encoder.encode_messages(messages, thinking_mode=mode, context=context),
                             reference.encode_messages(messages, thinking_mode=mode, context=context))

    def test_legal_wrapper_extensions_remain(self):
        messages = [{"role": "developer", "content": [{"type": "input_text", "text": "Guard"}]},
                    {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
                    {"role": "assistant", "content": [{"type": "output_text", "text": "Hello"}]},
                    {"role": "latest_reminder", "content": "Recall"}]
        normalized = normalize(messages)
        self.assertEqual(normalized[0], {"role": "system", "content": "Guard"})
        self.assertIn("<｜latest_reminder｜>Recall", encoder.encode_messages(normalized, thinking_mode="chat"))
        tools = [{"type": "function", "namespace": {"name": "repo", "description": "Local"},
                  "function": {"name": "read", "parameters": {"type": "object"}}}]
        encoded = encoder.tools_from_openai_format(tools)
        self.assertEqual(encoded[0]["name"], "repo::read")
        self.assertIn("Local", encoded[0]["description"])

    def test_stable_unknown_ids_and_non_tool_boundaries(self):
        messages = history()[:2] + [
            {"role": "tool", "tool_call_id": "unknown-A", "content": "A"},
            {"role": "tool", "tool_call_id": "second", "content": "SECOND"},
            {"role": "tool", "tool_call_id": "unknown-B", "content": "B"},
            {"role": "user", "content": "Boundary"},
            {"role": "tool", "tool_call_id": "first", "content": "FIRST"}]
        ordered = encoder.sort_tool_messages_for_media(messages)
        self.assertEqual([m.get("tool_call_id") for m in ordered[2:5]],
                         ["unknown-A", "unknown-B", "second"])
        self.assertEqual(ordered[5:], messages[5:])
        self.assertEqual([m["tool_call_id"] for m in messages if m["role"] == "tool"],
                         ["unknown-A", "second", "unknown-B", "first"])


class RendererOrderTests(unittest.TestCase):
    def renderer(self, model_type="deepseek_v41"):
        source = ast.parse((PATCH / "patched/vllm/renderers/deepseek_v4.py").read_text())
        node = next(n for n in source.body if isinstance(n, ast.ClassDef))
        node.bases = [ast.Name(id="object", ctx=ast.Load())]
        trace = {}

        def parse(messages, *args, **kwargs):
            trace["parsed"] = [m.get("tool_call_id") for m in messages if m["role"] == "tool"]
            images = [p["image_url"]["url"] for m in messages for p in m.get("content", [])
                      if isinstance(p, dict) and p.get("type") == "image_url"]
            return messages, {"image": images}, {"image": images}

        async def parse_async(*args, **kwargs):
            return parse(*args, **kwargs)

        namespace = {"VllmConfig": object, "DeepseekV4Tokenizer": object,
                     "ChatCompletionMessageParam": dict, "ConversationMessage": dict,
                     "DictPrompt": dict, "ChatParams": object,
                     "parse_chat_messages": parse, "parse_chat_messages_async": parse_async,
                     "parse_dec_only_prompt": lambda text: {"prompt": text}}
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(module, "patched-renderer.py", "exec"), namespace)
        renderer_type = namespace["DeepseekV4Renderer"]
        instance = renderer_type.__new__(renderer_type)
        instance.model_config = types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type=model_type))

        def apply_template(**kwargs):
            trace["rendered"] = [m.get("tool_call_id") for m in kwargs["messages"] if m["role"] == "tool"]
            return encoder.encode_messages(normalize(kwargs["messages"]), thinking_mode="thinking")

        instance.get_tokenizer = lambda: types.SimpleNamespace(apply_chat_template=apply_template)

        async def apply_async(**kwargs):
            return instance._apply_chat_template(**kwargs)

        instance._apply_chat_template_async = apply_async
        return instance, trace

    def exercise(self, asynchronous, model_type):
        renderer, trace = self.renderer(model_type)
        params = types.SimpleNamespace(media_io_kwargs=None, mm_processor_kwargs=None,
                                       get_apply_chat_template_kwargs=lambda: {})
        modules = {"vllm.tokenizers.deepseek_v41_encoding": encoder}
        messages = history(with_images=True)
        before = copy.deepcopy(messages)
        with patch.dict(sys.modules, modules):
            if asynchronous:
                _, prompt = asyncio.run(renderer.render_messages_async(messages, params))
            else:
                _, prompt = renderer.render_messages(messages, params)
        wanted = ["first", "second"] if model_type == "deepseek_v41" else ["second", "first"]
        self.assertEqual(trace["parsed"], wanted)
        self.assertEqual(trace["rendered"], wanted)
        self.assertEqual(prompt["multi_modal_data"]["image"], [x.upper() + ".png" for x in wanted])
        if model_type == "deepseek_v41":
            self.assertLess(prompt["prompt"].index("<tool_result>FIRST"),
                            prompt["prompt"].index("<tool_result>SECOND"))
            self.assertEqual(prompt["prompt"].count(encoder.IMAGE_PLACEHOLDER), 2)
            official_prompt, official_media = reference.encode_messages(
                messages, thinking_mode="thinking", return_multi_modal_data=True)
            self.assertEqual(prompt["prompt"], official_prompt)
            self.assertEqual(prompt["multi_modal_data"]["image"],
                             [item["url"] for item in official_media["images"]])
        self.assertEqual(messages, before)

    def test_sync_v41_collects_images_in_encoded_tool_order(self):
        self.exercise(False, "deepseek_v41")

    def test_async_v41_collects_images_in_encoded_tool_order(self):
        self.exercise(True, "deepseek_v41")

    def test_sync_other_model_preserves_input_order(self):
        self.exercise(False, "deepseek_v4")

    def test_async_other_model_preserves_input_order(self):
        self.exercise(True, "deepseek_v4")


class InstallerTests(unittest.TestCase):
    def test_installer_rejects_everything_before_any_write(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            package = folder / "package"
            shutil.copytree(PATCH, package)
            site = folder / "site"
            rows = json.loads((package / "manifest.json").read_text())["files"]
            for row in rows:
                target = site / row["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(package / "patched" / row["path"], target)
            installer.install(site, package)
            installer.install(site, package)
            first = site / rows[0]["path"]
            second = site / rows[1]["path"]
            expected = first.read_bytes()
            second.write_text("unexpected upstream modification\n")
            with self.assertRaisesRegex(RuntimeError, "Unexpected encoding base"):
                installer.install(site, package)
            self.assertEqual(first.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
