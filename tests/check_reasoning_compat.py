"""Exercise installed Chat, Responses and Messages reasoning without a GPU.

Run through build/test_reasoning_compat.py in the pinned runtime image. The
actual native request models, conversion methods and 1017 prompt encoder are
used; no engine, vocabulary, model weights or inference are created.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import datetime
import hashlib
import importlib
import json
import re
import socket
from pathlib import Path
import traceback
from unittest.mock import patch


EFFORTS = {"none": None, "minimal": 25, "low": 50, "medium": 63,
           "high": 75, "xhigh": 90, "max": 100}
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")
MESSAGES = [{"role": "user", "content": "Give a brief answer."}]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class Checker:
    def __init__(self, output):
        self.output = output
        self.report = {
            "schema": 1, "suite": "reasoning-protocol-compatibility", "status": "RUNNING",
            "gpu_executed": False, "weights_loaded": False, "network_used": False,
            "real_inference_executed": False,
            "scope": "Installed native request validation, protocol conversion and official 1017 string encoder only; no generated reasoning quality, budget enforcement on GPU or TPS claim.",
            "effort_mapping": EFFORTS, "checks": [],
        }

    def save(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temp = self.output.with_name(self.output.name + ".tmp")
        temp.write_text(json.dumps(self.report, ensure_ascii=False, indent=2) + "\n")
        temp.replace(self.output)

    def check(self, name, action):
        row = {"name": name, "status": "FAIL"}
        try:
            row["details"] = action()
            row["status"] = "PASS"
        except Exception as error:
            row.update(error=type(error).__name__ + ": " + str(error),
                       traceback=traceback.format_exc()[-10000:])
        self.report["checks"].append(row)
        self.save()
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def setup(self):
        interfaces = socket.if_nameindex()
        require([name for _, name in interfaces] == ["lo"], "Docker --network none is required")
        route = Path("/proc/self/net/route").read_text().splitlines()[1:]
        require(not any(line.split()[1] == "00000000" for line in route if line.split()),
                "Default route must be absent")
        import vllm.platforms
        from vllm.platforms.cpu import CpuPlatform
        vllm.platforms._current_platform = CpuPlatform()
        self.torch = importlib.import_module("torch")
        self.chat = importlib.import_module("vllm.entrypoints.openai.chat_completion.protocol").ChatCompletionRequest
        self.responses = importlib.import_module("vllm.entrypoints.openai.responses.protocol").ResponsesRequest
        self.messages = importlib.import_module("vllm.entrypoints.anthropic.protocol").AnthropicMessagesRequest
        self.serving = importlib.import_module("vllm.entrypoints.anthropic.serving").AnthropicServingMessages
        self.error_response = importlib.import_module("vllm.entrypoints.serve.exception_handling.error_response").create_error_response
        self.sampling = importlib.import_module("vllm.sampling_params")
        self.validation_error = importlib.import_module("vllm.exceptions").VLLMValidationError
        self.guard = importlib.import_module("dsv41_guard").OfflineGuard
        tokenizer = importlib.import_module("vllm.tokenizers.deepseek_v41")

        class StubTokenizer:
            def get_added_vocab(self):
                return {}

        self.wrapped = tokenizer.get_deepseek_v41_tokenizer(StubTokenizer())
        require(not self.torch.cuda.is_initialized(), "CPU checks must not initialize CUDA")
        return {"interfaces": [name for _, name in interfaces], "cuda_context_initialized": False}

    def convert(self, protocol, options):
        body = {"model": "DeepSeek-Flash", "messages": copy.deepcopy(MESSAGES)}
        if protocol == "responses":
            body = {"model": "DeepSeek-Flash", "input": "Give a brief answer."}
            model = self.responses
        elif protocol == "messages":
            body["max_tokens"] = 8192
            model = self.messages
        else:
            model = self.chat
        body.update(copy.deepcopy(options))
        original = copy.deepcopy(body)
        # Native Responses preprocessing normalizes its input mapping in place.
        # Isolate that framework construction step, then verify this patch's
        # conversion and rendering do not mutate the request or its kwargs.
        request = model.model_validate(copy.deepcopy(body))
        request_original = copy.deepcopy(request.model_dump())
        converted = self.serving.to_chat_completion_request(request) if protocol == "messages" else request
        params = converted.build_chat_params(None, "auto")
        kwargs = copy.deepcopy(params.chat_template_kwargs)
        prompt = self.wrapped.apply_chat_template(copy.deepcopy(MESSAGES), tokenize=False, **kwargs)
        require(body == original, "Request input was mutated")
        require(request.model_dump() == request_original, "Native request model was mutated")
        require(params.chat_template_kwargs == kwargs, "Chat template kwargs were mutated")
        return prompt, converted, kwargs

    def prompt_matches(self, prompt, value):
        prefix = re.search(r"Reasoning Effort: ([0-9]+) ", prompt)
        if value is None:
            require(prefix is None, "Disabled thinking emitted an effort prefix")
            require(prompt.endswith("</think>"), "Disabled thinking did not use the chat suffix")
        else:
            require(prefix is not None and int(prefix.group(1)) == value,
                    "Wrong reasoning effort prefix: expected " + str(value))
            require(prompt.endswith("<think>"), "Thinking did not use the thinking suffix")

    def protocol_case(self, protocol, effort, stream=False, template=False, mode=None):
        options = {"stream": stream}
        if template:
            options["chat_template_kwargs"] = {"reasoning_effort": effort}
        elif protocol == "responses":
            options["reasoning"] = {"effort": effort, "summary": "auto"}
        elif protocol == "messages":
            options.update(thinking={"type": mode}, output_config={"effort": effort})
        else:
            options["reasoning_effort"] = effort
        prompt, _, _ = self.convert(protocol, options)
        value = EFFORTS[effort] if isinstance(effort, str) else effort
        self.prompt_matches(prompt, value)
        return {"protocol": protocol, "stream": stream, "effort": effort,
                "encoded_effort": value, "input_unchanged": True}

    def explicit_flags(self):
        cases = [
            ("chat", {"reasoning_effort": "medium", "chat_template_kwargs": {"thinking": False}}, None),
            ("chat", {"reasoning_effort": "xhigh", "chat_template_kwargs": {"enable_thinking": False}}, None),
            ("responses", {"reasoning": {"effort": "medium"}, "chat_template_kwargs": {"thinking": False}}, None),
            ("responses", {"reasoning": {"effort": "xhigh"}, "chat_template_kwargs": {"enable_thinking": False}}, None),
            ("chat", {"reasoning_effort": "none", "chat_template_kwargs": {"thinking": True}}, None),
            ("responses", {"reasoning": {"effort": "none"}, "chat_template_kwargs": {"thinking": True}}, None),
            ("chat", {"chat_template_kwargs": {"thinking": True, "enable_thinking": False, "reasoning_effort": "medium"}}, 63),
            ("chat", {"chat_template_kwargs": {"thinking": False, "enable_thinking": True, "reasoning_effort": "medium"}}, None),
            ("chat", {"reasoning_effort": "medium", "chat_template_kwargs": {"reasoning_effort": "low"}}, 63),
            ("responses", {"reasoning": {"effort": "xhigh"}, "chat_template_kwargs": {"reasoning_effort": "low"}}, 90),
        ]
        for protocol, options, value in cases:
            prompt, _, _ = self.convert(protocol, options)
            self.prompt_matches(prompt, value)
        return {"cases": len(cases), "template_flags_take_precedence": True,
                "none_disables_thinking": True, "native_top_level_effort_precedence_preserved": True}

    def defaults_and_claude(self):
        cases = [
            ("chat", {}, 75), ("responses", {}, 75), ("messages", {}, None),
            ("messages", {"output_config": {"effort": "medium"}}, None),
            ("messages", {"thinking": {"type": "disabled"}, "output_config": {"effort": "xhigh"}}, None),
            ("messages", {"thinking": {"type": "adaptive"}}, 75),
            ("messages", {"thinking": {"type": "enabled"}}, 75),
            ("messages", {"thinking": {"type": "disabled"}, "chat_template_kwargs": {"thinking": True, "reasoning_effort": "medium"}}, 63),
        ]
        for protocol, options, value in cases:
            prompt, _, _ = self.convert(protocol, options)
            self.prompt_matches(prompt, value)
        return {"cases": len(cases), "chat_and_responses_default": 75,
                "messages_default": "off", "explicit_template_override_preserved": True}

    def claude_budget(self):
        for value in (1024, 4096):
            prompt, converted, _ = self.convert("messages", {
                "thinking": {"type": "enabled", "budget_tokens": value},
                "output_config": {"effort": "medium"},
            })
            self.prompt_matches(prompt, 63)
            require(converted.thinking_token_budget == value, "Claude thinking token budget was lost")
            require(converted.to_sampling_params(8192, {}).thinking_token_budget == value,
                    "Thinking token budget did not reach native sampling params")
        prompt, converted, _ = self.convert("messages", {
            "thinking": {"type": "disabled", "budget_tokens": 1024},
            "output_config": {"effort": "medium"},
        })
        self.prompt_matches(prompt, None)
        require(converted.thinking_token_budget is None, "Disabled thinking retained an active budget")
        _, converted, _ = self.convert("messages", {"thinking": {"type": "adaptive"}})
        require(converted.thinking_token_budget is None, "Adaptive thinking acquired an invented token cap")
        disabled_cases = [
            {"chat_template_kwargs": {"thinking": False}, "output_config": {"effort": "medium"}},
            {"chat_template_kwargs": {"enable_thinking": False}, "output_config": {"effort": "xhigh"}},
            {"chat_template_kwargs": {"reasoning_effort": "none"}},
            {"chat_template_kwargs": {"thinking": False, "enable_thinking": True}},
        ]
        for options in disabled_cases:
            prompt, converted, _ = self.convert("messages", dict(
                options, thinking={"type": "enabled", "budget_tokens": 1024}))
            self.prompt_matches(prompt, None)
            require(converted.thinking_token_budget is None,
                    "Explicitly disabled thinking retained an active token budget")
            require(converted.to_sampling_params(8192, {}).thinking_token_budget is None,
                    "Explicitly disabled thinking retained a native sampling token cap")
        prompt, converted, _ = self.convert("messages", {
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "chat_template_kwargs": {"reasoning_effort": "none"},
            "output_config": {"effort": "medium"},
        })
        self.prompt_matches(prompt, 63)
        require(converted.thinking_token_budget == 1024,
                "Top-level effort precedence did not retain the enabled budget")
        prompt, converted, _ = self.convert("messages", {
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "chat_template_kwargs": {"thinking": True, "enable_thinking": False},
            "output_config": {"effort": "medium"},
        })
        self.prompt_matches(prompt, 63)
        require(converted.thinking_token_budget == 1024,
                "Explicit thinking=True precedence lost the enabled budget")
        return {"enabled_budgets_forwarded": [1024, 4096], "native_sampling_params_verified": True,
                "disabled_budget": None, "adaptive_without_budget": None,
                "explicit_disabled_budget_cases": len(disabled_cases),
                "enabled_budget_precedence_cases": 2,
                "gpu_budget_enforcement": "NOT_RUN"}

    def invalid_values(self):
        rejected = 0
        for value in (0, 101, -1, True, False, 1.5, "unknown", "63", {}, []):
            try:
                self.wrapped.apply_chat_template(copy.deepcopy(MESSAGES), tokenize=False, reasoning_effort=value)
            except (ValueError, TypeError) as error:
                require(self.error_response(error).error.code == 400, "Invalid effort did not map to HTTP 400")
                rejected += 1
            else:
                raise AssertionError("Invalid template effort was accepted: " + repr(value))
        for protocol in ("chat", "responses", "messages"):
            options = {"reasoning_effort": "unknown"}
            if protocol == "responses":
                options = {"reasoning": {"effort": "unknown"}}
            elif protocol == "messages":
                options = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "unknown"}}
            try:
                self.convert(protocol, options)
            except ValueError as error:
                require(self.error_response(error).error.code == 400, "Invalid protocol enum did not map to HTTP 400")
                rejected += 1
            else:
                raise AssertionError("Unknown native effort enum was accepted")
        return {"invalid_cases_rejected": rejected, "mapped_http_status": 400}

    def sampling_budget_validation(self):
        for value in (None, -1, 0, 1024):
            result = self.sampling.validate_thinking_token_budget(value)
            require(result == (None if value == -1 else value), "Native sampling budget normalization changed")
        for value in (True, False, 1.0, -2):
            try:
                self.sampling.validate_thinking_token_budget(value)
            except self.validation_error as error:
                require(self.error_response(error).error.code == 400, "Native budget rejection was not HTTP 400")
            else:
                raise AssertionError("Native sampling accepted an invalid budget")
        require(not self.torch.cuda.is_initialized(), "CUDA context was initialized")
        return {"native_valid_cases": 4, "native_invalid_cases": 4,
                "cuda_context_initialized": False}

    def guard_budgets(self):
        async def invoke(payload, path):
            raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            captured, events = [], []

            async def app(scope, receive, send):
                request = await receive()
                captured.append(request["body"])
                await send({"type": "http.response.start", "status": 200, "headers": []})
                await send({"type": "http.response.body", "body": b"{}"})

            async def receive():
                return {"type": "http.request", "body": raw, "more_body": False}

            async def send(event):
                events.append(event)

            with patch.dict("os.environ", {"VLLM_API_KEY": "", "DSV41_DP_SIZE": "1"}):
                await self.guard(app)({"type": "http", "method": "POST", "path": path, "headers": []}, receive, send)
            status = next(event["status"] for event in events if event["type"] == "http.response.start")
            return status, raw, captured

        rejected = 0
        for path in ("/v1/messages", "/v1/messages/count_tokens"):
            for value in (True, False, "1024", 1024.0, [], {}, 0, -1):
                payload = {"model": "DeepSeek-Flash", "messages": copy.deepcopy(MESSAGES),
                           "thinking": {"type": "enabled", "budget_tokens": value}}
                if path == "/v1/messages":
                    payload["max_tokens"] = 8192
                status, _, captured = asyncio.run(invoke(payload, path))
                require(status == 400 and not captured, "Invalid raw Claude budget reached native coercion")
                rejected += 1
        for value in (8192, 8193):
            payload = {"model": "DeepSeek-Flash", "messages": copy.deepcopy(MESSAGES), "max_tokens": 8192,
                       "thinking": {"type": "enabled", "budget_tokens": value}}
            status, _, captured = asyncio.run(invoke(payload, "/v1/messages"))
            require(status == 400 and not captured, "Budget >= max_tokens was accepted")
            rejected += 1
        invalid_maximum = ("8192", 8192.0, True, False, None, [], {})
        for value in invalid_maximum:
            payload = {"model": "DeepSeek-Flash", "messages": copy.deepcopy(MESSAGES), "max_tokens": value,
                       "thinking": {"type": "enabled", "budget_tokens": 1024}}
            status, _, captured = asyncio.run(invoke(payload, "/v1/messages"))
            require(status == 400 and not captured,
                    "Invalid raw max_tokens reached native coercion with an enabled budget")
            rejected += 1
        forwarded = 0
        for path in ("/v1/messages", "/v1/messages/count_tokens"):
            payload = {"model": "DeepSeek-Flash", "messages": copy.deepcopy(MESSAGES),
                       "thinking": {"type": "enabled", "budget_tokens": 1024}}
            if path == "/v1/messages":
                payload["max_tokens"] = 8192
            status, raw, captured = asyncio.run(invoke(payload, path))
            require(status == 200 and captured == [raw], "Valid request body bytes were modified")
            forwarded += 1
        return {"raw_invalid_budgets_rejected_before_pydantic": rejected,
                "invalid_raw_max_tokens_cases": len(invalid_maximum),
                "valid_request_bodies_forwarded_byte_identically": forwarded, "http_status": 400}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    checker = Checker(args.output)
    checker.check("pinned-image-imports-and-offline-namespace", checker.setup)
    if checker.report["checks"][-1]["status"] == "PASS":
        for protocol in ("chat", "responses"):
            for effort in EFFORTS:
                for stream in (False, True):
                    name = protocol + "-" + effort + ("-sse" if stream else "-json")
                    checker.check(name, lambda p=protocol, e=effort, s=stream: checker.protocol_case(p, e, s))
        for effort in EFFORTS:
            checker.check("chat-template-" + effort,
                          lambda e=effort: checker.protocol_case("chat", e, template=True))
        for effort in (1, 37, 100):
            checker.check("chat-template-numeric-" + str(effort),
                          lambda e=effort: checker.protocol_case("chat", e, template=True))
        for mode in ("enabled", "adaptive"):
            for effort in CLAUDE_EFFORTS:
                for stream in (False, True):
                    name = "messages-" + mode + "-" + effort + ("-sse" if stream else "-json")
                    checker.check(name, lambda m=mode, e=effort, s=stream: checker.protocol_case("messages", e, s, mode=m))
        checker.check("explicit-flags-and-effort-precedence", checker.explicit_flags)
        checker.check("protocol-defaults-and-disabled-thinking", checker.defaults_and_claude)
        checker.check("claude-budget-to-native-sampling-params", checker.claude_budget)
        checker.check("invalid-efforts-have-client-error-status", checker.invalid_values)
        checker.check("native-thinking-budget-validation", checker.sampling_budget_validation)
        checker.check("guard-budget-validation-before-native-coercion", checker.guard_budgets)
    checker.report["status"] = "PASS" if all(row["status"] == "PASS" for row in checker.report["checks"]) else "FAIL"
    checker.report["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    checker.report["checker_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    checker.save()
    print(json.dumps({"status": checker.report["status"], "checks": len(checker.report["checks"]),
                      "gpu_executed": False, "weights_loaded": False}), flush=True)
    return 0 if checker.report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
