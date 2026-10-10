# Upstream

This directory is a hard fork of nanobot (https://github.com/HKUDS/nanobot, MIT),
copied at commit f75470e72f0993dcf92accc81282adaa48b16f56. It was imported by
commit 8dd5ddb1 of this repository.

License: `LICENSE` is upstream's, byte for byte. Its git blob id is
e06eb24cf4abdc5b46d354fc7d1c9531b2f52dfa. The notices are in the root
`THIRD_PARTY_NOTICES.md` (section nanobot) and in `THIRD_PARTY_NOTICES.md` here.

hard fork: upstream changes are not tracked or merged. The code here is modified
directly for invisible_dots.

## What was left out of the import

Only the engine core was copied. Left out: nanobot/channels (the host owns all
communication), nanobot/webui and webui/, tui/, nanobot/audio, nanobot/pairing,
nanobot/web, docs/, tests/, packages/ and the deployment files.

## What the first cut removed

The engine is a library core (runner, tools, providers, cron, sessions); the
app layers are gone:

- the shells: nanobot/cli, apps, gateway, api, sdk, command, bus, triggers,
  nanobot.py (the facade), process_runtime.py, optional_features.py,
  hatch_build.py, and the console scripts of pyproject.toml;
- the turn machinery of the agent loop: agent/loop.py, autocompact.py,
  automation_turns.py, cron_turns.py, goal_permission.py, model_presets.py,
  model_runtime.py, progress_hook.py, turn_delivery.py, turn_hooks.py, hooks/,
  subagent.py, skills.py, plugins.py, and with them every bundled skill;
- the tools that only the loop, subagents, goals and channels used: spawn,
  message, self, long_task, sessions, session_messages, runtime_control, and the
  web, image generation and cli_apps tools;
- the session modules of goals, turn continuation, recovery, handles and the web
  UI, and the cron modules that bound a job to a session or a delivery route
  (a job is now a schedule and a message);
- the Dream memory machinery and the git store behind it, the SOUL.md, USER.md,
  AGENTS.md and HEARTBEAT.md templates, and the prompts of the deleted features;
- helpers only the deleted code used (tool hints, restart, evaluator, run
  records, search usage, log configuration, media decoding, rotating output,
  workspace prompt overrides, OAuth guidance and transcription).

## What the second cut removed

The engine offers exec, exec_session, list_exec_sessions, read_file, write_file,
edit_file, apply_patch, list_dir, find_files, grep, cron and the MCP client's
tools, and talks to OpenRouter only:

- the tool plugin machinery (agent/tools/loader.py, the entry points, `enabled`,
  `create`, `config_cls`, `config_key`, scopes and the per-tool config classes),
  the rg tool and the MCP OAuth client (MCP keeps its stdio and HTTP transports);
- every provider but the OpenAI-compatible one: Anthropic, Azure OpenAI, Bedrock,
  GitHub Copilot, OpenAI Codex, xAI, the fallback chain, the provider factory,
  the OAuth model catalog, and the Responses API (providers/openai_responses/
  and the provider's Responses branch); the registry keeps one spec,
  `openrouter`, and no credential is read from the environment;
- usage telemetry (llm_usage/, the call observer of the provider base) and
  utils/artifacts.py;
- the configuration files and their loader (config/, config_base.py), the
  `${ENV}` interpolation and every home-directory default; the JSONL session
  store and SessionManager (session/manager.py keeps `Session`);
- the file memory (MemoryStore, history.jsonl, Dream's cron job) and the
  workspace templates; agent/memory.py keeps the transcript summary;
- document and image reading (utils/document.py: a binary file is reported as
  binary), workspace template syncing, and the dependencies nothing imports
  (tests/test_dependencies.py keeps the list honest).

## What the third cut changed: the model's computer

The model's commands and files are the Dot's own, as the user dot, through
dot-agentd; the engine user never runs a model command or touches a model file.

- `nanobot/dots/computer.py` is the one door: `AgentdComputer` speaks HTTP to
  agentd.sock for files and runs programs through `dot-agentd relay` (the relay
  is a local child that lives as long as the remote command; killing it ends the
  remote process group). `exec`, the exec sessions, the file tools, `find_files`,
  `grep` and `apply_patch` take a `Computer`.
- deleted: bubblewrap (agent/tools/sandbox.py), the deny and allow patterns and
  the command guard, the workspace path policy and its extra directories
  (security/), the SSRF guard, the PowerShell and Windows job object code, the
  `shell`/`login`/`allowed_env_keys`/`path_*` parameters, the SSRF and
  workspace-violation classification of tool errors in the runner. What dot may
  touch is decided by the operating system; what the model may call is decided
  by the permission table (nanobot/dots/permissions.py).
- `find_files` and `grep` run `find`, `stat` and `grep -P` on the Dot's computer
  as a prefilter (argv, no shell) and read only the files that can match, so no
  tree is walked over the socket; the tools' own filtering, sorting, paging and
  limits are unchanged.
- added: `exec` takes `tty` (upstream has no pseudo-terminal): the relay is started
  with `--tty` and `TERM`, the call is always an exec session, and a tty session's
  output goes through `terminal_text` (exec_session.py), the text of the screen
  without escape sequences, instead of the raw stream.
- tests use `tests/fakes/local_computer.py` (the same protocol over a tmp
  directory) and `tests/fakes/fake_relay.py` (parses the relay flags as relay.go
  does and execs the program); production has only `AgentdComputer`.

## What the fourth cut changed: the gate, the commits and the turn

Every model turn is one `TurnRunner.run` (nanobot/dots/turns.py) over nanobot's
`AgentRunner`, and every tool call crosses one policy gate.

- agent/tools/execution.py: `_admit_tool_call` asks the gate (a required `ToolGate`,
  nanobot/agent/tools/gate_types.py, implemented by nanobot/dots/gate.py) after the
  call is validated and before anything runs, for every call of a concurrent batch
  before any of them starts; a parked call ends the round, and the calls after it
  are skipped. The web tools' repeated-lookup throttle is deleted with them. `ToolRegistry.execute` and the branch that called it are deleted, so no
  other path runs a tool; `ToolRegistry.view(names)` gives each turn its tools.
- agent/runner.py: every message the runner adds goes, one at a time, through a
  commit callback before the runner goes on, and a parked call ends the turn.
  The aggregate checkpoint payloads (awaiting tools, tools completed, provider
  state) and the runtime checkpoint file are gone.
- agent/hook.py: the generic hook machinery (composite hooks, the SDK capture
  hook, the turn hook factory types); the Dot has one hook.
- agent/context.py: the system prompt is the Dot's section, the tool contract,
  a note on the Dot's computer and its memory notes, and the time. The bootstrap
  files, the skills section and the identity, platform-policy and
  untrusted-content templates are replaced by templates/agent/platform.md.
- session/manager.py keeps `Session` and `get_history` only; the transcripts are
  the Dot's SQLite tables (nanobot/dots/store.py). The request context keeps the
  session key only.
- providers: the attribution headers of OpenRouter are the Dot's, and are sent
  only to an openrouter.ai base URL.

## What the fifth cut removed: what the Dot never reaches

Code that nothing in the engine's production path calls, and options that no
caller sets, are deleted rather than kept disabled.

- agent/runner.py: the options `initial_messages` (a run starts from a
  `TranscriptInput` and its builder, both required), `terminal_injection_callback`,
  `continuation_callback`, `finalize_on_max_iterations` (a run that reaches its step
  limit ends on the limit message, without a last request), `max_iterations_message`,
  `error_message` and `provider_retry_mode`; the streaming of the answer and of the
  reasoning (`wants_streaming` and the hook methods `on_stream`, `on_stream_end`,
  `on_provider_tool_event`, `emit_reasoning` and `emit_reasoning_end`), with the timing
  and hosted-tool bookkeeping that served it.
- the runtime-context markers (runtime_context.py, their merge in
  context_governance.py, `Tool.runtime_context_provider`): nothing appends them.
- package re-exports: the `__init__.py` of agent, agent/tools, session, providers,
  cron and utils name their package and export nothing.
- `LLMProvider.chat_with_retry` and `chat_with_context` (the retry policy is
  exercised through `chat_stream_with_retry`, the one entry the runner uses),
  `LLMUsage.to_dict`, `from_dict` and `to_turn_dict`, `ProviderConversationState`'s
  private record, `ExecSessionManager.terminate_by_owner`, `MCPProvider`'s status
  reports, `NumberSchema`, `RecoveryStateEvent`, `Session.add_message`,
  `ContextBuilder.build_messages`, `LLMRuntime.with_generation_overrides`,
  utils/path.py, and helpers no caller uses (`atomic_write_lines`, `split_message`,
  `build_status_content`, the image-block builders, the incremental think extractor).
- in the Dot's layer: `KeyHolder.get` and `on_change`, `DotStore.read_last_seq` and
  `is_task_session_key`, `EngineSettings.memory_enabled`.

One criterion decides what of upstream's provider API stays: what the Dot's code path
calls (`chat_stream_with_retry` from the runner and the transcript summary, then
`_safe_chat_stream`, `chat_stream_with_context` and the provider's `chat_stream`). The
rest is deleted, with the tests that pinned it:

- the streaming callbacks `on_content_delta`, `on_thinking_delta`, `on_tool_call_delta`
  and `on_stream_recover` of `chat_stream` and `chat_stream_with_retry` (nothing in
  nanobot/agent or nanobot/dots passes them), the retry guard that decided whether a
  failed stream could be retried after content was emitted, the source attribution that
  fired on the first delta, and `supports_stream_recover_callback`;
- `LLMProvider.chat` and the default `chat_stream` that fell back to it (`chat_stream`
  is now the abstract method), `OpenAICompatProvider.chat`, the provider's
  `chat_stream_with_context` override and the `provider_context` it forwarded and never
  read, and `OpenAICompatProvider._parse`, the response parser of the non-streaming
  call (`_parse_chunks` is the one parser);
- the `persistent` retry mode (`retry_mode`, `_PERSISTENT_MAX_DELAY`,
  `_PERSISTENT_IDENTICAL_ERROR_LIMIT`, the identical-error counter);
- the provider-owned conversation state and the provider-native compaction: no provider
  of the Dot resumes state kept on the provider's side or compacts there
  (`OpenAICompatProvider` returns no state and reports no compaction), so the runner
  only ever took the paths of a provider without them. Deleted: providers/conversation_state.py
  (`ProviderConversationStateController`), `ProviderConversationState`,
  `LLMProvider.can_resume_conversation_state` and `supports_pre_request_compaction`,
  `LLMResponse.provider_state`, `provider_compaction_applied`, `provider_compaction_state`,
  `provider_compaction_scope` and `preserve_provider_state_on_error`, the fields
  `conversation_state`, `context_window_tokens`, `session_id` and `compaction_input_budget`
  of `ProviderCallContext` (it keeps the event sink and the response preset),
  `AgentRunSpec.provider_state` and `consolidate_provider_compaction`,
  `AgentRunResult.provider_state` and `provider_compaction_applied`, the resumed-state
  measurement of `ContextGovernor.measure_request`, `ContextGovernor.summarize_provider_compaction`,
  `Consolidator.summarize_provider_compaction` and its `provider_state` argument, and the
  read results a native compaction made stale (`compacted_tool_results`).

The one text of a provider failure is `LLMProvider.failure_text`: the provider's error
body or the exception's message. The provider's own handling and the base class's `_safe_chat_stream` both use it.

Kept on purpose, with the reason, for the next cut to decide:

- `CronService.update_job`, `enable_job`, `run_job`, `register_system_job` and
  `remove_system_job`: the Dot's cron tool adds, lists and removes, but the service's
  tests drive its timer, its store and its recovery through them.
- `_BoundedOutputBuffer.retained_chars`: the invariant that an unpolled session holds
  at most its bound is asserted through it.
- the retry notifications of `LLMProvider` (the Dot's runs carry no event sink, so the
  provider context has no publisher and no notification is ever sent), the spill of a
  long tool result to a file (the Dot passes no workspace) and
  `AgentHook`'s lifecycle methods the Dot's hook does not override: each is exercised
  by upstream tests that pin its behavior, and none is reachable from the Dot.

## What was added: the Dot's layer

nanobot/dots/ is new: the contract with the host (protocol, server, engine,
store, transcript outbox, gate, permission table, projection), the key holder
and the credentials check, the OpenRouter provider holder, the computer
(nanobot/dots/computer.py, above), the guest checks and the
entry point (`python -I -B -m nanobot`, which answers `--version` and refuses
every other command). It is described in docs/architecture.md, section 8.8.

## What the browser phase changed: images and the Dot's one MCP server

- agent/tools/mcp.py: `MCPServerConfig.images` and the wrapper's `images` flag return a
  tool result that has an image as a list of content blocks (text and `image_url` data
  URLs), where every other caller still gets text and the bytes of an image dropped;
  `MCPProvider.connect()` returns the servers that did not connect, so a caller can fail
  a launch instead of reading the log; the per-server `tool_timeout` is pinned by a test.
  `MCPProvider(on_terminated=...)` replaces the silent reconnect: a server whose session
  ended is reported by name and not reconnected (a restarted browser server has lost its
  browser), both when a call finds out and, through `connect_mcp_servers(on_ended=...)` and
  the read-stream filter, when the transport ends with no call in flight. Without
  `on_terminated` the provider reconnects as before.
  The one server is `invisible-playwright-mcp`, started by the `BrowserManager`
  (nanobot/dots/browser.py) through `dot-agentd relay`; `main.py` no longer builds an
  `MCPProvider` of its own.
- agent/runner.py: `AgentRunSpec.request_attachments` takes the messages of one model
  request and returns the ones to send. The Dot uses it to show the newest screenshots of
  a turn (nanobot/dots/images.py) without them entering the transcript.

## What the automations change: runs missed while the computer was off

- cron/service.py: `_recompute_next_runs` no longer counts a next run from now for every enabled job at
  start: only for a job that has none. A next run already stored, in the past or not, is kept, so the
  first tick runs a job that came due while no process ran once (a one-time job, which has no next
  run once its time passed, ran never before) and counts its next run from that moment, one run for
  all the occurrences a recurring job missed. `CronService(on_next_wake=...)` is told the earliest
  next run of the enabled jobs (None when there is none) each time the timer is armed, which is
  after every change and every tick; a failure of the callback is logged and the jobs go on. The Dot
  uses it to tell the host (`Engine.automations_next_run`, the `automation.next_run` event; docs/architecture.md
  sections 5.4, 8.8 and 9.5).
