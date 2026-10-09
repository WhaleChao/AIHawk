<div align="center">
<picture>
  <source media="(max-width: 374px) and (prefers-color-scheme: dark)" srcset="docs/images/hero-small-dark.gif">
  <source media="(max-width: 374px)" srcset="docs/images/hero-small-light.gif">
  <source media="(max-width: 1239px) and (prefers-color-scheme: dark)" srcset="docs/images/hero-mobile-dark.gif">
  <source media="(max-width: 1239px)" srcset="docs/images/hero-mobile-light.gif">
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/hero-dark.gif">
  <img alt="invisible_dots, an open-source, self-hosted alternative to OpenAI Dots, Meta Muse, Grok Bot, Manus Cue and Claude Cowork. An animation of what a Dot does: asked once in its chat to save a shop's invoices every month and send the total to Telegram, it works on a computer of its own (a desktop, a terminal, files and a Firefox that keeps the shop's login), writes the steps down as a skill, asks before it adds the monthly automation, and then sends the total every month; several Dots, one computer each." src="docs/images/hero-light.gif" width="100%">
</picture>
</div>

Open-source, self-hosted alternative to OpenAI Dots, Grok Bot. Built to be undetectable by anti-bot systems.

## Quickstart

You need Node 24+, Go 1.25+, Git and hardware virtualization (x86-64), plus an OpenRouter key.

**Windows** (PowerShell):

```powershell
winget install -e --id OpenJS.NodeJS.LTS; winget install -e --id GoLang.Go; winget install -e --id Git.Git
git clone https://github.com/feder-cr/invisible_dots; cd invisible_dots
npm ci; npm run build --workspace @invisible-dots/cli
node apps/cli/dist/invisible-dots.mjs setup --all
node apps/cli/dist/invisible-dots.mjs server
```

**Linux** (Ubuntu 24.04):

```bash
curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash - && sudo apt-get install -y nodejs git && sudo snap install go --classic
git clone https://github.com/feder-cr/invisible_dots && cd invisible_dots
npm ci && npm run build --workspace @invisible-dots/cli
node apps/cli/dist/invisible-dots.mjs setup --all
node apps/cli/dist/invisible-dots.mjs server
```

`setup --all` installs QEMU and turns on the accelerator (KVM, or the Windows Hypervisor Platform), then builds or
downloads the Dot's images; run it again after a restart and it carries on. Then open **http://127.0.0.2:3000**,
paste your OpenRouter key and create your first Dot. Every step and its failure modes:
[the guide's quickstart](docs/guide.md#quickstart).

## What to ask a Dot

Anything that needs a computer and a person's judgement, for as long as it takes:

> Open a browser identity called research, go to https://example.com and tell me the page's title and its first
> sentence. Leave the browser open.

> Every morning, check one-way fares from Milan to Lisbon for the next two weeks, write them to
> ~/workspace/fares.csv and tell me the cheapest day.

> Log in to the shop with the shopping identity, download this month's invoices to ~/documents, and remember where
> the invoices page is for next time.

## What a Dot has

| | |
|---|---|
| **[A computer of its own](docs/guide.md#what-a-dot-can-do)** | A hardware-accelerated VM with a Linux desktop and a persistent disk. It sleeps when idle and wakes for the next message, task or automation. |
| **[A browser that is not blocked](docs/guide.md#the-browser)** | [invisible_playwright_mcp](https://github.com/feder-cr/invisible_playwright_mcp): Firefox patched in C++, the fingerprint set inside the engine. Each identity keeps its own cookies and logins. |
| **[Memory and skills](docs/guide.md#what-a-dot-can-do)** | It writes its own notes and how-tos in its home folder, as Claude Code does, and reads them on the next task. |
| **[Permissions you set](docs/guide.md#approvals)** | Every tool belongs to a permission: allow, ask or deny. An ask waits in your Inbox, survives a restart and runs the call once. |
| **[Tasks and automations](docs/guide.md#the-web-ui)** | A queue with priorities and start times, and its own recurring jobs, for which its computer is started on time. |
| **[Many ways to reach it](docs/guide.md#talk-to-it-from-your-phone)** | Web UI, command line, HTTP API with live events, Telegram, and WhatsApp as an opt-in. |
| **[Nothing lost on a crash](docs/guide.md#how-it-works)** | A restart or a `kill -9` keeps every message and task you saw accepted, and a tool call cut short is never run twice. |

## Documentation

- [Guide](docs/guide.md): install, the web UI, the command line, the browser, channels, configuration, updating
- [Architecture](docs/architecture.md): every component and why it is there
- [Troubleshooting](docs/guide.md#troubleshooting) and [development and tests](docs/guide.md#development-and-tests)

## Related projects

**The pieces of this one.** A Dot's browser is
[invisible_playwright_mcp](https://github.com/feder-cr/invisible_playwright_mcp), on
[invisible_playwright](https://github.com/feder-cr/invisible_playwright) and
[invisible_core](https://github.com/feder-cr/invisible_core); the engine is a fork of
[nanobot](https://github.com/HKUDS/nanobot).

## License

MIT, see [LICENSE](LICENSE). Third-party components and their licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Disclaimer

This project is provided as-is, with no warranties. Use it at your own risk and in compliance with the laws of your
jurisdiction. A Dot acts with your accounts and from your connection: respect the terms of the sites it visits and
their robots.txt.

---

<p align="center">
  <a href="https://github.com/feder-cr/invisible_dots/actions/workflows/tests.yml"><img alt="tests" src="https://github.com/feder-cr/invisible_dots/actions/workflows/tests.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="license: MIT" src="https://img.shields.io/badge/license-MIT-blue"></a>
  <img alt="status: alpha" src="https://img.shields.io/badge/status-alpha-orange">
  <img alt="hosts: Linux and Windows" src="https://img.shields.io/badge/hosts-Linux%20%7C%20Windows-lightgrey">
</p>

<p align="center">
  Built by <a href="https://it.linkedin.com/in/federico-elia-5199951b6">Federico Elia</a>
</p>
