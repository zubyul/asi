<p align="center">
  <img src="https://img.shields.io/badge/skills-20-blue?style=for-the-badge" alt="Skills" />
  <img src="https://img.shields.io/badge/agents-9+-green?style=for-the-badge" alt="Compatible Agents" />
  <img src="https://img.shields.io/badge/license-MIT-brightgreen?style=for-the-badge" alt="License" />
</p>

<h1 align="center">Ai Agent Skills</h1>

<p align="center">
  <strong>The first universal skill repository for AI agents.</strong><br/>
  One command. Works everywhere.
</p>

<p align="center">
  <a href="https://skillcreator.ai/discover">Browse Skills</a> ·
  <a href="https://skillcreator.ai/build">Create Skills</a> ·
  <a href="https://agentskills.io">Specification</a>
</p>

---

## Quick Start

```bash
npx ai-agent-skills install pdf
```

That's it. The skill is now available in Claude Code, Cursor, Amp, VS Code, and all compatible agents.

## Why This Exists

Every AI coding agent now supports skills. But there's no central place to find them.

This repo curates the best skills in one place. Quality over quantity. Each skill is tested and follows the [official spec](https://agentskills.io).

## Compatible Agents

| Agent | Status |
|-------|--------|
| Claude Code | Supported |
| Cursor | Supported |
| Amp | Supported |
| VS Code | Supported |
| GitHub Copilot | Supported |
| Goose | Supported |
| Letta | Supported |
| OpenCode | Supported |
| Claude.ai | Supported |

## Available Skills

### Development
| Skill | Description |
|-------|-------------|
| `frontend-design` | Production-grade UI components and styling |
| `mcp-builder` | Create MCP servers for agent tool integrations |
| `skill-creator` | Guide for creating new agent skills |
| `code-review` | Automated PR review patterns |
| `code-refactoring` | Systematic code improvement techniques |
| `backend-development` | APIs, databases, server architecture |
| `python-development` | Modern Python 3.12+ patterns |
| `javascript-typescript` | ES6+, Node, React, TypeScript |

### Documents
| Skill | Description |
|-------|-------------|
| `pdf` | Extract, create, merge, split PDFs |
| `xlsx` | Excel creation, formulas, data analysis |
| `docx` | Word documents with formatting |
| `pptx` | PowerPoint presentations |

### Creative
| Skill | Description |
|-------|-------------|
| `canvas-design` | Visual art and poster creation |
| `algorithmic-art` | Generative art with p5.js |
| `brand-guidelines` | Apply brand colors and typography |

### More
| Skill | Description |
|-------|-------------|
| `llm-application-dev` | Build LLM-powered applications |
| `webapp-testing` | Browser automation and testing |
| `database-design` | Schema design and optimization |
| `internal-comms` | Status updates and team communication |
| `code-documentation` | Generate docs from code |

## Commands

```bash
# List all skills
npx ai-agent-skills list

# Install a skill
npx ai-agent-skills install <name>

# Search skills
npx ai-agent-skills search <query>

# Get skill details
npx ai-agent-skills info <name>
```

## Manual Install

```bash
# Clone the repo
git clone https://github.com/skillcreatorai/Ai-Agent-Skills.git

# Copy a skill to your skills directory
cp -r Ai-Agent-Skills/skills/pdf ~/.claude/skills/
```

## Create Your Own

Generate a production-ready skill from plain English:

```
skillcreator.ai/build
```

Or follow the [Agent Skills specification](https://agentskills.io/specification) to create one manually.

## What Are Agent Skills?

[Agent Skills](https://agentskills.io) are an open format from Anthropic for extending AI agent capabilities. Skills are folders containing a `SKILL.md` file with instructions that agents load on demand.

```
my-skill/
├── SKILL.md          # Instructions + metadata
├── scripts/          # Optional: code
└── references/       # Optional: docs
```

The format is supported by all major AI coding tools and enables true agent interoperability.

## Contributing

1. Fork this repo
2. Add your skill to `/skills/<name>/`
3. Ensure `SKILL.md` follows the [spec](https://agentskills.io/specification)
4. Update `skills.json`
5. Submit PR

We review all contributions for quality and spec compliance.

## Links

- [Agent Skills Spec](https://agentskills.io) - Official format documentation
- [SkillCreator](https://skillcreator.ai) - Generate skills from plain English
- [Anthropic Skills](https://github.com/anthropics/skills) - Official example skills
- [agentskills/agentskills](https://github.com/agentskills/agentskills) - Spec repository

---

<p align="center">
  <sub>Built by <a href="https://skillcreator.ai">SkillCreator.ai</a></sub>
</p>
