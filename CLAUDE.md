# Project Skills and Guidelines

This project uses multiple skill frameworks to improve code quality and development practices.

## Available Skills

### 1. Ponytail (Lazy Senior Developer)
**Location**: `.claude/skills/ponytail/`

Philosophy: Write the minimum code that actually works. Follow the decision ladder:
1. Does this need to be built at all? (YAGNI)
2. Does it already exist in this codebase?
3. Does the standard library already do this?
4. Does a native platform feature cover it?
5. Does an already-installed dependency solve it?
6. Can this be one line?
7. Only then: write the minimum code that works

**When to use**: Before writing any code, when considering implementation approaches.

### 2. Andrej Karpathy Skills
**Location**: `.claude/skills/karpathy/`

Core principles for better coding:
1. **Think Before Coding**: State assumptions, surface tradeoffs, ask when uncertain
2. **Simplicity First**: Minimum code, no speculative features or abstractions
3. **Surgical Changes**: Touch only what you must, match existing style
4. **Goal-Driven Execution**: Define success criteria, verify with tests

**When to use**: For all coding tasks, especially refactoring and implementation.

### 3. Superpowers Framework
**Location**: `.claude/skills/superpowers/`

Comprehensive agentic skills including:
- Test-driven development
- Systematic debugging
- Subagent-driven development
- Writing plans and executing them
- Code review processes
- Git worktree management
- Brainstorming and planning

**When to use**: For complex tasks requiring structured approaches, debugging, or planning.

## Usage Guidelines

- These skills are automatically available to Claude Code
- Reference them explicitly when tackling appropriate tasks
- Skills can be combined (e.g., use ponytail's efficiency with superpowers' TDD)
- Check `.claude/skills/` for detailed documentation

## Project-Specific Notes

- Primary language: Python
- Focus: Job recommendation system with embeddings
- Data sources: Hugging Face (BAAI/bge-m3), Kaggle job datasets

---
*Skills loaded from local .claude/skills/ directory*