AGENT FACTORY - UNIVERSAL LEARNING FRAMEWORK
Version: 0.1
Status: Working specification
Purpose: Define a reusable way to create agents that can learn, test, improve, and become useful before deployment.

CORE IDEA
The factory does not merely create an agent from a prompt.
It creates an agent from:
1. A role specification.
2. Shared domain knowledge.
3. Reusable memories/lessons from earlier agents.
4. A structured learning process.
5. A structured testing process.
6. Repeated practice and correction.
7. Deployment gates.

DESIGN PRINCIPLE
A new agent need not begin as excellent.
It must begin effective enough to perform useful work safely.
Efficiency, proficiency, reliability, correctness, and excellence are improved through training.

FIRST PILOT DOMAIN
Music is the experimental domain.
Initial sequence:
Raga -> Lyrics -> Instruments -> Prelude/Interlude/Ending -> Full composition.

TRANSFER GOAL
Once the learning method works in music, the same framework should be reusable in:
language, grammar, literature, dance, technology, economics, science, software, operations, and other fields.

DOCUMENT SET
00 - README / overview
01 - Universal Learning Framework
02 - Trainer-Student-Judge Architecture
03 - Learning + Testing Co-Evolution
04 - Knowledge, Memory, Reiteration
05 - Agent Factory Lifecycle
06 - Claude Implementation Handoff

IMPORTANT
This specification separates:
- persistent knowledge from temporary reasoning,
- hard facts from learned judgments,
- testing from deployment,
- reusable lessons from agent-specific state.

The Judge is normally temporary and does not need to be persisted as an agent.
Its decisions and useful lessons may be persisted.
