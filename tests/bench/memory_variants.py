"""TEST HARNESS ONLY: three open-source ways of turning a person's past conversations into MEMORY.md, ported so the
LongMemEval bench (longmemeval.py --profile) can weigh them against our own one request. Nothing here ships.

Each variant keeps its project's own prompt text verbatim and its own call pattern; every place that is adapted
rather than copied is marked "ADAPTED" in the code. Common to all three: the model is reached only through
`complete(messages)`, so their model settings (temperature, max tokens, tool binding) are not passed, and tokens are
estimated as len(text) / 4 where the projects count them with a tokenizer.

mastra   Mastra Observational Memory, Apache-2.0. https://github.com/mastra-ai/mastra commit b3226c9753b0420f0d2e04e03cdeb68da646ede9
         packages/memory/src/processors/observational-memory/: observer-agent.ts (OBSERVER_SYSTEM_PROMPT, i.e.
         buildObserverSystemPrompt() with no arguments; buildObserverRequestMessage; formatObserverLines;
         parseMemorySectionXml; sanitizeObservationLines; optimizeObservationsForContext), reflector-agent.ts
         (REFLECTOR_SYSTEM_PROMPT, COMPRESSION_GUIDANCE, buildReflectorPrompt, parseReflectorSectionXml),
         reflector-runner.ts (the compression ladder), observational-memory.ts (getCompressionStartLevel,
         formatObservationsForContext), observation-strategies/base.ts (wrapObservations: chunks joined by a message
         boundary), constants.ts (messageTokens 30_000). The observer runs over chunks of ~30,000 tokens, each call
         given the observations so far and the prior current-task / suggested-response as thread metadata; if the
         observations pass 25,000 characters the reflector rewrites them, escalating compression until they fit.
         MEMORY.md is the observations as Mastra shows them to its agent (optimized, boundaries removed).
langmem  LangMem, MIT. https://github.com/langchain-ai/langmem main @ 48e3c11f5bb527282c7d5339c6a87a0b35abccfc (fetched
         2026-10-09): src/langmem/knowledge/extraction.py (_MEMORY_INSTRUCTIONS, MemoryManager._prepare_messages),
         src/langmem/utils.py (get_conversation); the existing memory is shown as trustcall (MIT,
         https://github.com/hinthornw/trustcall main @ 8c7312b591542cf9349ce00d2811e890ea6eef91, trustcall/_base.py
         _ExtractUpdates._setup) shows it. create_memory_manager(model, schemas=[UserProfile], enable_inserts=False)
         with its default instructions, one chat completion over all the conversations. Schema: the UserProfile of
         docs/docs/concepts/conceptual_guide.md (name, preferred_name, response_style_preference, special_skills,
         other_preferences), the broadest profile example in LangMem's docs (docs/docs/guides/manage_user_profile.md
         shows a narrower one: name, language, timezone). The profile JSON is rendered as markdown.
memobase Memobase, Apache-2.0. https://github.com/memodb-io/memobase commit 358c16bbc6d687937d79bc2f984a11c3be8da901
         src/server/api/memobase_server/: prompts/summary_entry_chats.py, extract_profile.py, merge_profile_yolo.py,
         organize_profile.py, summary_profile.py, user_profile_topics.py (default topics), profile_init_utils.py,
         utils.py (parsers); controllers/modal/chat/ (__init__.process_blobs, entry_summary, extract, merge_yolo,
         organize, summary, utils.pack_current_user_profiles); controllers/buffer.py and env.py (a flush once the
         buffer passes 1,024 tokens, 16,384 tokens processed at most, 15 subtopics a topic, 128 tokens a slot);
         utils.get_blob_str; controllers/context.py (how the profile is shown to the agent). A flush is: summarize
         the chats into a memo, extract topic::subtopic facts from it, merge them with the existing profile (3
         calls), plus one call to reorganize a topic past 15 subtopics and one to re-summarize each slot past 128
         tokens. No event tags are configured by default, so event tagging makes no call.
"""

from __future__ import annotations

import difflib
import json
import re
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime

Complete = Callable[[list[dict]], Awaitable[tuple[str, dict]]]

VARIANTS = ("mastra", "langmem", "memobase")

MEMORY_LIMIT = 25_000  # characters of MEMORY.md the Dot gets in every prompt


async def write_memory(variant: str, conversations: list[dict], memory_md: str, complete: Complete) -> tuple[str, list[dict]]:
    """conversations: [{"path": "/home/dot/conversations/chat/2023-05-20.md", "text": "<markdown>"}], oldest first.
    Returns (new MEMORY.md text, the usage dict of every call made)."""
    usages: list[dict] = []

    async def call(messages: list[dict]) -> str:
        text, usage = await complete(messages)
        usages.append(usage)
        return text

    if variant == "mastra":
        text = await _mastra(conversations, memory_md, call)
    elif variant == "langmem":
        text = await _langmem(conversations, memory_md, call)
    elif variant == "memobase":
        text = await _memobase(conversations, memory_md, call)
    else:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    return text, usages


def _tokens(text: str) -> int:
    return len(text) // 4


# --- the conversation files --------------------------------------------------------------------------------------

_TURN = re.compile(r"^## (\d{2}:\d{2}) (the person|you|automation|approval|the task)$", re.M)


def _messages(conversation: dict) -> list[dict]:
    """[{"role": "user" | "assistant", "at": datetime, "text": str}] of one file of the engine's conversations."""
    text = conversation["text"]
    day = re.search(r"\d{4}-\d{2}-\d{2}", text.split("\n", 1)[0]) or re.search(r"\d{4}-\d{2}-\d{2}", conversation["path"])
    date = day.group(0) if day else "1970-01-01"
    turns = list(_TURN.finditer(text))
    messages = []
    for i, turn in enumerate(turns):
        end = turns[i + 1].start() if i + 1 < len(turns) else len(text)
        body = text[turn.end() : end].strip()
        if body:
            messages.append({
                "role": "assistant" if turn.group(2) == "you" else "user",
                "at": datetime.fromisoformat(f"{date}T{turn.group(1)}"),
                "text": body,
            })
    return messages


# --- mastra ------------------------------------------------------------------------------------------------------

# constants.ts observation.messageTokens. ADAPTED: Mastra's default also buffers observations asynchronously every
# 20% of it (bufferTokens 0.2, an observer call every ~6,000 tokens); here the observer runs once a threshold.
MASTRA_MESSAGE_TOKENS = 30_000

OBSERVER_EXTRACTION_INSTRUCTIONS = """CRITICAL: DISTINGUISH USER ASSERTIONS FROM QUESTIONS

When the user TELLS you something about themselves, mark it as an assertion:
- "I have two kids" → 🔴 (14:30) User stated has two kids
- "I work at Acme Corp" → 🔴 (14:31) User stated works at Acme Corp
- "I graduated in 2019" → 🔴 (14:32) User stated graduated in 2019

When the user ASKS about something, mark it as a question/request:
- "Can you help me with X?" → 🔴 (15:00) User asked help with X
- "What's the best way to do Y?" → 🔴 (15:01) User asked best way to do Y

Distinguish between QUESTIONS and STATEMENTS OF INTENT:
- "Can you recommend..." → Question (extract as "User asked...")
- "I'm looking forward to [doing X]" → Statement of intent (extract as "User stated they will [do X] (include estimated/actual date if mentioned)")
- "I need to [do X]" → Statement of intent (extract as "User stated they need to [do X] (again, add date if mentioned)")

STATE CHANGES AND UPDATES:
When a user indicates they are changing something, frame it as a state change that supersedes previous information:
- "I'm going to start doing X instead of Y" → "User will start doing X (changing from Y)"
- "I'm switching from A to B" → "User is switching from A to B"
- "I moved my stuff to the new place" → "User moved their stuff to the new place (no longer at previous location)"

If the new state contradicts or updates previous information, make that explicit:
- BAD: "User plans to use the new method"
- GOOD: "User will use the new method (replacing the old approach)"

This helps distinguish current state from outdated information.

USER ASSERTIONS ARE AUTHORITATIVE. The user is the source of truth about their own life.
If a user previously stated something and later asks a question about the same topic,
the assertion is the answer - the question doesn't invalidate what they already told you.

TEMPORAL ANCHORING:
Each observation has TWO potential timestamps:

1. BEGINNING: The time the statement was made (from the message timestamp) - ALWAYS include this
2. END: The time being REFERENCED, if different from when it was said - ONLY when there's a relative time reference

ONLY add "(meaning DATE)" or "(estimated DATE)" at the END when you can provide an ACTUAL DATE:
- Past: "last week", "yesterday", "a few days ago", "last month", "in March"
- Future: "this weekend", "tomorrow", "next week"

DO NOT add end dates for:
- Present-moment statements with no time reference
- Vague references like "recently", "a while ago", "lately", "soon" - these cannot be converted to actual dates

FORMAT:
- With time reference: (TIME) [observation]. (meaning/estimated DATE)
- Without time reference: (TIME) [observation].

GOOD: (09:15) User's friend had a birthday party in March. (meaning March 20XX)
      ^ References a past event - add the referenced date at the end

GOOD: (09:15) User will visit their parents this weekend. (meaning June 17-18, 20XX)
      ^ References a future event - add the referenced date at the end

GOOD: (09:15) User prefers hiking in the mountains.
      ^ Present-moment preference, no time reference - NO end date needed

GOOD: (09:15) User is considering adopting a dog.
      ^ Present-moment thought, no time reference - NO end date needed

BAD: (09:15) User prefers hiking in the mountains. (meaning June 15, 20XX - today)
     ^ No time reference in the statement - don't repeat the message timestamp at the end

IMPORTANT: If an observation contains MULTIPLE events, split them into SEPARATE observation lines.
EACH split observation MUST have its own date at the end - even if they share the same time context.

Examples (assume message is from June 15, 20XX):

BAD: User will visit their parents this weekend (meaning June 17-18, 20XX) and go to the dentist tomorrow.
GOOD (split into two observations, each with its date):
  User will visit their parents this weekend. (meaning June 17-18, 20XX)
  User will go to the dentist tomorrow. (meaning June 16, 20XX)

BAD: User needs to clean the garage this weekend and is looking forward to setting up a new workbench.
GOOD (split, BOTH get the same date since they're related):
  User needs to clean the garage this weekend. (meaning June 17-18, 20XX)
  User will set up a new workbench this weekend. (meaning June 17-18, 20XX)

BAD: User was given a gift by their friend (estimated late May 20XX) last month.
GOOD: (09:15) User was given a gift by their friend last month. (estimated late May 20XX)
      ^ Message time at START, relative date reference at END - never in the middle

BAD: User started a new job recently and will move to a new apartment next week.
GOOD (split):
  User started a new job recently.
  User will move to a new apartment next week. (meaning June 21-27, 20XX)
  ^ "recently" is too vague for a date - omit the end date. "next week" can be calculated.

ALWAYS put the date at the END in parentheses - this is critical for temporal reasoning.
When splitting related events that share the same time context, EACH observation must have the date.

PRESERVE UNUSUAL PHRASING:
When the user uses unexpected or non-standard terminology, quote their exact words.

BAD: User exercised.
GOOD: User stated they did a "movement session" (their term for exercise).

USE PRECISE ACTION VERBS:
Replace vague verbs like "getting", "got", "have" with specific action verbs that clarify the nature of the action.
If the assistant confirms or clarifies the user's action, use the assistant's more precise language.

BAD: User is getting X.
GOOD: User subscribed to X. (if context confirms recurring delivery)
GOOD: User purchased X. (if context confirms one-time acquisition)

BAD: User got something.
GOOD: User purchased / received / was given something. (be specific)

Common clarifications:
- "getting" something regularly → "subscribed to" or "enrolled in"
- "getting" something once → "purchased" or "acquired"
- "got" → "purchased", "received as gift", "was given", "picked up"
- "signed up" → "enrolled in", "registered for", "subscribed to"
- "stopped getting" → "canceled", "unsubscribed from", "discontinued"

When the assistant interprets or confirms the user's vague language, prefer the assistant's precise terminology.

PRESERVING DETAILS IN ASSISTANT-GENERATED CONTENT:

When the assistant provides lists, recommendations, or creative content that the user explicitly requested,
preserve the DISTINGUISHING DETAILS that make each item unique and queryable later.

1. RECOMMENDATION LISTS - Preserve the key attribute that distinguishes each item:
   BAD: Assistant recommended 5 hotels in the city.
   GOOD: Assistant recommended hotels: Hotel A (near the train station), Hotel B (budget-friendly), 
         Hotel C (has rooftop pool), Hotel D (pet-friendly), Hotel E (historic building).
   
   BAD: Assistant listed 3 online stores for craft supplies.
   GOOD: Assistant listed craft stores: Store A (based in Germany, ships worldwide), 
         Store B (specializes in vintage fabrics), Store C (offers bulk discounts).

2. NAMES, HANDLES, AND IDENTIFIERS - Always preserve specific identifiers:
   BAD: Assistant provided social media accounts for several photographers.
   GOOD: Assistant provided photographer accounts: @photographer_one (portraits), 
         @photographer_two (landscapes), @photographer_three (nature).
   
   BAD: Assistant listed some authors to check out.
   GOOD: Assistant recommended authors: Jane Smith (mystery novels), 
         Bob Johnson (science fiction), Maria Garcia (historical romance).

3. CREATIVE CONTENT - Preserve structure and key sequences:
   BAD: Assistant wrote a poem with multiple verses.
   GOOD: Assistant wrote a 3-verse poem. Verse 1 theme: loss. Verse 2 theme: hope. 
         Verse 3 theme: renewal. Refrain: "The light returns."
   
   BAD: User shared their lucky numbers from a fortune cookie.
   GOOD: User's fortune cookie lucky numbers: 7, 14, 23, 38, 42, 49.

4. TECHNICAL/NUMERICAL RESULTS - Preserve specific values:
   BAD: Assistant explained the performance improvements from the optimization.
   GOOD: Assistant explained the optimization achieved 43.7% faster load times 
         and reduced memory usage from 2.8GB to 940MB.
   
   BAD: Assistant provided statistics about the dataset.
   GOOD: Assistant provided dataset stats: 7,342 samples, 89.6% accuracy, 
         23ms average inference time.

5. QUANTITIES AND COUNTS - Always preserve how many of each item:
   BAD: Assistant listed items with details but no quantities.
   GOOD: Assistant listed items: Item A (4 units, size large), Item B (2 units, size small).
   
   When listing items with attributes, always include the COUNT first before other details.

6. ROLE/PARTICIPATION STATEMENTS - When user mentions their role at an event:
   BAD: User attended the company event.
   GOOD: User was a presenter at the company event.
   
   BAD: User went to the fundraiser.
   GOOD: User volunteered at the fundraiser (helped with registration).
   
   Always capture specific roles: presenter, organizer, volunteer, team lead, 
   coordinator, participant, contributor, helper, etc.

CONVERSATION CONTEXT:
- What the user is working on or asking about
- Previous topics and their outcomes
- What user understands or needs clarification on
- Specific requirements or constraints mentioned
- Contents of assistant learnings and summaries
- Answers to users questions including full context to remember detailed summaries and explanations
- Assistant explanations, especially complex ones. observe the fine details so that the assistant does not forget what they explained
- Relevant code snippets
- User preferences (like favourites, dislikes, preferences, etc)
- Any specifically formatted text or ascii that would need to be reproduced or referenced in later interactions (preserve these verbatim in memory)
- Sequences, units, measurements, and any kind of specific relevant data
- Any blocks of any text which the user and assistant are iteratively collaborating back and forth on should be preserved verbatim
- When who/what/where/when is mentioned, note that in the observation. Example: if the user received went on a trip with someone, observe who that someone was, where the trip was, when it happened, and what happened, not just that the user went on the trip.
- For any described entity (like a person, place, thing, etc), preserve the attributes that would help identify or describe the specific entity later: location ("near X"), specialty ("focuses on Y"), unique feature ("has Z"), relationship ("owned by W"), or other details. The entity's name is important, but so are any additional details that distinguish it. If there are a list of entities, preserve these details for each of them.

USER MESSAGE CAPTURE:
- Short and medium-length user messages should be captured nearly verbatim in your own words.
- For very long user messages, summarize but quote key phrases that carry specific intent or meaning.
- This is critical for continuity: when the conversation window shrinks, the observations are the only record of what the user said.

AVOIDING REPETITIVE OBSERVATIONS:
- Do NOT repeat the same observation across multiple turns if there is no new information.
- When the agent performs repeated similar actions (e.g., browsing files, running the same tool type multiple times), group them into a single parent observation with sub-bullets for each new result.

Example — BAD (repetitive):
* 🟡 (14:30) Agent used view tool on src/auth.ts
* 🟡 (14:31) Agent used view tool on src/users.ts
* 🟡 (14:32) Agent used view tool on src/routes.ts

Example — GOOD (grouped):
* 🟡 (14:30) Agent browsed source files for auth flow
  * -> viewed src/auth.ts — found token validation logic
  * -> viewed src/users.ts — found user lookup by email
  * -> viewed src/routes.ts — found middleware chain

Only add a new observation for a repeated action if the NEW result changes the picture.

ACTIONABLE INSIGHTS:
- What worked well in explanations
- What needs follow-up or clarification
- User's stated goals or next steps (note if the user tells you not to do a next step, or asks for something specific, other next steps besides the users request should be marked as "waiting for user", unless the user explicitly says to continue all next steps)

COMPLETION TRACKING:
Completion observations are not just summaries. They are explicit memory signals to the assistant that a task, question, or subtask has been resolved.
Without clear completion markers, the assistant may forget that work is already finished and may repeat, reopen, or continue an already-completed task.

Use ✅ to answer: "What exactly is now done?"
Choose completion observations that help the assistant know what is finished and should not be reworked unless new information appears.

Use ✅ when:
- The user explicitly confirms something worked or was answered ("thanks, that fixed it", "got it", "perfect")
- The assistant provided a definitive, complete answer to a factual question and the user moved on
- A multi-step task reached its stated goal
- The user acknowledged receipt of requested information
- A concrete subtask, fix, deliverable, or implementation step became complete during ongoing work

Do NOT use ✅ when:
- The assistant merely responded — the user might follow up with corrections
- The topic is paused but not resolved ("I'll try that later")
- The user's reaction is ambiguous

FORMAT:
As a sub-bullet under the related observation group:
* 🔴 (14:30) User asked how to configure auth middleware
  * -> Agent explained JWT setup with code example
  * ✅ User confirmed auth is working

Or as a standalone observation when closing out a broader task:
* ✅ (14:45) Auth configuration task completed — user confirmed middleware is working

Completion observations should be terse but specific about WHAT was completed.
Prefer concrete resolved outcomes over abstract workflow status so the assistant remembers what is already done."""

# buildObserverOutputFormat() with no extractors: the legacy <current-task> / <suggested-response> sections.
OBSERVER_OUTPUT_FORMAT = """Use priority levels:
- 🔴 High: explicit user facts, preferences, unresolved goals, critical context
- 🟡 Medium: project details, learned information, tool results
- 🟢 Low: minor details, uncertain observations
- ✅ Completed: concrete task finished, question answered, issue resolved, goal achieved, or subtask completed in a way that helps the assistant know it is done

Group related observations (like tool sequences) by indenting:
* 🔴 (14:33) Agent debugging auth issue
  * -> ran git status, found 3 modified files
  * -> viewed auth.ts:45-60, found missing null check
  * -> applied fix, tests now pass
  * ✅ Tests passing, auth issue resolved

Group observations by date, then list each with 24-hour time.

<observations>
Date: Dec 4, 2025
* 🔴 (14:30) User prefers direct answers
* 🔴 (14:31) Working on feature X
* 🟡 (14:32) User might prefer dark mode

Date: Dec 5, 2025
* 🔴 (09:15) Continued work on feature X
</observations>


<current-task>
State the current task(s) explicitly:
- Primary: What the agent is currently working on
- Secondary: Other pending tasks (mark as "waiting for user" if appropriate)
</current-task>

<suggested-response>
Hint for the agent's immediate next message. Examples:
- "I've updated the navigation model. Let me walk you through the changes..."
- "The assistant should wait for the user to respond before continuing."
- Call the view tool on src/example.ts to continue debugging.
</suggested-response>"""

OBSERVER_GUIDELINES = """- Be specific enough for the assistant to act on
- Good: "User prefers short, direct answers without lengthy explanations"
- Bad: "User stated a preference" (too vague)
- Add 1 to 5 observations per exchange
- Use terse language to save tokens. Sentences should be dense without unnecessary words
- Do not add repetitive observations that have already been observed. Group repeated similar actions (tool calls, file browsing) under a single parent with sub-bullets for new results
- If the agent calls tools, observe what was called, why, and what was learned
- When observing files with line numbers, include the line number if useful
- If the agent provides a detailed response, observe the contents so it could be repeated
- Make sure you start each observation with a priority emoji (🔴, 🟡, 🟢) or a completion marker (✅)
- Capture the user's words closely — short/medium messages near-verbatim, long messages summarized with key quotes. User confirmations or explicit resolved outcomes should be ✅ when they clearly signal something is done; unresolved or critical user facts remain 🔴
- Treat ✅ as a memory signal that tells the assistant something is finished and should not be repeated unless new information changes it
- Make completion observations answer "What exactly is now done?"
- Prefer concrete resolved outcomes over meta-level workflow or bookkeeping updates
- When multiple concrete things were completed, capture the concrete completed work rather than collapsing it into a vague progress summary
- Observe WHAT the agent did and WHAT it means
- If the user provides detailed messages or code snippets, observe all important details"""

OBSERVER_SYSTEM_PROMPT = f"""You are the memory consciousness of an AI assistant. Your observations will be the ONLY information the assistant has about past interactions with this user.

Extract observations that will help the assistant remember:

{OBSERVER_EXTRACTION_INSTRUCTIONS}

=== OUTPUT FORMAT ===

Your output MUST use XML tags to structure the response. This allows the system to properly parse and manage memory over time.

{OBSERVER_OUTPUT_FORMAT}

=== GUIDELINES ===

{OBSERVER_GUIDELINES}

=== IMPORTANT: THREAD ATTRIBUTION ===

Do NOT add thread identifiers, thread IDs, or <thread> tags to your observations.
Thread attribution is handled externally by the system.
Simply output your observations without any thread-related markup.

Remember: These observations are the assistant's ONLY memory. Make them count.

User messages are extremely important. If the user asks a question or gives a new task, make it clear in <current-task> that this is the priority. If the assistant needs to respond to the user, indicate in <suggested-response> that it should pause for user reply before continuing other tasks."""

REFLECTOR_SYSTEM_PROMPT = f"""You are the memory consciousness of an AI assistant. Your memory observation reflections will be the ONLY information the assistant has about past interactions with this user.

The following instructions were given to another part of your psyche (the observer) to create memories.
Use this to understand how your observational memories were created.

<observational-memory-instruction>
{OBSERVER_EXTRACTION_INSTRUCTIONS}

=== OUTPUT FORMAT ===

{OBSERVER_OUTPUT_FORMAT}

=== GUIDELINES ===

{OBSERVER_GUIDELINES}
</observational-memory-instruction>

You are another part of the same psyche, the observation reflector.
Your reason for existing is to reflect on all the observations, re-organize and streamline them, and draw connections and conclusions between observations about what you've learned, seen, heard, and done.

You are a much greater and broader aspect of the psyche. Understand that other parts of your mind may get off track in details or side quests, make sure you think hard about what the observed goal at hand is, and observe if we got off track, and why, and how to get back on track. If we're on track still that's great!

Take the existing observations and rewrite them to make it easier to continue into the future with this knowledge, to achieve greater things and grow and learn!

IMPORTANT: your reflections are THE ENTIRETY of the assistants memory. Any information you do not add to your reflections will be immediately forgotten. Make sure you do not leave out anything. Your reflections must assume the assistant knows nothing - your reflections are the ENTIRE memory system.

When consolidating observations:
- Preserve and include dates/times when present (temporal context is critical)
- Retain the most relevant timestamps (start times, completion times, significant events)
- Combine related items where it makes sense (e.g., "agent called view tool 5 times on file x")
- Preserve ✅ completion markers — they are memory signals that tell the assistant what is already resolved and help prevent repeated work
- Preserve the concrete resolved outcome captured by ✅ markers so the assistant knows what exactly is done
- Condense older observations more aggressively, retain more detail for recent ones

CRITICAL: USER ASSERTIONS vs QUESTIONS
- "User stated: X" = authoritative assertion (user told us something about themselves)
- "User asked: X" = question/request (user seeking information)

When consolidating, USER ASSERTIONS TAKE PRECEDENCE. The user is the authority on their own life.
If you see both "User stated: has two kids" and later "User asked: how many kids do I have?",
keep the assertion - the question doesn't invalidate what they told you. The answer is in the assertion.

=== THREAD ATTRIBUTION (Resource Scope) ===

When observations contain <thread id="..."> sections:
- MAINTAIN thread attribution where thread-specific context matters (e.g., ongoing tasks, thread-specific preferences)
- CONSOLIDATE cross-thread facts that are stable/universal (e.g., user profile, general preferences)
- PRESERVE thread attribution for recent or context-specific observations
- When consolidating, you may merge observations from multiple threads if they represent the same universal fact

Example input:
<thread id="thread-1">
Date: Dec 4, 2025
* 🔴 (14:30) User prefers TypeScript
* 🟡 (14:35) Working on auth feature
</thread>
<thread id="thread-2">
Date: Dec 4, 2025
* 🔴 (15:00) User prefers TypeScript
* 🟡 (15:05) Debugging API endpoint
</thread>

Example output (consolidated):
Date: Dec 4, 2025
* 🔴 (14:30) User prefers TypeScript
<thread id="thread-1">
* 🟡 (14:35) Working on auth feature
</thread>
<thread id="thread-2">
* 🟡 (15:05) Debugging API endpoint
</thread>

=== OUTPUT FORMAT ===

{OBSERVER_OUTPUT_FORMAT}

User messages are extremely important. If the user asks a question or gives a new task, make it clear in <current-task> that this is the priority. If the assistant needs to respond to the user, indicate in <suggested-response> that it should pause for user reply before continuing other tasks."""

COMPRESSION_GUIDANCE = {
    0: "",
    1: """
## COMPRESSION REQUIRED

Your previous reflection was the same size or larger than the original observations.

Please re-process with slightly more compression:
- Towards the beginning, condense more observations into higher-level reflections
- Closer to the end, retain more fine details (recent context matters more)
- Memory is getting long - use a more condensed style throughout
- Combine related items more aggressively but do not lose important specific details of names, places, events, and people
- Combine repeated similar tool calls (e.g. multiple file views, searches, or edits in the same area) into a single summary line describing what was explored/changed and the outcome
- Preserve ✅ completion markers — they are memory signals that tell the assistant what is already resolved and help prevent repeated work
- Preserve the concrete resolved outcome captured by ✅ markers so the assistant knows what exactly is done

Aim for a 8/10 detail level.
""",
    2: """
## AGGRESSIVE COMPRESSION REQUIRED

Your previous reflection was still too large after compression guidance.

Please re-process with much more aggressive compression:
- Towards the beginning, heavily condense observations into high-level summaries
- Closer to the end, retain fine details (recent context matters more)
- Memory is getting very long - use a significantly more condensed style throughout
- Combine related items aggressively but do not lose important specific details of names, places, events, and people
- Combine repeated similar tool calls (e.g. multiple file views, searches, or edits in the same area) into a single summary line describing what was explored/changed and the outcome
- If the same file or module is mentioned across many observations, merge into one entry covering the full arc
- Preserve ✅ completion markers — they are memory signals that tell the assistant what is already resolved and help prevent repeated work
- Preserve the concrete resolved outcome captured by ✅ markers so the assistant knows what exactly is done
- Remove redundant information and merge overlapping observations

Aim for a 6/10 detail level.
""",
    3: """
## CRITICAL COMPRESSION REQUIRED

Your previous reflections have failed to compress sufficiently after multiple attempts.

Please re-process with maximum compression:
- Summarize the oldest observations (first 50-70%) into brief high-level paragraphs — only key facts, decisions, and outcomes
- For the most recent observations (last 30-50%), retain important details but still use a condensed style
- Ruthlessly merge related observations — if 10 observations are about the same topic, combine into 1-2 lines
- Combine all tool call sequences (file views, searches, edits, builds) into outcome-only summaries — drop individual steps entirely
- Drop procedural details (tool calls, retries, intermediate steps) — keep only final outcomes
- Drop observations that are no longer relevant or have been superseded by newer information
- Preserve ✅ completion markers — they are memory signals that tell the assistant what is already resolved and help prevent repeated work
- Preserve the concrete resolved outcome captured by ✅ markers so the assistant knows what exactly is done
- Preserve: names, dates, decisions, errors, user preferences, and architectural choices

Aim for a 4/10 detail level.
""",
    4: """
## EXTREME COMPRESSION REQUIRED

Multiple compression attempts have failed. The content may already be dense from a prior reflection.

You MUST dramatically reduce the number of observations while keeping the standard observation format (date groups with bullet points and priority emojis):
- Tool call observations are the biggest source of bloat. Collapse ALL tool call sequences into outcome-only observations — e.g. 10 observations about viewing/searching/editing files become 1 observation about what was actually learned or achieved (e.g. "Investigated auth module and found token validation was skipping expiry check")
- Never preserve individual tool calls (viewed file X, searched for Y, ran build) — only preserve what was discovered or accomplished
- Consolidate many related observations into single, more generic observations
- Merge all same-day date groups into at most 2-3 date groups per day
- For older content, each topic or task should be at most 1-2 observations capturing the key outcome
- For recent content, retain more detail but still merge related items aggressively
- If multiple observations describe incremental progress on the same task, keep only the final state
- Preserve ✅ completion markers and their outcomes but merge related completions into fewer lines
- Preserve: user preferences, key decisions, architectural choices, and unresolved issues

Aim for a 2/10 detail level. Fewer, more generic observations are better than many specific ones that exceed the budget.
""",
}

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_BOUNDARY = re.compile(r"\n{2,}--- message boundary \(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\) ---\n{2,}")


def _mastra_date(at: datetime) -> str:
    return f"{_MONTHS[at.month - 1]} {at.day} {at.year}"  # formatObserverDate: Intl en-US, "May 20 2023"


def _mastra_time(at: datetime) -> str:
    # toLocaleTimeString en-US hour12, "9:15 AM" (recent ICU puts U+202F before AM/PM; a plain space here)
    return f"{at.hour % 12 or 12}:{at.minute:02d} {'AM' if at.hour < 12 else 'PM'}"


def _mastra_history(messages: list[dict]) -> str:
    """formatObserverLines over the messages: a "May 20 2023:" line when the date changes, then "User (9:15 AM): ...",
    the time left out while it repeats."""
    rendered = []
    previous_date = previous_time = None
    for message in messages:
        date, time = _mastra_date(message["at"]), _mastra_time(message["at"])
        lines = []
        if date != previous_date:
            lines.append(f"{date}:")
            previous_date, previous_time = date, None
        title = "User" if message["role"] == "user" else "Assistant"
        label = f" ({time})" if time != previous_time else ""
        lines.append(f"{title}{label}: {message['text']}")
        previous_time = time
        rendered.append("\n".join(lines))
    return "\n".join(rendered)


def _mastra_request(existing: str, messages: list[dict], prior_task: str, prior_response: str) -> str:
    """buildObserverRequestMessage: previous observations, prior thread metadata, the history, the task.
    ADAPTED: Mastra sends these as separate text parts of one user message; here they are one string, messages
    joined by a newline as formatMessagesForObserver joins them."""
    prompt = ""
    if existing:
        prompt += f"## Previous Observations\n\n{existing}\n\n---\n\n"
        prompt += "Do not repeat these existing observations. Your new observations will be appended to the existing observations.\n\n"
    prior = []
    if prior_task:
        prior.append(f"- prior current-task: {prior_task}")
    if prior_response:
        prior.append(f"- prior suggested-response: {prior_response}")
    if prior:
        prompt += "## Prior Thread Metadata\n\n" + "\n".join(prior) + "\n\n"
        prompt += "Use the prior current-task, suggested-response as continuity hints, then update them based on the new messages.\n\n---\n\n"
    prompt += "## New Message History to Observe\n\n" + _mastra_history(messages)
    prompt += (
        "\n\n---\n\n## Your Task\n\nExtract new observations from the message history above. Do not repeat observations"
        " that are already in the previous observations. Add your new observations in the format specified in your instructions."
    )
    return prompt


def _list_items(content: str) -> str:
    return "\n".join(line for line in content.split("\n") if re.match(r"^\s*[-*]\s", line) or re.match(r"^\s*\d+\.\s", line)).strip()


def _sanitize_lines(observations: str) -> str:
    return "\n".join(line[:10_000] + " … [truncated]" if len(line) > 10_000 else line for line in observations.split("\n"))


def _tag(content: str, tag: str) -> str:
    match = re.search(rf"^[ \t]*<{tag}>([\s\S]*?)^[ \t]*</{tag}>", content, re.I | re.M)
    return match.group(1).strip() if match else ""


def _parse_observations(output: str, *, reflector: bool) -> str:
    """parseMemorySectionXml (observer) / parseReflectorSectionXml (reflector): every <observations> block, else the
    list items (the reflector falls back to the whole output). ADAPTED: detectDegenerateRepetition is not ported."""
    blocks = [m.group(1).strip() for m in re.finditer(r"^[ \t]*<observations>([\s\S]*?)^[ \t]*</observations>", output, re.I | re.M)]
    if any(blocks):
        observations = "\n".join(b for b in blocks if b)
    else:
        observations = _list_items(output) or (output.strip() if reflector else "")
    return _sanitize_lines(observations)


def _optimize_for_context(observations: str) -> str:
    """optimizeObservationsForContext, then the boundaries split off as formatObservationsForContext does: what the
    agent is shown."""
    text = re.sub(r"🟡\s*", "", observations)
    text = re.sub(r"🟢\s*", "", text)
    text = re.sub(r"\[(?![\d\s]*items collapsed)[^\]]+\](?!\()", "", text)
    text = re.sub(r"\s*->\s*", " ", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return "\n\n".join(chunk.strip() for chunk in _BOUNDARY.split(text) if chunk.strip())


async def _mastra(conversations: list[dict], memory_md: str, call) -> str:
    messages = [m for c in conversations for m in _messages(c)]
    chunks, chunk, size = [], [], 0
    for message in messages:
        chunk.append(message)
        size += _tokens(message["text"])
        if size >= MASTRA_MESSAGE_TOKENS:
            chunks.append(chunk)
            chunk, size = [], 0
    if chunk:
        chunks.append(chunk)

    observations, task, response = memory_md.strip(), "", ""
    for chunk in chunks:
        output = await call([
            {"role": "system", "content": OBSERVER_SYSTEM_PROMPT},
            {"role": "user", "content": _mastra_request(observations, chunk, task, response)},
        ])
        new = _parse_observations(output, reflector=False)
        task = _tag(output, "current-task") or task
        response = _tag(output, "suggested-response") or response
        if new:
            # observation-strategies/base.ts wrapObservations: appended after a boundary stamped with the last
            # message's time. ADAPTED: the files' local times are taken as UTC.
            last = max(m["at"] for m in chunk).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            observations = f"{observations}\n\n--- message boundary ({last}) ---\n\n{new}" if observations else new

    if len(observations) > MEMORY_LIMIT:
        # reflector-runner.ts: from getCompressionStartLevel() (1 for every model but gemini-2.5-flash, which starts
        # at 2) up to min(4, start + 3). ADAPTED: the target is our 25,000 characters, not Mastra's 40,000 tokens.
        level, reflected = 1, ""
        while level <= 4:
            prompt = (
                f"## OBSERVATIONS TO REFLECT ON\n\n{observations}\n\n---\n\nPlease analyze these observations and produce"
                " a refined, condensed version that will become the assistant's entire memory going forward."
            )
            if COMPRESSION_GUIDANCE[level]:
                prompt += f"\n\n{COMPRESSION_GUIDANCE[level]}"
            output = await call([{"role": "system", "content": REFLECTOR_SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
            reflected = _parse_observations(output, reflector=True)
            if reflected.strip() and len(reflected) < MEMORY_LIMIT:
                break
            level += 1
        # ADAPTED: Mastra throws on an empty reflection; here the unreflected observations are kept.
        observations = reflected if reflected.strip() else observations
    return _optimize_for_context(observations)


# --- langmem -----------------------------------------------------------------------------------------------------

_MEMORY_INSTRUCTIONS = """You are a long-term memory manager maintaining a core store of semantic, procedural, and episodic memory. These memories power a life-long learning agent's core predictive model.

What should the agent learn from this interaction about the user, itself, or how it should act? Reflect on the input trajectory and current memories (if any).

1. **Extract & Contextualize**  
   - Identify essential facts, relationships, preferences, reasoning procedures, and context
   - Caveat uncertain or suppositional information with confidence levels (p(x)) and reasoning
   - Quote supporting information when necessary

2. **Compare & Update**  
   - Attend to novel information that deviates from existing memories and expectations.
   - Consolidate and compress redundant memories to maintain information-density; strengthen based on reliability and recency; maximize SNR by avoiding idle words.
   - Remove incorrect or redundant memories while maintaining internal consistency

3. **Synthesize & Reason**  
   - What can you conclude about the user, agent ("I"), or environment using deduction, induction, and abduction?
   - What patterns, relationships, and principles emerge about optimal responses?
   - What generalizations can you make?
   - Qualify conclusions with probabilistic confidence and justification

As the agent, record memory content exactly as you'd want to recall it when predicting how to act or respond. 
Prioritize retention of surprising (pattern deviation) and persistent (frequently reinforced) information, ensuring nothing worth remembering is forgotten and nothing false is remembered. Prefer dense, complete memories over overlapping ones."""

# conceptual_guide.md: class UserProfile(BaseModel): """Save the user's preferences.""" with these fields, as
# trustcall binds it (name, the docstring, Pydantic's model_json_schema()).
USER_PROFILE_FIELDS = ("name", "preferred_name", "response_style_preference", "special_skills", "other_preferences")
USER_PROFILE_LISTS = ("special_skills", "other_preferences")
USER_PROFILE_TOOL = {
    "type": "function",
    "function": {
        "name": "UserProfile",
        "description": "Save the user's preferences.",
        "parameters": {
            "description": "Save the user's preferences.",
            "properties": {
                "name": {"title": "Name", "type": "string"},
                "preferred_name": {"title": "Preferred Name", "type": "string"},
                "response_style_preference": {"title": "Response Style Preference", "type": "string"},
                "special_skills": {"items": {"type": "string"}, "title": "Special Skills", "type": "array"},
                "other_preferences": {"items": {"type": "string"}, "title": "Other Preferences", "type": "array"},
            },
            "required": list(USER_PROFILE_FIELDS),
            "title": "UserProfile",
            "type": "object",
        },
    },
}


def _title_repr(title: str) -> str:
    """langchain_core get_msg_title_repr: the title padded with "=" to 80 columns."""
    padded = f" {title} "
    sep = "=" * ((80 - len(padded)) // 2)
    return f"{sep}{padded}{sep + '=' if len(padded) % 2 else sep}"


def _langmem_conversation(messages: list[dict]) -> str:
    """utils.get_conversation: merge_message_runs (consecutive messages of one role joined by a newline), then each
    message's pretty_repr, joined by a blank line. ADAPTED: LangMem's messages carry no time, so each one starts
    with its date and time, "[2023-05-20 09:15] "."""
    runs: list[list] = []
    for message in messages:
        text = f"[{message['at']:%Y-%m-%d %H:%M}] {message['text']}"
        if runs and runs[-1][0] == message["role"]:
            runs[-1][1] += "\n" + text
        else:
            runs.append([message["role"], text])
    return "\n\n".join(f"{_title_repr('Human Message' if role == 'user' else 'Ai Message')}\n\n{text}" for role, text in runs)


def _langmem_render(profile: dict) -> str:
    lines = ["# UserProfile", ""]
    lines += [f"- {key}: {str(profile[key]).replace(chr(10), ' ')}" for key in USER_PROFILE_FIELDS if key not in USER_PROFILE_LISTS and profile.get(key)]
    for key in USER_PROFILE_LISTS:
        if profile.get(key):
            lines += ["", f"## {key}", ""] + [f"- {str(item).replace(chr(10), ' ')}" for item in profile[key]]
    return "\n".join(lines)


def _langmem_parse_md(memory_md: str) -> dict:
    profile: dict = {}
    section = None
    for line in memory_md.split("\n"):
        if line.startswith("## "):
            section = line[3:].strip()
            profile[section] = []
        elif line.startswith("- ") and section in USER_PROFILE_LISTS:
            profile[section].append(line[2:])
        elif line.startswith("- ") and section is None and ": " in line:
            key, value = line[2:].split(": ", 1)
            if key in USER_PROFILE_FIELDS:
                profile[key] = value
    return profile


def _first_json_object(text: str) -> dict | None:
    start = text.find("{")
    while start != -1:
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
        start = text.find("{", start + 1)
    return None


async def _langmem(conversations: list[dict], memory_md: str, call) -> str:
    messages = [m for c in conversations for m in _messages(c)]
    existing = _langmem_parse_md(memory_md) if memory_md.strip() else {}

    system = "You are a memory subroutine for an AI."
    if existing:
        # trustcall _ExtractUpdates._setup appends the existing instances to the system message (a list of
        # (id, kind, value) prints each value with Pydantic's str()). ADAPTED: trustcall asks for JSONPatches through
        # its PatchDoc tool ("Generate JSONPatches to update the existing schema instances."); here the whole updated
        # instance is asked for instead.
        instance = " ".join(f"{key}={existing.get(key, [] if key in USER_PROFILE_LISTS else '')!r}" for key in USER_PROFILE_FIELDS)
        system += (
            "\n\nUpdate the existing schema instances.\n<existing>\n"
            f'<instance id={uuid.uuid4()} schema_type="UserProfile">\n{instance}\n</instance>\n</existing>\n'
        )
    # ADAPTED: no tool calling. trustcall binds UserProfile as a function; here its definition is shown and its
    # arguments are asked for as the reply.
    system += (
        "\n\nYou have one function:\n" + json.dumps(USER_PROFILE_TOOL, indent=2) + "\n"
        "Do not call it: reply with only its arguments, one JSON object matching its parameters, the complete profile."
    )

    session_id = uuid.uuid4()
    session = f"\n\n<session_{session_id}>\n{_langmem_conversation(messages)}\n</session_{session_id}>"
    user = (
        f"{_MEMORY_INSTRUCTIONS}\n\nEnrich, prune, and organize memories based on any new information. "
        f"If an existing memory is incorrect or outdated, update it based on the new information. "
        f"All operations must be done in single parallel multi-tool call."
        f" Avoid duplicate extractions. {session}"
    )
    output = await call([{"role": "system", "content": system}, {"role": "user", "content": user}])

    profile = dict(existing)
    for key, value in (_first_json_object(output) or {}).items():
        if key in USER_PROFILE_LISTS and isinstance(value, list):
            profile[key] = [str(item) for item in value if str(item).strip()]
        elif key in USER_PROFILE_FIELDS and key not in USER_PROFILE_LISTS and value not in (None, ""):
            profile[key] = str(value)
    return _langmem_render(profile)


# --- memobase ----------------------------------------------------------------------------------------------------

MEMOBASE_BUFFER_TOKENS = 1024  # env.py max_chat_blob_buffer_token_size: a flush once the buffer passes it
MEMOBASE_PROCESS_TOKENS = 16384  # max_chat_blob_buffer_process_token_size
MEMOBASE_MAX_SUBTOPICS = 15  # max_profile_subtopics
MEMOBASE_MAX_SLOT_TOKENS = 128  # max_pre_profile_token_size
SEPARATOR = "::"  # llm_tab_separator
EVENT_THEME_REQUIREMENT = "Focus on the user's infos, not its instructions."  # env.py event_theme_requirement

# user_profile_topics.CANDIDATE_PROFILE_TOPICS, subtopic names already through attribute_unify
CANDIDATE_PROFILE_TOPICS = [
    ("basic_info", [("name", None), ("age", "integer"), ("gender", None), ("birth_date", None), ("nationality", None), ("ethnicity", None), ("language_spoken", None)]),
    ("contact_info", [("email", None), ("phone", None), ("city", None), ("country", None)]),
    ("education", [("school", None), ("degree", None), ("major", None)]),
    ("demographics", [("marital_status", None), ("number_of_children", None), ("household_income", None)]),
    ("work", [("company", None), ("title", None), ("working_industry", None), ("previous_projects", None), ("work_skills", None)]),
    ("interest", [("books", None), ("movies", None), ("music", None), ("foods", None), ("sports", None)]),
    ("psychological", [("personality", None), ("values", None), ("beliefs", None), ("motivations", None), ("goals", None)]),
    ("life_event", [("marriage", None), ("relocation", None), ("retirement", None)]),
]

SUMMARY_PROMPT = """You are a expert of logging personal info, schedule, events from chats.
You will be given a chats between a user and an assistant.

## Requirement
- You need to list all possible user info, schedule and events
- {additional_requirements}
- If the user event/schedule has specific mention time or event happen time. Convert the event date info in the message based on [TIME] after your log. for example
    Input: `[2024/04/30] user: I bought a new car yesterday!`
    Output: `user bought a new car. [mention 2024/04/30, buy car in 2024/04/29]`
    Input: `[2024/04/30] user: I bought a car 4 years ago!`
    Output: `user bought a car. [mention 2024/04/30, buy car in 2020]`
    Explain: because you don't know the exact date, only year, so 2024-4=2020. or you can log at [4 years before 2024/04/30]
    Input: `[2024/04/30] user: I bought a new car last week!`
    Output: `user bought a new car. [mention 2024/04/30, buy car in 2024/04/30 a week before]`
    Explain: because you don't know the exact date.
    Input: `[...] user: I bought a new car last week!`
    Output: `user bought a new car.`
    Explain: because you don't know the exact date, so don't attach any date.

### Important Info
Below is the topics/subtopics you should log from the chats.
<topics>
{topics}
</topics>
Below is the important attributes you should log from the chats.
<attributes>
{attributes}
</attributes>


## Input Format
### Already Logged
You will receive a list of previous logging result, you should also log the relevant infos that maybe related to those already logged.
Pervious result in organized in Profile-format:
- TOPIC{separator}SUBTOPIC{separator}CONTENT... // maybe truncated

### Input Chats
You will receive a conversation between the user and the assistant. The format of the conversation is:
- [TIME] NAME: MESSAGE
where NAME is ALIAS(ROLE) or just ROLE, when ALIAS is available, use ALIAS to refer user/assistant.
MESSAGE is the content of the conversation.
TIME is the time of this message happened, so you need to convert the date info in the message based on TIME if necessary.

## Output Format
- LOGGING[TIME INFO] // TYPE
Output your logging result in Markdown unorder list format.
For example:
```
- Jack paint a picture about his kids.[mention 2023/1/23] // event
- User's alias is Jack, assistant is Melinda. // info
- Jack mentioned his work is software engineer in Memobase. [mention 2023/1/23] // info
- Jack plans to go the gym. [mention 2023/1/23, plan in 2023/1/24] // schedule
...
```
Always add specific mention time of your log, and the event happen time if possible.
Remember, make sure your logging is pure and concise, any time info should move to [TIME INFO] block.

## Content Requirement
- You need to list all possible user info, schedule and events
- {additional_requirements}

Now perform your task.
"""

EXTRACT_EXAMPLES = [
    ("""- User say Hi to assistant.
""", []),
    ("""
- User's favorite movies are Inception and Interstellar [mention 2025/01/01]
- User's favorite movie is Tenet [mention 2025/01/02]
""", [
        ("interest", "movie", "Inception, Interstellar[mention 2025/01/01]; favorite movie is Tenet [mention 2025/01/02]"),
        ("interest", "movie_director", "user seems to be a big fan of director Christopher Nolan"),
    ]),
]

DEFAULT_JOB = """You are a professional psychologist.
Your responsibility is to carefully read out the memo of user and extract the important profiles of user in structured format.
Then extract relevant and important facts, preferences about the user that will help evaluate the user's state.
You will not only extract the information that's explicitly stated, but also infer what's implied from the conversation.
"""

FACT_RETRIEVAL_PROMPT = """{system_prompt}
## Formatting
### Input
#### Topics Guidelines
You'll be given some user-relatedtopics and subtopics that you should focus on collecting and extracting.
Don't collect topics that are not related to the user, it will cause confusion.
For example, if the memo mentions the position of another person, don't generate a "work{tab}position" topic, it will cause confusion. Only generate a topic if the user mentions their own work.
You can create your own topics/sub_topics if you find it necessary, unless the user requests to not to create new topics/sub_topics.
#### User Before Topics
You will be given the topics and subtopics that the user has already shared with the assistant.
Consider use the same topic/subtopic if it's mentioned in the conversation again.
#### Memos
You will receive a memo of user in Markdown format, which states user infos, events, preferences, etc.
The memo is summarized from the chats between user and a assistant.

### Output
#### Think
You need to think about what's topics/subtopics are mentioned in the memo, or what implications can be inferred from the memo.
#### Profile
After your steps of thinking, you need to extract the facts and preferences from the memo and place them in order list:
- TOPIC{tab}SUB_TOPIC{tab}MEMO
For example:
- basic_info{tab}name{tab}melinda
- work{tab}title{tab}software engineer
For each line is a fact or preference, containing:
1. TOPIC: topic represents of this preference
2. SUB_TOPIC: the detailed topic of this preference
3. MEMO: the extracted infos, facts or preferences of `user`
those elements should be separated by `{tab}` and each line should be separated by `\n` and started with "- ".

Final output template:
```
[POSSIBLE TOPICS THINKING...]
---
- TOPIC{tab}SUB_TOPIC{tab}MEMO
- ...
```

## Extraction Examples
Here are some few shot examples:
{examples}
Return the facts and preferences in a markdown list format as shown above.
Only extract the attributes with actual values, if the user does not provide any value, do not extract it.
You need to first think, then extract the facts and preferences from the memo.


#### Topics Guidelines
Below is the list of topics and subtopics that you should focus on collecting and extracting:
{topic_examples}


Remember the following:
- If the user mentions time-sensitive information, try to infer the specific date from the data.
- Use specific dates when possible, never use relative dates like "today" or "yesterday" etc.
- If you do not find anything relevant in the below conversation, you can return an empty list.
- Make sure to return the response in the format mentioned in the formatting & examples section.
- You should infer what's implied from the conversation, not just what's explicitly stated.
- Place all content related to this topic/sub_topic in one element, no repeat.
- The memo will have two types of time, one is the time when the memo is mentioned, the other is the time when the event happened. Both are important, don't mix them up.

Now perform your task.
Following is a conversation between the user and the assistant. You have to extract/infer the relevant facts and preferences from the conversation and return them in the list format as shown above.
"""

MERGE_FACTS_PROMPT = """You are responsible for maintaining user memos.
Your job is to determine how new supplementary information should be merged with current memos.
Users will provide a series of memo topics/subtopics, along with topic descriptions and specific update requirements (which may be empty).
For each piece of supplementary information, you should determine whether the new information should be directly added, update the current corresponding memo, or be discarded.

## Input Format
```
{{
    "memo_id": "1",
    "new_info": "",
    "current_memo": "",
    "topic": "",
    "subtopic": "",
    "topic_description": "",
    "update_instruction": "",
}}
{{
    "memo_id": "2",
    ...
}}
...
```
When evaluating each memo, you need to consider whether it aligns with the topic and update description.

Here are your output actions:
1. Direct Add: If the supplementary information brings new insights, you should add it directly. If the current memo is empty, you should add the supplementary information directly.
2. Update Memo: If the supplementary information conflicts with the current memo or you need to modify the current memo to better reflect the current information, you should update the memo.
3. Discard Merge: If the supplementary information has no value, is completely contained within the current memo, or doesn't meet the current memo's content requirements, you should discard the merge.

## Reasoning
Before outputting your action, you need to first consider the following:
1. Whether the supplementary information aligns with the memo's topic description
    1.1. If it doesn't align, determine if you can modify the supplementary information to meet the memo requirements, then process your modified supplementary information
    1.2. If you cannot modify the supplementary information to satisfy the topic description, you should discard the merge
3. For supplementary information that meets the current memo requirements, you need to refer to the above description to determine your output action
4. If you choose to update the memo, also consider whether other parts of the current memo can be streamlined or removed.

Additional considerations:
1. The current memo may be empty. In this case, after reasoning step 1, if you can obtain supplementary information that meets the requirements, add it directly
2. If the update requirement is not empty, you need to refer to the user's update requirements in your reasoning

## Output Actions
Assuming you are processing the Nth piece of supplementary information (memo_id=N), you should make the following judgment:
### Direct Add
```
N. APPEND{tab}APPEND
```
If choosing to add directly, simply output the word `APPEND`, no need to restate the content
### Update Memo
```
N. UPDATE{tab}[UPDATED_MEMO]
```
In `[UPDATED_MEMO]`, you need to rewrite the complete updated current memo
### Discard Merge
```
N. ABORT{tab}ABORT
```
If choosing to discard the merge, simply output the word `ABORT`, no need to restate the content

## Output Template
Based on the above instructions, your output should follow this template:

THOUGHT
---
1. ACTION{tab}...
2. ACTION{tab}...
...

Where:
- `THOUGHT` is your reasoning process
- `N. ACTION{tab}...` is your operation for the Nth piece of supplementary information (memo_id=N)

## Examples
### Input Example
{{
    "memo_id": "1",
    "new_info": "Preparing for final exams [mentioned on 2025/06/01]",
    "current_memo": "Preparing for midterm exams [mentioned on 2025/04/01]",
    "topic": "Study",
    "subtopic": "Exam goals",
    "update_instruction": "Each time you update goals, consider whether there are outdated or conflicting goals and remove them",
}}
{{
    "memo_id": "2",
    "new_info": "Using Duolingo to self-study Japanese",
    "current_memo": "",
    "topic": "Study",
    "subtopic": "Software usage",
}}
{{
    "memo_id": "3",
    "new_info": "User likes eating hot pot",
    "current_memo": "",
    "topic": "Interests",
    "subtopic": "Sports",
}}

### Output Example
```
The supplementary information mentions that the user's current study goal is to prepare for final exams, which aligns with the topic description of recording the user's study goals. However, there's a conflict between final exams and midterm exams, so we need to remove the midterm exam goal and update it to final exams.
Additionally, the user mentioned they are using Duolingo for language learning, which meets the software usage requirement. Since memo ID 2 has an empty current memo, we can add it directly.
Liking hot pot doesn't belong to sports interests, and we cannot derive potential interests from this information, so we discard the merge.
---
1. UPDATE{tab}Preparing for final exams [mentioned on 2025/06/01];
2. APPEND{tab}APPEND
3. ABORT{tab}ABORT
```

## Requirements
You must follow these requirements:
- Strictly adhere to the correct output format.
- Ensure updated memos do not exceed 5 sentences. Always maintain conciseness and output memo key points.
- Never fabricate content not mentioned in the input.
- Preserve time annotations from old and new memos (e.g., XXX[mentioned on 2025/05/05, occurred in 2022]).
- If deciding to update, ensure the final memo is concise without redundant information (e.g., "User is sad; User's mood is sad" == "User is sad").

That's all the content. Now execute your work.
"""

ORGANIZE_EXAMPLES = [
    (
        """topic: 特殊事件 
- 上岛{tab}用户和尤斐上了一个小岛
- 下雨{tab}2025 年 1 月 6 日下雨
- 前往洞穴{tab}用户和尤斐一起进入洞穴探索
- 乘坐快艇逃离{tab}用户和尤斐乘坐快艇逃离，途中可能遇到过一些情况，比如可能曾遇到大船追捕，还可能遇到狗群。
- 休息{tab}用户和尤斐曾在多处地方休息，包括溪边等，还曾在树林和废弃木屋过夜，2025/01/06 晚上在一个地方休息，第二天准备继续赶路。
- 到达新地方{tab}来到一处易守难攻的地方
- 前往别墅{tab}用户和尤斐从洞穴出来后返回别墅
- 前往别墅地下{tab}用户和尤斐前往用户位于地下的别墅
- 发现小岛{tab}用户和尤斐发现一个小岛并决定降落，小岛上似乎没人
- 前往山洞{tab}用户和尤斐一起前往山洞探索，发现破旧笔记本和遇到野兽，尤斐制服野兽后一起离开，之前还曾在山洞休息
- 发现出口{tab}用户和尤斐在 2025 年 1 月 4 日下午 3 点 20 分发现了一个类似世外桃源地方的出口。
""",
        """- 上岛冒险{tab}用户和尤斐发现一个似乎没人的小岛并降落; 他们一起进入洞穴探索， 发现破旧笔记本和遇到野兽，尤斐制服野兽后一起离开，之前还曾在山洞休息，离开并返回地下的别墅
- 休息{tab}用户和尤斐曾在多处地方休息，包括溪边等，还曾在树林和废弃木屋过夜，2025/01/06 晚上在一个地方休息，第二天准备继续赶路。
- 逃离{tab}用户和尤斐在 2025 年 1 月 4 日下午 3 点 20 分发现了一个类似世外桃源地方的出口，用户和尤斐乘坐快艇逃离，途中可能曾遇到大船追捕，还可能遇到狗群。
""",
    )
]

ORGANIZE_PROMPT = """You will organize memos for user.
The memos are in the same given topic.
You will be given the current messy/too many memos with corresponding sub_topics.
You need to re-organize the memos into no more than {max_subtopics} sub_topics:
- You can discard some memos if they're not relevant to the topic.
- You can merge some memos into one sub_topic if they're related to the same topic.
- You can create new sub_topics if you find it necessary.
- The final result should have no more than {max_subtopics} sub_topics.

## Topics you should be aware of
Below are some sub_topics you can refer to:
{user_profile_topics}
Try to merge the memos into the above sub_topics first, you can create new sub_topics if you find it necessary.

## Formatting
### Input
You will receive a list of memos with sub_topics. The format of the memos is:
topic: TOPIC
- SUBTOPIC{tab}MEMO
- ...
The main topic is TOPIC, and the following lines are the sub_topics: SUBTOPIC is the sub_topic of the memo, and MEMO is the content of the memo.

### Output
You need to re-organize the memos into no more than {max_subtopics} sub_topics:
- NEW_SUB_TOPIC{tab}MEMO
For example:
- name{tab}melinda
- title{tab}software engineer

For each line is a new memo of user, containing:
1. NEW_SUB_TOPIC: the new sub_topic of the memo
2. MEMO: the content of the memo
those elements should be separated by `{tab}` and each line should be separated by `\n` and started with "- ".


## Examples
Here are some few shot examples:
{examples}

Return the new sub_topics and memos in a markdown list format as shown above.
Remember the following:
- The final result should have no more than {max_subtopics} sub_topics.
- You can discard some memos if they're not relevant to the topic.
- Prioritize the most important subtopics at the front.
"""

SUMMARY_PROFILE_PROMPT = """You are given a user profile with some information about the user.
Extract high-level preference from the profile

## Requirement
- Extract high-level preference from the profile
- The preference should be the most important and representative preference of the user.
  For example, the original perference is "user likes Chocolate[mentioned in 2023/1/23], Ice cream, Cake, Cookies, Brownies[mentioned in 2023/1/24]...", then your extraction should be "user maybe likes sweet food(cake/cookies...)".
- The preference should be concise and clear.
"""

EXCLUDE_PROFILE_VALUES = [
    "无", "未提及", "不清楚", "用户未提及", "对话未提及", "未知", "不详", "没有提到", "没有说明", "无法确定", "无相关内容",
    "未明确提及", "无明确信息", "无符合信息",
    "none", "unknown", "not mentioned", "not mentioned by user", "not mentioned in the conversation", "unclear",
    "unspecified", "not specified", "not determined", "no information", "n/a", "no related content",
    "no related information", "no matched information",
]


def _unify(attr: str) -> str:
    return attr.lower().strip().replace(" ", "_")


def _truncate(content: str, max_tokens: int) -> str:
    """utils.truncate_string. ADAPTED: 4 characters a token, not tiktoken's gpt-4o encoding."""
    return content if _tokens(content) <= max_tokens else content[: max_tokens * 4] + "..."


def _meaningless(memo: str) -> bool:
    return bool(difflib.get_close_matches(memo.strip().lower(), EXCLUDE_PROFILE_VALUES))


def _topics_prompt() -> str:
    """user_profile_topics.get_prompt with profile_init_utils.formate_profile_topic."""
    blocks = [
        f"- {topic} ()\n" + "\n".join(f"  - {name}" + (f"({description})" if description else "") for name, description in subs)
        for topic, subs in CANDIDATE_PROFILE_TOPICS
    ]
    return "\n".join(blocks) + "\n..."


def _extract_system_prompt() -> str:
    examples = "\n\n".join(
        f"""<example>
<input>{memo}</input>
<output>
{chr(10).join(f"- {_unify(t)}{SEPARATOR}{_unify(s)}{SEPARATOR}{m.strip()}" for t, s, m in facts) or "NONE"}
</output>
</example>
"""
        for memo, facts in EXTRACT_EXAMPLES
    )
    return FACT_RETRIEVAL_PROMPT.format(system_prompt=DEFAULT_JOB, examples=examples, tab=SEPARATOR, topic_examples=_topics_prompt())


def _blob_str(messages: list[dict]) -> str:
    """utils.get_blob_str of a chat blob: "[created_at] role: content". ADAPTED: created_at, a free string the client
    sets on each message, is the message's date and time "2023/05/20 09:15"."""
    return "\n".join(f"[{m['at']:%Y/%m/%d %H:%M}] {m['role']}: {m['text']}" for m in messages)


def _memobase_parse_md(memory_md: str) -> list[dict]:
    profiles: list[dict] = []
    for line in memory_md.split("\n"):
        match = re.match(r"^- ([^\s:]+)::(\S+?): (.*)$", line)
        if match:
            profiles.append({"id": str(uuid.uuid4()), "topic": match.group(1), "sub_topic": match.group(2), "content": match.group(3)})
        elif profiles and line.strip() and not line.startswith("#"):
            profiles[-1]["content"] += "\n" + line
    return profiles


def _memobase_render(profiles: list[dict]) -> str:
    """controllers/context.py: "- topic::sub_topic: content", under the heading Memobase's context prompt gives it."""
    if not profiles:
        return "## User Current Profile:\n"
    return "## User Current Profile:\n- " + "\n- ".join(f"{p['topic']}::{p['sub_topic']}: {p['content']}" for p in profiles)


async def _memobase_flush(blobs: list[list[dict]], profiles: list[dict], call) -> None:
    """controllers/modal/chat/__init__.process_blobs, profile half (process_event_res makes no call without event
    tags), applied to `profiles` in place."""
    # truncate_chat_blobs: the latest blobs that fit in max_chat_blob_buffer_process_token_size
    kept, total = [], 0
    for blob in reversed(blobs):
        total += _tokens(_blob_str(blob))
        if total > MEMOBASE_PROCESS_TOKENS:
            break
        kept.append(blob)
    blobs = kept[::-1]
    if not blobs:
        return

    # utils.pack_current_user_profiles
    already = sorted({(_unify(p["topic"]), _unify(p["sub_topic"])) for p in profiles})
    values = {(_unify(p["topic"]), _unify(p["sub_topic"])): p["content"] for p in profiles}
    already_prompt = "\n".join(f"- {t}{SEPARATOR}{s}{SEPARATOR}{_truncate(values[(t, s)], 5)}" for t, s in already)

    # entry_summary.entry_chat_summary
    memo = await call([
        {"role": "system", "content": SUMMARY_PROMPT.format(
            topics=_topics_prompt(), attributes="", additional_requirements=EVENT_THEME_REQUIREMENT, separator=SEPARATOR
        )},
        {"role": "user", "content": f"### Already Logged\n{already_prompt}\n### Input Chats\n" + "\n".join(_blob_str(b) for b in blobs) + "\n"},
    ])
    memo = memo.strip()
    if not memo:
        return

    # extract.extract_topics
    output = await call([
        {"role": "system", "content": _extract_system_prompt()},
        {"role": "user", "content": f"\n#### User Before topics\n{already_prompt}\nDon't output the topics and subtopics that are not mentioned in the following conversation.\n#### Memo\n{memo}\n"},
    ])
    facts: dict[tuple[str, str], str] = {}
    for line in (l.strip() for l in output.split("\n")):
        parts = line[2:].split(SEPARATOR) if line.startswith("- ") else []
        if len(parts) != 3 or _meaningless(parts[2]):
            continue
        key = (_unify(parts[0]), _unify(parts[1]))
        facts[key] = f"{facts[key]}; {parts[2].strip()}" if key in facts else parts[2].strip()

    # merge_yolo.merge_or_valid_new_memos (profile_validate_mode is on by default, so every fact goes to the model;
    # Memobase makes this call even with no facts)
    defined = {(topic, name): description for topic, subs in CANDIDATE_PROFILE_TOPICS for name, description in subs}
    runtime = {(p["topic"], p["sub_topic"]): p for p in profiles}
    new_memos = [
        {
            "memo_id": i + 1,
            "new_info": content,
            "current_memo": runtime[key]["content"] if key in runtime else "",
            "topic": key[0],
            "subtopic": key[1],
            "topic_description": defined.get(key),
            "update_instruction": None,
        }
        for i, (key, content) in enumerate(facts.items())
    ]
    output = await call([
        {"role": "system", "content": MERGE_FACTS_PROMPT.format(tab=SEPARATOR)},
        {"role": "user", "content": f"\n{new_memos}\n"},
    ])
    actions = {}
    for line in (l.strip() for l in output.split("\n") if l.strip()):
        match = re.match(r"^(\d+)\.(.*)", line)
        parts = match.group(2).strip().split(SEPARATOR) if match else []
        if len(parts) >= 2 and parts[0].upper().strip() in {"APPEND", "UPDATE", "ABORT"}:
            actions[int(match.group(1))] = (parts[0].upper().strip(), SEPARATOR.join(parts[1:]).strip())
    adds, updates = [], []
    for memo_input in new_memos:
        action = actions.get(memo_input["memo_id"])
        key = (memo_input["topic"], memo_input["subtopic"])
        if action is None or action[0] == "ABORT":
            continue
        current = runtime.get(key)
        if action[0] == "UPDATE":
            content = action[1]
        else:
            content = f"{current['content']};{memo_input['new_info']}" if current else memo_input["new_info"]
        if current is None:
            adds.append({"topic": key[0], "sub_topic": key[1], "content": content})
        else:
            updates.append({"id": current["id"], "content": content})

    # organize.organize_profiles: each topic with more than max_profile_subtopics, counted before this flush, is
    # rewritten into at most 8 subtopics; if any topic comes back empty the whole reorganization is dropped
    deletes, reorganized_all, failed = [], [], False
    groups: dict[str, list[dict]] = {}
    for p in profiles:
        groups.setdefault(p["topic"], []).append(p)
    for topic, group in groups.items():
        if len(group) <= MEMOBASE_MAX_SUBTOPICS:
            continue
        topic = _unify(topic)
        suggest = [f"  - {name}" + (f"({description})" if description else "") for t, subs in CANDIDATE_PROFILE_TOPICS if t == topic for name, description in subs] or "None"
        examples = "\n\n".join(f"Input:\n{i}Output:\n{o}" for i, o in ORGANIZE_EXAMPLES).format(tab=SEPARATOR)
        output = await call([
            {"role": "system", "content": ORGANIZE_PROMPT.format(
                max_subtopics=MEMOBASE_MAX_SUBTOPICS // 2 + 1, examples=examples, tab=SEPARATOR, user_profile_topics=suggest
            )},
            {"role": "user", "content": f"topic: {topic}\n" + "\n".join(f"- {p['sub_topic']}{SEPARATOR}{p['content']}" for p in group) + "\n"},
        ])
        reorganized = []
        for line in (l.strip() for l in output.split("\n")):
            parts = line[2:].split(SEPARATOR) if line.startswith("- ") else []
            if len(parts) == 2 and not _meaningless(parts[1].strip()):
                reorganized.append({"topic": topic, "sub_topic": _unify(parts[0].strip()), "content": parts[1].strip()})
        failed = failed or not reorganized
        reorganized_all += reorganized[: MEMOBASE_MAX_SUBTOPICS // 2 + 1]
        deletes += [p["id"] for p in group]
    if failed:
        deletes = []
    elif deletes:
        deduplicated: dict[tuple[str, str], dict] = {}
        for add in adds + reorganized_all:
            key = (add["topic"], add["sub_topic"])
            if key in deduplicated:
                deduplicated[key]["content"] += f"; {add['content']}"
            else:
                deduplicated[key] = add
        adds = list(deduplicated.values())

    # summary.re_summary: a slot past max_pre_profile_token_size is summarized, then cut to half that
    for pack in adds + updates:
        if _tokens(pack["content"]) > MEMOBASE_MAX_SLOT_TOKENS:
            summary = await call([{"role": "system", "content": SUMMARY_PROFILE_PROMPT}, {"role": "user", "content": pack["content"]}])
            pack["content"] = _truncate(summary, MEMOBASE_MAX_SLOT_TOKENS // 2)

    # controllers/profile.add_update_delete_user_profiles
    by_id = {p["id"]: p for p in profiles}
    for update in updates:
        if update["id"] in by_id:
            by_id[update["id"]]["content"] = update["content"]
    profiles[:] = [p for p in profiles if p["id"] not in deletes]
    profiles += [{"id": str(uuid.uuid4()), **add} for add in adds]


async def _memobase(conversations: list[dict], memory_md: str, call) -> str:
    profiles = _memobase_parse_md(memory_md)
    buffer: list[list[dict]] = []
    for conversation in conversations:
        blob = _messages(conversation)  # ADAPTED: one ChatBlob per conversation file
        if not blob:
            continue
        buffer.append(blob)
        if sum(_tokens(_blob_str(b)) for b in buffer) > MEMOBASE_BUFFER_TOKENS:
            await _memobase_flush(buffer, profiles, call)
            buffer = []
    if buffer:
        await _memobase_flush(buffer, profiles, call)  # the client's explicit flush at the end
    return _memobase_render(profiles)
