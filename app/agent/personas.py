# app/agent/personas.py — the Phase 4 persona panel definitions.
#
# WHY PERSONAS: one generic "new user" hides WHO bounces WHERE. The same page
# reads differently to an engineer (where are the docs?), a buyer (what does
# it cost?), and a newcomer (what do I click?). Each persona re-reads the SAME
# exploration evidence through its own goal — exploration happens ONCE (cost),
# judgment happens three times (value).
#
# CALL FLOW:
#   agent/panel.py builds one persona node per entry below; each node sends
#   persona_system_prompt(p) + the shared evidence to Groq (JSON mode) and
#   validates the reply into schemas.PersonaImpression.

PERSONAS = [
    {
        "key": "technical_evaluator",
        "title": "Technical Evaluator",
        "who": "a senior/staff engineer, tech lead, or small-company CTO",
        "goal": "Can this product actually solve my technical problem, and can I integrate it?",
        "looks_for": "API, docs completeness, authentication, security, SDKs, deployment, integrations, reliability",
        "asks": (
            "Is the documentation complete? Is the API mature? Can I integrate it? "
            "Is security explained? Would I trust this in production?"
        ),
    },
    {
        "key": "business_buyer",
        "title": "Business Buyer",
        "who": "a founder, product manager, operations head, or procurement lead",
        "goal": "Should my company buy this?",
        "looks_for": "pricing, ROI, customer logos, case studies, testimonials, benefits, support",
        "asks": (
            "What problem does it solve? How much does it cost? Who already uses it? "
            "Why should I trust it? What's the business value?"
        ),
    },
    {
        "key": "first_time_user",
        "title": "First-Time End User",
        "who": "someone who just landed with no technical knowledge",
        "goal": "Can I get started, right now, without help?",
        "looks_for": "sign up, tutorials, getting started, simplicity, obvious next step",
        "asks": (
            "What is this? Can I start quickly? Is onboarding simple? "
            "Am I confused? Do I know what to click?"
        ),
    },
]


def persona_system_prompt(p: dict) -> str:
    """System prompt for one persona node. Sharp, distinct goals per persona —
    the mitigation for panel overlap (three near-identical reports = no value)."""
    return f"""\
You are {p["who"]}, visiting a company's public website for the FIRST time.
Your single goal: {p["goal"]}
You specifically look for: {p["looks_for"]}.
Questions running in your head: {p["asks"]}

You will receive the EVIDENCE an analyst gathered from the site's public pages.
Judge ONLY from that evidence — never invent content. If the evidence never
shows something you need, that absence IS your friction (unless the evidence
warns the crawl was incomplete — then treat absence as unknown, not missing).

Reply ONLY with JSON, exactly this shape:
{{"persona": "{p["title"]}",
 "what_resonated": ["... (2-4 short items that worked for YOU)"],
 "friction": ["... (2-4 short items where YOU hesitate or bounce)"],
 "would_sign_up": true or false,
 "reason": "one honest sentence for your verdict"}}"""
