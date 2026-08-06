# -*- coding: utf-8 -*-
"""
Questionnaire definition.

Question wording and option lists are verbatim from the original Google
Form response CSV, so model answers are directly comparable with the human
responses without remapping.
"""

import json

# ------------------------- verbatim option sets ---------------------------

LIKERT_5 = ["Strongly agree", "Agree", "OK", "Disagree", "Strongly disagree"]

YES_NO = ["Yes", "No"]          # Q1, Q7 in the original form
TRUE_FALSE = ["True", "False"]  # Q4 in the original form

CLUSTER_COUNTS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                  "I couldn't tell"]

ALGORITHM_STEPS = [
    "Computing the center point",
    "Computing the edge points",
    "Dividing the space into slices",
    "Dividing the space into squares",     # distractor
    "Reducing empty space",
    "Re-arranging empty space",            # distractor
    "Finding gradients",
    "Finding outliers",                    # distractor
    "Finding empty space",
    "Introducing new empty spaces",        # distractor
    "Computing new cluster centers",
    "Recursing into sub-regions",
    "Early termination",
]

DISTRACTORS = {
    "Dividing the space into squares",
    "Re-arranging empty space",
    "Finding outliers",
    "Introducing new empty spaces",
}

Q6_COLUMNS = ["Yes (included in algorithm)", "No (not present in algorithm)"]

# --------------------------- question wording -----------------------------
PER_VIDEO_QUESTIONS = {
    "Q1": {
        "text": "The video content was easy to understand.",
        "type": "yes_no",
    },
    "Q2": {
        "text": "The step-by-step animation helped me understand the clustering process.",
        "type": "likert",
    },
    "Q3": {
        "text": "How many clusters do you see at the end?",
        "type": "cluster_count",
    },
    "Q4": {
        "text": "True or False: The user must specify the number of clusters before running the algorithm.",
        "type": "true_false",
        "options": ["True", "False"],
        "correct": "False",  # confirmed against the original answer key
    },
    "Q5": {
        "text": "How much of the video content is understandable to you?",
        "type": "int_0_100",
    },
    "Q6": {
        "text": "Which of the following steps did you observe in the animation and are part of the algorithm? (choose all that apply)",
        "type": "grid",
    },
    "Q7": {
        "text": "Do you think you could explain how this algorithm works to someone else?",
        "type": "yes_no",
    },
    "Q8": {
        "text": "What, if anything, was still unclear after watching the animation?",
        "type": "free_text",
    },
    "Q9": {
        "text": "How would you improve the animation to make the process even clearer?",
        "type": "free_text",
    },
}

OVERALL_QUESTIONS = {
    "Q17": {"text": "I can see how the clustering algorithm works.",
            "type": "scale_1_5"},
    "Q18": {"text": "I could explain this algorithm to someone else.",
            "type": "scale_1_5"},
    "Q19": {"text": "The animation was easy to follow without feeling overwhelming.",
            "type": "scale_1_5"},
    "Q20": {"text": "How confident are you that you understand the clustering algorithm?",
            "type": "scale_1_5"},
    "Q21": {"text": "What, if anything, was still unclear after watching the videos?",
            "type": "free_text"},
    "Q22": {"text": "How would you improve the animation to make the process clearer?", "type": "free_text"},
    "Q23": {"text": "Do you have any recommendations on how to make the algorithm more explainable and visible?",
            "type": "free_text"},
}

# ----------------------------- prompt builders ----------------------------

SYSTEM_PROMPT = (
    "You are taking part in a study about a clustering-algorithm "
    "visualization. You will watch four short animation videos, one at a "
    "time, and answer a fixed questionnaire after each video. Answer based "
    "ONLY on what you can see in the videos shown in this conversation. "
    "Always reply with a single valid JSON object and nothing else — no "
    "markdown fences, no commentary. Every answer must use the EXACT option "
    "strings given in the question."
)


def _fmt_options(opts):
    return " | ".join(f'"{o}"' for o in opts)


def per_video_prompt(video_name):
    """User-turn text that accompanies one video."""
    q = PER_VIDEO_QUESTIONS
    steps_json_keys = ",\n".join(
        f'    "{s}": "<one of: {_fmt_options(Q6_COLUMNS)}>"'
        for s in ALGORITHM_STEPS
    )
    return f"""You have just watched "{video_name}" (the video attached to this message).
Please answer the following questions about THIS video.

Q1. {q['Q1']['text']}
    Options: {_fmt_options(YES_NO)}
Q2. {q['Q2']['text']}
    Options: {_fmt_options(LIKERT_5)}
Q3. {q['Q3']['text']}
    Options: {_fmt_options(CLUSTER_COUNTS)}
Q4. {q['Q4']['text']}
    Options: {_fmt_options(TRUE_FALSE)}
Q5. {q['Q5']['text']}
    Please provide a percentage from 0 to 100 (answer with an integer).
Q6. {q['Q6']['text']}
    For EACH step below, answer with one of: {_fmt_options(Q6_COLUMNS)}.
Q7. {q['Q7']['text']}
    Options: {_fmt_options(YES_NO)}
Q8. {q['Q8']['text']}
    Free text, 1-3 sentences. You must provide a non-empty answer. If
    nothing was unclear, say so explicitly and briefly explain why.
Q9. {q['Q9']['text']}
    Free text, 1-3 sentences. You must provide a non-empty answer. Give at
    least one concrete suggestion, or explain why no change is needed.

Reply with exactly this JSON structure:
{{
  "Q1": "<option>",
  "Q2": "<option>",
  "Q3": "<option>",
  "Q4": "<option>",
  "Q5": <integer 0-100>,
  "Q6": {{
{steps_json_keys}
  }},
  "Q7": "<option>",
  "Q8": "<free text>",
  "Q9": "<free text>"
}}"""


def overall_prompt():
    """Final text-only turn after all four videos."""
    q = OVERALL_QUESTIONS
    return f"""You have now watched all four videos in this conversation.
Please answer the following overall questions, considering all four videos together.

Q17. {q['Q17']['text']}  (answer an integer 1-5, where 1 = Strongly disagree and 5 = Strongly agree)
Q18. {q['Q18']['text']}  (answer an integer 1-5, where 1 = Strongly disagree and 5 = Strongly agree)
Q19. {q['Q19']['text']}  (answer an integer 1-5, where 1 = Strongly disagree and 5 = Strongly agree)
Q20. {q['Q20']['text']}  (answer an integer 1-5, where 1 = Not confident at all and 5 = Very confident)
Q21. {q['Q21']['text']}  (required non-empty free text, 1-3 sentences; if
      nothing was unclear, say so explicitly and briefly explain why)
Q22. {q['Q22']['text']}  (required non-empty free text, 1-3 sentences; give
      at least one concrete suggestion or explain why none is needed)
Q23. {q['Q23']['text']}  (required non-empty free text, 1-3 sentences; give
      a concrete recommendation or explain why none is needed)

Reply with exactly this JSON structure:
{{
  "Q17": <1-5>, "Q18": <1-5>, "Q19": <1-5>, "Q20": <1-5>,
  "Q21": "<free text>", "Q22": "<free text>", "Q23": "<free text>"
}}"""


# --------------------------- response validation --------------------------

def validate_per_video(d):
    """Light validation; returns a list of warnings (empty = clean)."""
    if not isinstance(d, dict):
        return [f"answer is not a JSON object: {type(d).__name__}"]
    warns = []
    for k in ["Q1", "Q7"]:
        if d.get(k) not in YES_NO:
            warns.append(f"{k} not in YES_NO: {d.get(k)!r}")
    if d.get("Q2") not in LIKERT_5:
        warns.append(f"Q2 not in LIKERT_5: {d.get('Q2')!r}")
    if d.get("Q4") not in TRUE_FALSE:
        warns.append(f"Q4 not in TRUE_FALSE: {d.get('Q4')!r}")
    if d.get("Q3") not in CLUSTER_COUNTS:
        warns.append(f"Q3 not in CLUSTER_COUNTS: {d.get('Q3')!r}")
    q5 = d.get("Q5")
    if not (type(q5) is int and 0 <= q5 <= 100):
        warns.append(f"Q5 not int 0-100: {q5!r}")
    q6 = d.get("Q6")
    if not isinstance(q6, dict):
        warns.append(f"Q6 not an object: {q6!r}")
        q6 = {}
    for s in ALGORITHM_STEPS:
        if q6.get(s) not in Q6_COLUMNS:
            warns.append(f"Q6[{s}] invalid: {q6.get(s)!r}")
    for k in ["Q8", "Q9"]:
        value = d.get(k)
        if not isinstance(value, str) or not value.strip():
            warns.append(f"{k} not non-empty text: {value!r}")
    return warns


def validate_overall(d):
    if not isinstance(d, dict):
        return [f"answer is not a JSON object: {type(d).__name__}"]
    warns = []
    for k in ["Q17", "Q18", "Q19", "Q20"]:
        v = d.get(k)
        if not (type(v) is int and 1 <= v <= 5):
            warns.append(f"{k} not int 1-5: {v!r}")
    for k in ["Q21", "Q22", "Q23"]:
        value = d.get(k)
        if not isinstance(value, str) or not value.strip():
            warns.append(f"{k} not non-empty text: {value!r}")
    return warns


def parse_json_reply(text):
    """Robustly extract the first balanced JSON object from a model reply."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    start = t.find("{")
    if start == -1:
        raise ValueError("no JSON object found in reply")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(t[start:i + 1])
    raise ValueError("unbalanced JSON in reply")


# ------------------- structured-output (single-choice) schemas -------------
# Digital equivalent of the Google Form radio buttons: constrained decoding
# makes it impossible for the model to emit anything outside the option
# lists, exactly as the form UI constrained human participants.  No answer
# hints are added — only the same structural constraint humans had.

def per_video_schema():
    """JSON schema for one per-video Q1-Q9 reply."""
    q6_props = {s: {"type": "string", "enum": Q6_COLUMNS}
                for s in ALGORITHM_STEPS}
    return {
        "type": "object",
        "properties": {
            "Q1": {"type": "string", "enum": YES_NO},
            "Q2": {"type": "string", "enum": LIKERT_5},
            "Q3": {"type": "string", "enum": CLUSTER_COUNTS},
            "Q4": {"type": "string", "enum": TRUE_FALSE},
            "Q5": {"type": "integer", "minimum": 0, "maximum": 100},
            "Q6": {"type": "object", "properties": q6_props,
                   "required": list(ALGORITHM_STEPS),
                   "additionalProperties": False},
            "Q7": {"type": "string", "enum": YES_NO},
            "Q8": {"type": "string"},
            "Q9": {"type": "string"},
        },
        "required": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9"],
        "additionalProperties": False,
    }


def overall_schema():
    """JSON schema for the final Q17-Q23 reply."""
    scale = {"type": "integer", "minimum": 1, "maximum": 5}
    return {
        "type": "object",
        "properties": {
            "Q17": scale, "Q18": scale, "Q19": scale, "Q20": scale,
            "Q21": {"type": "string"},
            "Q22": {"type": "string"},
            "Q23": {"type": "string"},
        },
        "required": ["Q17", "Q18", "Q19", "Q20", "Q21", "Q22", "Q23"],
        "additionalProperties": False,
    }
