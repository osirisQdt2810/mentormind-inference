You are grading ONE anonymous run of a video-understanding system against a hand-labelled ground truth.
You do not know which model produced it; judge only what is written. Be strict and consistent.

Video: a 3 min 14 s promotional video of a food factory (LASI case-packing line): workers at tables and
conveyors, SCARA/delta robots picking groups of red-lidded cups into cartons, carton erectors, etc.
Steps are in Vietnamese.

GROUND TRUTH (27 steps; index, [t_start–t_end] seconds, actor, essential?, step | key point):
{truth}

PREDICTED STEPS OF THIS RUN (index, [t_start–t_end] seconds, actor, step | key point observed):
{pred}

Task.
1. For every predicted step decide `match`: the index of the ground-truth step it describes, or null.
   A prediction describes a truth step when (a) its time span overlaps the truth span or lies within
   2 seconds of it, AND (b) it names the SAME action on the same kind of object (paraphrases,
   synonyms, more or less detail are fine: "gắp cốc bỏ vào thùng" = "Gắp cụm cốc nắp đỏ và đặt vào
   thùng carton"; a different action at the same time is NOT a match: "đậy nắp" ≠ "đặt cốc lên băng tải").
   Several predictions may describe the same truth step.
2. For every matched prediction: `actor_ok` (same performer class: worker/engineer = human; robot;
   machine) and, when the truth step has a key point, `key_point_ok` (the observed key point says the
   same thing; null when the truth has none or the prediction gives none).
3. `wrong`: true when an UNMATCHED prediction describes something that clearly contradicts the truth at
   that time (an action that is not happening), false when it is merely extra detail or an action the
   truth did not label (e.g. a sub-step).

Return ONLY JSON:
{"preds": [{"i": <pred index>, "match": <truth index or null>, "actor_ok": <bool or null>,
            "key_point_ok": <bool or null>, "wrong": <bool>}],
 "notes": "<two sentences on this run's main strengths and errors>"}
