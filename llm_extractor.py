"""
LLM-based extractor for CohortX Task 1.
Uses Mistral 7B Instruct v0.3 Q4_K_M (llama-cpp-python) to extract:
  - conditions (zero-shot: title + abstract only)
  - minimum_age, maximum_age, eligibility_criteria (one-shot: methods/eligibility text)

Prompt format: Mistral [INST]...[/INST] chat template.
"""

import json
import re
import os

MODEL_PATH = '/mnt/extra_storage/kkolpetinou/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf'

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=16384,
            n_threads=12,
            verbose=False,
        )
    return _llm


# ── Shared utilities ─────────────────────────────────────────────────────────

def _parse_json_output(raw):
    """Parse raw LLM output as JSON, stripping markdown fences if present."""
    text = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    text = re.sub(r'\s*```\s*$', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def _is_looping(raw):
    """Detect LLM repetition loop: one word dominates > 15% of output."""
    words = raw.split()
    if len(words) < 20:
        return False
    from collections import Counter
    top_count = Counter(words).most_common(1)[0][1]
    return top_count / len(words) > 0.15


def _normalise_age(val, default):
    val = str(val).strip() if val else ''
    if not val:
        return default
    if re.match(r'^\d+\s+[Yy]ears?$', val):
        return val
    m = re.search(r'(\d+)', val)
    if m:
        return f"{m.group(1)} Years"
    return default


# ── Conditions extraction (two-shot, title + abstract only) ──────────────────

_CONDITIONS_INSTRUCTIONS = """\
You are a biomedical study tagger. Extract the main medical topics this study is about.

Rules:
- Include diseases, conditions, syndromes, imaging modalities, and procedures if they are a central focus of the study
- Use the name as it appears in the article — abbreviations and mixed forms are fine (e.g. "Hepatocellular Carcinoma (HCC)", "CT", "SPECT-CT")
- Return ONLY valid JSON: {"conditions": ["Topic A", "Topic B"]}"""

# Two-shot examples from GT (PMC12279614 and PMC12680986)
_COND_EXAMPLE1_IN = """\
Title: Fatal Outcome of Intravenous Thrombolysis With an Unexpected Finding of Amyloid-β-Related Angiitis—A Case Report Highlighting a Relevant Scenario With Acute Focal Neurological Deficits and Minimal Radiological Presentation

Abstract: Cerebral amyloid angiopathy (CAA) has been implicated as a risk for developing lobar intracerebral hemorrhage (ICH) after intravenous thrombolysis (IVT) applied for acute ischemic stroke (AIS). However, there is a paucity of cases reported with histopathological CAA diagnosis in this setting, with a single report to imply the role of CAA-related inflammation (CAA-RI). We report clinical, radiological, and neuropathological findings of a patient with a fatal outcome after IVT for AIS."""

_COND_EXAMPLE1_OUT = '{"conditions": ["Hemorrhage", "Cerebral Amyloid Angiopathy", "Stroke"]}'

_COND_EXAMPLE2_IN = """\
Title: Artificial Intelligence in Drug-Coated Cardiovascular Devices: A Narrative Review

Abstract: Drug-coated cardiovascular devices (DCCDs), including drug-eluting stents (DESs) and drug-coated balloons (DCBs), have significantly advanced interventional cardiology by reducing restenosis and improving long-term outcomes. AI-based imaging with computed tomography (CT) and coronary physiology assessment are increasingly integrated into percutaneous coronary intervention (PCI) planning and follow-up."""

_COND_EXAMPLE2_OUT = '{"conditions": ["Percutaneous Coronary Intervention", "Coronary Physiology", "Computed Tomography"]}'


def extract_conditions_with_llm(d):
    """Zero-shot conditions extraction from title + abstract only."""
    llm = _get_llm()

    title = d.get('title', '').strip()
    abstract = d.get('abstract_text', '').strip()[:1500]
    article_text = f"Title: {title}\n\nAbstract: {abstract}"

    prompt = (
        f"[INST] {_CONDITIONS_INSTRUCTIONS}\n\nArticle:\n{article_text} [/INST]"
    )

    out = llm(prompt, max_tokens=150, temperature=0.0, stop=['</s>', '[INST]'])
    raw = out['choices'][0]['text'].strip()

    if _is_looping(raw):
        return str(['Not Specified'])

    result = _parse_json_output(raw)
    conditions = result.get('conditions', [])
    if not isinstance(conditions, list):
        conditions = [str(conditions)] if conditions else []
    conditions = [str(c).strip() for c in conditions if str(c).strip()][:8]
    return str(conditions) if conditions else str(['Not Specified'])


# ── Eligibility extraction (one-shot, methods/eligibility text) ───────────────

def _build_article_text(d):
    """Assemble eligibility/methods focused text for the LLM."""
    parts = []

    abstract = d.get('abstract_text', '').strip()
    if abstract:
        parts.append(f"Abstract: {abstract[:1200]}")

    eligib = d.get('eligibility_text', '').strip()
    if eligib:
        parts.append(f"Eligibility section: {eligib[:3000]}")

    methods = d.get('methods_text', '').strip()
    if methods:
        parts.append(f"Methods/Patients section: {methods[:5000]}")

    if len(parts) <= 2:
        body = d.get('full_body_text', '').strip()
        if body:
            parts.append(f"Article body: {body[:4000]}")

    return '\n\n'.join(parts)


# Example 1 from real GT: PMC11093470 (Hepatocellular Carcinoma, patient study, explicit headings)
_ELIGIBILITY_EXAMPLE_IN = """\
Title: A novel stratification scheme combined with internal arteries in CT imaging for predicting microvascular invasion in hepatocellular carcinoma

Abstract: The surgical and histological databases of two medical institutions were searched to identify patients who underwent hepatectomy for HCC, between January 2012 and December 2017.

Methods/Patients section: The study inclusion criteria were: (a) CT scans acquired within 1 month prior to the surgery; (b) confirmation of HCC diagnosis through pathological examination; and (c) curative surgical resection. The exclusion criteria were: prior antitumor treatment; macrovascular thrombosis or metastasis; perioperative mortality; unqualified image artifacts; tumor rupture; MVI status not reported. Patients aged 18 to 80 years were included."""

_ELIGIBILITY_EXAMPLE_OUT = """\
{"minimum_age": "18 Years", "maximum_age": "80 Years", "eligibility_criteria": "Inclusion Criteria:\\n\\n* CT scans acquired no more than one month before surgery\\n* Confirmation of HCC diagnosis by pathological examination\\n* Curative surgical resection\\n\\nExclusion Criteria:\\n\\n* Prior antitumor treatment\\n* Macrovascular thrombosis or metastasis\\n* Perioperative mortality\\n* Unqualified image artifacts\\n* Tumor rupture\\n* MVI status not reported"}"""

# Example 2 from real GT: PMC10862304 (clinician/reader study — participants are radiologists)
_ELIGIBILITY_EXAMPLE2_IN = """\
Title: AI assisted reader evaluation in acute CT head interpretation (AI-REACT): protocol for a multireader multicase study

Abstract: A non-contrast CT head scan (NCCTH) is the most common imaging investigation in the emergency department. This study evaluates the impact of AI assistance on radiologists, radiographers, and emergency clinicians who review NCCTH in clinical practice using a multireader multicase design.

Methods/Patients section: Participants: 30 volunteer participant readers will be selected from the following groups: Emergency medicine consultants and registrars, general radiologist consultants and registrars, and CT radiographers. Inclusion criteria: Radiologists/radiographers/EM clinicians who review NCCTH as part of their clinical practice. Exclusion criteria: Neuroradiologists. (Non-radiologist groups) Clinicians with previous formal postgraduate CT reporting training. (Emergency medicine group) Clinicians with previous career in radiology/neurosurgery to registrar level."""

_ELIGIBILITY_EXAMPLE2_OUT = """\
{"minimum_age": "", "maximum_age": "", "eligibility_criteria": "Inclusion Criteria:\\n\\n* Radiologists/Radiographers/ED clinicians who review CT head scans as part of their clinical practice\\n\\nExclusion Criteria:\\n\\n* Neuroradiologists\\n* Non-radiologist groups: Clinicians with previous formal postgraduate CT reporting training\\n* Emergency Medicine group: Clinicians with previous career in radiology/neurosurgery to registrar level"}"""

# Example 3 from real GT: PMC11318925 (patient study, prose criteria without explicit headings)
_ELIGIBILITY_EXAMPLE3_IN = """\
Title: Evaluation of left ventricular ejection fraction by a new automatic tool on a pocket ultrasound device: Concordance study with cardiac magnetic resonance imaging

Abstract: This was a prospective, monocentric, observational study. All adult patients with an indication for cardiac MRI underwent a point-of-care ultrasound performed by emergency physicians.

Methods/Patients section: All adult patients undergoing cardiac MRI were eligible for enrollment only when a study emergency physician was available. Patients could be enrolled if they were at least 18 years of age, were not pregnant or under legal guardianship, and if informed consent was obtained. Patients admitted for dyspnea, hypotension, or chest pain were systematically included. Patients unable to understand French or with no indication for MRI were excluded."""

_ELIGIBILITY_EXAMPLE3_OUT = """\
{"minimum_age": "18 Years", "maximum_age": "", "eligibility_criteria": "Inclusion criteria:\\n\\n* Patient over 18 years of age\\n* Management in the investigator centre\\n* Admitted for dyspnea or hypotension or chest pain\\n\\nExclusion criteria:\\n\\n* Age < 18 years\\n* Patient not benefiting from a social security system\\n* Patient deprived of liberty\\n* Patient under the protection of justice, under guardianship or curatorship\\n* Patient refusing to participate in the study\\n* Inability to provide the patient with informed information"}"""

_ELIGIBILITY_INSTRUCTIONS = """\
You are a biomedical data extractor. Extract eligibility information from the article.

Participants may be patients, healthy volunteers, clinicians, radiologists, or other healthcare workers — extract criteria for whoever the study enrolled.

Criteria may appear as explicit headings ("Inclusion Criteria:", "Exclusion Criteria:") OR as prose in the methods section (e.g. "we enrolled consecutive patients who...", "readers were eligible if...", "subjects were excluded when..."). Look for both.

Return ONLY a JSON object with these keys:
- "minimum_age": minimum participant age as "X Years", or "" if not stated
- "maximum_age": maximum participant age as "X Years", or "" if not stated
- "eligibility_criteria": inclusion and exclusion criteria formatted as:
  Inclusion Criteria:

  * criterion 1
  * criterion 2

  Exclusion Criteria:

  * criterion 1

  If no explicit criteria headings exist, extract the participant description from the methods section as inclusion criteria (e.g. who was enrolled, what conditions they had, any demographic restrictions mentioned). Always extract something — never return "Not Specified" for eligibility_criteria."""


def extract_eligibility_with_llm(d):
    """
    Three-shot eligibility extraction using Mistral [INST] template.
    Falls back to safe defaults if output is unparseable or looping.
    """
    llm = _get_llm()

    article_text = _build_article_text(d)

    # Three-shot with Mistral template
    prompt = (
        f"[INST] {_ELIGIBILITY_INSTRUCTIONS}\n\nArticle:\n{_ELIGIBILITY_EXAMPLE_IN} [/INST] "
        f"{_ELIGIBILITY_EXAMPLE_OUT} </s>"
        f"[INST] Article:\n{_ELIGIBILITY_EXAMPLE2_IN} [/INST] "
        f"{_ELIGIBILITY_EXAMPLE2_OUT} </s>"
        f"[INST] Article:\n{_ELIGIBILITY_EXAMPLE3_IN} [/INST] "
        f"{_ELIGIBILITY_EXAMPLE3_OUT} </s>"
        f"[INST] Article:\n{article_text} [/INST]"
    )

    out = llm(prompt, max_tokens=1800, temperature=0.0, stop=['</s>', '[INST]'])
    raw = out['choices'][0]['text'].strip()

    if _is_looping(raw):
        return _defaults()

    result = _parse_json_output(raw)

    min_age = _normalise_age(result.get('minimum_age', ''), default='18 Years')
    max_age = _normalise_age(result.get('maximum_age', ''), default='Not Specified')

    raw_elig = result.get('eligibility_criteria', '')
    if isinstance(raw_elig, dict):
        parts = []
        incl = raw_elig.get('Inclusion Criteria', raw_elig.get('inclusion_criteria', []))
        excl = raw_elig.get('Exclusion Criteria', raw_elig.get('exclusion_criteria', []))
        if incl:
            items = incl if isinstance(incl, list) else [str(incl)]
            parts.append('Inclusion Criteria:\n\n' + '\n'.join(f'* {i}' for i in items if i))
        if excl:
            items = excl if isinstance(excl, list) else [str(excl)]
            parts.append('Exclusion Criteria:\n\n' + '\n'.join(f'* {i}' for i in items if i))
        eligibility = '\n\n'.join(parts) if parts else str(raw_elig).strip()
    else:
        eligibility = str(raw_elig).strip()

    return {
        'minimum_age': min_age,
        'maximum_age': max_age,
        'eligibility_criteria': eligibility,
    }


def _defaults():
    return {
        'minimum_age': '18 Years',
        'maximum_age': 'Not Specified',
        'eligibility_criteria': '',
    }


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from nxml_parser import parse_nxml

    test_ids = ['10232610', '12279614', '12470843']
    if len(sys.argv) > 1:
        test_ids = sys.argv[1:]

    for pmcid in test_ids:
        d = parse_nxml(pmcid)
        if d is None:
            print(f"PMC{pmcid}: parse failed")
            continue
        print(f"\n{'='*60}")
        print(f"PMC{pmcid}: {d['title'][:80]}")
        print(f"  conditions: {extract_conditions_with_llm(d)}")
        result = extract_eligibility_with_llm(d)
        for k, v in result.items():
            print(f"  {k}: {str(v)[:120]}")
