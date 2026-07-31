"""
Targeted eligibility-age extractor (article text only).

The gold ages come from the trial registry, but for many trials the age criterion
is ALSO stated in the paper's eligibility/methods text ("aged >= 18 years",
"18 to 65 years of age", ...). We extract ONLY when a genuine age pattern fires,
otherwise fall back to the train-optimal constants — so we keep the ~70% of min /
~58% of max that the constants already nail, and only override on strong evidence.

The scoring metric is Jaccard over numbers, so precision matters more than recall:
a wrong extra number hurts. Hence conservative, age-context-anchored patterns.

Public: extract_ages(parsed) -> (min_age_str, max_age_str)
"""
import re

MIN_DEFAULT = '18 Years'
MAX_DEFAULT = 'Not Specified'

# Measured-best policy on Train (number_similarity vs gold):
#   extract max only = +0.0028 overall; extracting min HURTS (the "18" constant is
#   right ~70% of the time and adult trials mention subgroup ages that mislead it).
EXTRACT_MIN = False
EXTRACT_MAX = True

# normalize unicode comparators / dashes
_norm = lambda s: (s.replace('≥', '>=').replace('≤', '<=')
                    .replace('–', '-').replace('—', '-').replace('‒', '-'))

Y = r'(?:years?|yrs?|y\.?o\.?)'          # "years", "yrs", "y.o."
AGE = r'(?:years?|yrs?)\s*(?:of\s*age|old)?'

# High-precision only: an over-eager pattern that grabs a stray number destroys the
# ~70% of min / ~58% of max the constants already get right (Jaccard is unforgiving).

# ── range: "aged 18 to 65 years", "18-65 years of age", "between 18 and 65 years" ──
_RANGE = re.compile(
    rf'(?:aged?\s*|age[ds]?\s*)(?:between\s*)?(\d{{1,3}})\s*(?:-|to|and|until)\s*(\d{{1,3}})\s*{AGE}',
    re.I)

# ── min-only (anchored to age/eligibility language) ──
_MIN = [
    re.compile(rf'(?:aged?\s*)?>=\s*(\d{{1,3}})\s*{Y}', re.I),
    re.compile(rf'(?:at least|minimum age (?:of|:)?)\s*(\d{{1,3}})\s*{Y}', re.I),
    re.compile(rf'(\d{{1,3}})\s*{Y}\s*(?:or older|and older|of age or older|and above|or above)', re.I),
]
# ── max-only ──
_MAX = [
    re.compile(rf'(?:aged?\s*)?<=\s*(\d{{1,3}})\s*{Y}', re.I),
    re.compile(rf'(?:up to|no older than|maximum age (?:of|:)?)\s*(\d{{1,3}})\s*{Y}', re.I),
    re.compile(rf'(\d{{1,3}})\s*{Y}\s*(?:or younger|and younger|or below)', re.I),
]


def _plausible(a):
    return 0 < a <= 120


def _search_text(parsed):
    """Eligibility criteria, then methods — inclusion age cutoffs live in one or the
    other (many papers have no parsed eligibility section and state 'aged >= X years'
    in methods). Abstract is excluded: too many descriptive ages (mean/median). The
    high-precision patterns keep methods' 'mean age 74 years' from matching."""
    for key in ('eligibility_text', 'methods_text'):
        t = (parsed.get(key, '') or '').strip()
        if t:
            yield _norm(t)


_INCL = re.compile(r'inclusion|eligib|recruit|enroll|criteria|were included', re.I)
_SENT = re.compile(r'(?<=[.\n;])')


def _incl_sentences(text):
    return ' '.join(s for s in _SENT.split(text) if _INCL.search(s))


def extract_ages(parsed):
    mn = mx = None
    for text in _search_text(parsed):
        incl = _incl_sentences(text)      # inclusion-criteria sentences only
        # 1) explicit range (strong signal) — prefer inclusion context, else whole text
        for src in (incl, text):
            for a, b in _RANGE.findall(src):
                a, b = int(a), int(b)
                if _plausible(a) and _plausible(b) and a < b:
                    mn = a if mn is None else min(mn, a)
                    mx = b if mx is None else max(mx, b)
            if mn is not None:
                break
        # 2) min-only: ONLY inside inclusion sentences (subgroup/mean ages elsewhere are noise)
        if mn is None:
            for rx in _MIN:
                m = rx.search(incl)
                if m and _plausible(int(m.group(1))):
                    mn = int(m.group(1)); break
        # 3) max-only: whole text (constant captures nothing, so recall is worth more here)
        if mx is None:
            for rx in _MAX:
                m = rx.search(text)
                if m and _plausible(int(m.group(1))):
                    mx = int(m.group(1)); break
        if mn is not None and mx is not None:
            break

    min_str = f'{mn} Years' if (EXTRACT_MIN and mn is not None) else MIN_DEFAULT
    max_str = f'{mx} Years' if (EXTRACT_MAX and mx is not None) else MAX_DEFAULT
    return min_str, max_str
