"""
Step 1: NXML Parser for CohortX Challenge Task 1
JATS XML structure: front/article-meta, body/sec, back/ref-list
Python: miniconda base (lxml + openpyxl available; no pandas)
"""

import xml.etree.ElementTree as ET
import os
import re


NXML_DIR = '/home/kkolpetinou/cohort-x-task-1/PMC_NXML_Archives'


def itertext_no_refs(el):
    """Yield text from element, skipping <ref>, <xref>, <fn> subtrees."""
    SKIP_TAGS = {'ref', 'xref', 'fn', 'ref-list', 'table-wrap', 'fig',
                 'supplementary-material', 'inline-formula', 'disp-formula'}
    tag = el.tag.split('}')[-1] if '}' in el.tag else el.tag
    if tag in SKIP_TAGS:
        return
    if el.text:
        yield el.text
    for child in el:
        yield from itertext_no_refs(child)
        if child.tail:
            yield child.tail


def get_text(el):
    """Get clean concatenated text from an element, stripping refs."""
    return re.sub(r'\s+', ' ', ''.join(itertext_no_refs(el))).strip()


def parse_nxml(pmcid):
    """
    Parse a single NXML file and return a structured dict with:
      - pmcid
      - title
      - abstract_text       (full abstract as single string)
      - abstract_sections   (dict: section_title -> text)
      - article_type        (raw JATS article-type attribute)
      - body_sections       (list of dicts: {title, text, id})
      - eligibility_text    (text from eligibility/inclusion/exclusion sections, joined)
      - methods_text        (text from methods/participants sections, joined)
      - full_body_text      (all body text joined)
    Returns None if the file cannot be parsed.
    """
    path = os.path.join(NXML_DIR, f'PMC{pmcid}.nxml')
    if not os.path.exists(path):
        return None

    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return None

    root = tree.getroot()
    result = {'pmcid': pmcid}

    # ── Article-level metadata ──────────────────────────────────────────────
    result['article_type'] = root.get('article-type', '')

    # Title
    title_el = root.find('.//{*}article-title') if '{' in (root.tag) else root.find('.//article-title')
    # Use plain find since we're using stdlib ET (no namespace in these files)
    title_el = root.find('.//article-title')
    result['title'] = get_text(title_el) if title_el is not None else ''

    # ── Abstract ────────────────────────────────────────────────────────────
    abstract_el = root.find('.//abstract')
    result['abstract_text'] = get_text(abstract_el) if abstract_el is not None else ''

    # Structured abstract subsections (Background, Methods, Results, etc.)
    abstract_sections = {}
    if abstract_el is not None:
        for sec in abstract_el.findall('sec'):
            t_el = sec.find('title')
            sec_title = get_text(t_el).strip() if t_el is not None else 'Abstract'
            sec_text = get_text(sec)
            # remove the title text from the start of sec_text
            if sec_text.startswith(sec_title):
                sec_text = sec_text[len(sec_title):].strip()
            abstract_sections[sec_title] = sec_text
    result['abstract_sections'] = abstract_sections

    # ── Body sections ────────────────────────────────────────────────────────
    body_el = root.find('.//body')
    body_sections = []
    eligibility_chunks = []
    methods_chunks = []

    ELIGIBILITY_KWS = {'eligib', 'inclusion', 'exclusion', 'inclusion/exclusion'}
    METHODS_KWS = {'method', 'patient', 'participant', 'study design', 'subject',
                   'study population', 'study sample', 'cohort'}

    if body_el is not None:
        # Walk top-level sections only (depth-1 children of body)
        # but also recurse for nested subsections
        def collect_sections(parent_el, depth=0):
            for sec in parent_el.findall('sec'):
                title_el = sec.find('title')
                title = get_text(title_el).strip() if title_el is not None else ''
                text = get_text(sec)
                # strip title from text start
                if text.startswith(title):
                    text = text[len(title):].strip()

                entry = {
                    'id': sec.get('id', ''),
                    'title': title,
                    'text': text,
                    'depth': depth,
                }
                body_sections.append(entry)

                title_lower = title.lower()
                # Eligibility heuristic
                if any(kw in title_lower for kw in ELIGIBILITY_KWS):
                    eligibility_chunks.append(text)
                # Methods heuristic
                if any(kw in title_lower for kw in METHODS_KWS):
                    methods_chunks.append(text)

                collect_sections(sec, depth + 1)

        collect_sections(body_el)

    result['body_sections'] = body_sections
    result['eligibility_text'] = '\n\n'.join(eligibility_chunks)
    result['methods_text'] = '\n\n'.join(methods_chunks)
    result['full_body_text'] = '\n\n'.join(
        f"{s['title']}\n{s['text']}" for s in body_sections if s['depth'] == 0
    )

    # ── All paragraphs from body (for inline criteria / age search) ──────────
    # Collect every <p> element in the body, preserving reading order
    all_paragraphs = []
    if body_el is not None:
        for p_el in body_el.iter('p'):
            text = get_text(p_el)
            if text:
                all_paragraphs.append(text)
    result['all_paragraphs'] = all_paragraphs
    result['all_paragraphs_text'] = ' '.join(all_paragraphs)

    # Extract author-assigned keywords from kwd-group
    keywords = []
    for kwd_group in root.iter('kwd-group'):
        for kwd in kwd_group.iter('kwd'):
            text = get_text(kwd).strip()
            if text:
                keywords.append(text)
    result['keywords'] = keywords if keywords else []

    return result


def demo(pmcid):
    """Print a structured summary for one article."""
    d = parse_nxml(pmcid)
    if d is None:
        print(f"Could not parse PMC{pmcid}")
        return

    print(f"{'='*70}")
    print(f"PMC{d['pmcid']}  |  article-type: {d['article_type']}")
    print(f"TITLE: {d['title'][:120]}")
    print(f"\nABSTRACT ({len(d['abstract_text'])} chars):")
    print(f"  {d['abstract_text'][:300]}...")
    if d['abstract_sections']:
        print(f"\n  Structured subsections: {list(d['abstract_sections'].keys())}")

    print(f"\nBODY SECTIONS ({len(d['body_sections'])} total):")
    for s in d['body_sections']:
        indent = '  ' * s['depth']
        print(f"  {indent}[{s['id']}] {s['title']!r}  ({len(s['text'])} chars)")

    if d['eligibility_text']:
        print(f"\nELIGIBILITY TEXT ({len(d['eligibility_text'])} chars):")
        print(f"  {d['eligibility_text'][:400]}...")
    else:
        print("\nELIGIBILITY TEXT: (none found via section headings)")

    if d['methods_text']:
        print(f"\nMETHODS TEXT ({len(d['methods_text'])} chars):")
        print(f"  {d['methods_text'][:300]}...")


if __name__ == '__main__':
    import sys

    nxml_dir = NXML_DIR
    all_files = sorted(os.listdir(nxml_dir))
    pmcids = [f.replace('PMC', '').replace('.nxml', '') for f in all_files if f.endswith('.nxml')]

    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        # Demo on 3 representative files
        targets = ['10232610', '10410925', '10011737']

    for pmcid in targets:
        demo(pmcid)
        print()

    # Quick survey: how many files have eligibility text?
    if '--survey' in sys.argv:
        has_eligib = 0
        has_methods = 0
        total = 0
        for pmcid in pmcids:
            d = parse_nxml(pmcid)
            if d:
                total += 1
                if d['eligibility_text']:
                    has_eligib += 1
                if d['methods_text']:
                    has_methods += 1
        print(f"\nSurvey ({total} files):")
        print(f"  Has eligibility section text: {has_eligib} ({100*has_eligib/total:.1f}%)")
        print(f"  Has methods/patients section: {has_methods} ({100*has_methods/total:.1f}%)")
