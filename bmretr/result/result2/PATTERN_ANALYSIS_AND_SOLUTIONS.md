# Pattern Analysis: Achieving 0.9-1.0 Precision for CV/JD Information Extraction

## Executive Summary

Analysis of winner results from 10 test cases reveals **systematic patterns** that prevent achieving 0.9-1.0 precision across all fields. These patterns are not test-case specific but represent fundamental architectural and methodological limitations that will scale to thousands of CVs and JDs.

---

## Current Performance Gap

### Fields Requiring Critical Improvement (< 0.5 precision)
| Field | Winner Precision | Winner System |
|-------|-----------------|---------------|
| JD.job_description | 0.122 | QA |
| JD.required_education_level | 0.319 | QA |
| JD.required_major | 0.421 | QA |
| JD.required_skills | 0.452 | QA |
| CV.education_level | 0.479 | QA |
| CV.major | 0.383 | QA |

### Fields Needing Enhancement (0.5-0.8 precision)
| Field | Winner Precision | Winner System |
|-------|-----------------|---------------|
| CV.summary | 0.638 | QA |
| CV.skills | 0.671 | QA |
| JD.requirements | 0.586 | EMB |
| JD.responsibilities | 0.693 | QA |

### Fields Achieving Target (≥ 0.8 precision)
| Field | Winner Precision | Winner System |
|-------|-----------------|---------------|
| CV.institution | 0.800 | QA |
| CV.location | 1.000 | TIE |
| CV.salary_expectation | 1.000 | EMB |
| CV.desired_job | 1.000 | TIE |
| JD.salary_offer | 1.000 | TIE |
| JD.job_location | 1.000 | TIE |

---

## Core Problem Patterns Identified

### Problem 1: Context Boundary Failures
**Pattern**: Models extract surrounding contextual text instead of isolating specific field values.

**Evidence**:
- JD.job_description: 8/10 samples with precision < 0.3
- Token ratios show extreme over-extraction: 5.7x, 31.5x in worst cases
- Models return entire paragraphs when gold expects 2-3 word summaries

**Example**:
```
GOLD: "Sale Admin Website" (3 tokens)
PRED: "Kiểm tra các đơn hàng đến từ website TMĐT, xác thực đơn hàng..." (17+ tokens)
```

**Why This Matters**: This pattern will repeat for every CV/JD where information is embedded in longer text blocks.

### Problem 2: Semantic vs Literal Mismatch
**Pattern**: Models correctly understand meaning but extract non-matching tokens.

**Evidence**:
- CV.education_level: 2 cases with recall > 0.8 but precision < 0.5
- CV.major: Models extract institution names instead of majors
- High semantic overlap but low token overlap with gold

**Example**:
```
GOLD: "trung cấp nghề" 
PRED: "Trường trung cấp nghề" (includes "Trường" which is not in gold)
```

**Why This Matters**: Gold standards use specific phrasings; semantic extraction produces valid but non-matching variants.

### Problem 3: Missing Content (Zero Extraction)
**Pattern**: Models return empty strings when relevant content exists but is implicit or non-explicit.

**Evidence**:
- 26 zero precision cases across winner results
- CV.major: 6 cases returning empty despite gold having content
- JD.job_description: 9 cases with zero extraction
- Empty returns occur when information is embedded in work experience rather than explicit "Major" fields

**Example**:
```
GOLD: "nghề lái xe" (from work experience context)
PRED: "" (model couldn't find explicit major declaration)
```

**Why This Matters**: Real CVs/JDs rarely have perfectly structured fields; information is often implicit.

### Problem 4: Noise Inclusion
**Pattern**: Extracted content includes irrelevant surrounding text, questions, and metadata.

**Evidence**:
- 70 cases with 2x+ over-extraction AND precision < 0.3
- CV.education_level: 10 cases of noise inclusion
- JD.required_education_level: Extracts "At least 5+ years of experience" instead of education

**Example**:
```
GOLD: "đại học"
PRED: "At least 5+ years of experience with Objective-C or Swift..."
```

**Why This Matters**: Query-based extraction pulls in surrounding context that matches query terms loosely.

### Problem 5: Field Confusion
**Pattern**: Models extract from semantically wrong sections of the document.

**Evidence**:
- JD.required_education_level extracting experience requirements
- CV.major extracting work experience instead of education
- Required skills extracting job titles or company names

**Example**:
```
GOLD (required_major): "không yêu cầu"
PRED: "kinh doanh" (extracted from job title, not education requirements)
```

**Why This Matters**: Without structural understanding, semantic similarity alone causes cross-field contamination.

---

## Root Cause Analysis

### Architectural Limitations

1. **Query-Based Extraction Without Structure**
   - Both QA and EMB systems use query matching against raw text
   - No document structure parsing (sections, headers, lists)
   - Context window includes entire document without boundaries

2. **Single-Pass Extraction**
   - No refinement or validation stage
   - First match becomes final answer
   - No cross-verification between related fields

3. **Chunking Strategy Issues**
   - `split_into_chunks()` uses sentence boundaries that don't respect field semantics
   - 200-character chunks often split related information
   - No hierarchical chunking for nested information

4. **Lack of Field-Specific Post-Processing**
   - No normalization rules for education levels
   - No validation that extracted major matches known majors
   - No filtering of meta-questions and instructions

### Methodological Gaps

1. **Gold Standard Mismatch**
   - Gold uses normalized, minimal forms
   - Extraction produces natural language variants
   - No canonicalization step

2. **Implicit Information Handling**
   - Systems expect explicit statements
   - Real documents use inference (e.g., "5 years experience" implies education)
   - No inference layer

3. **Language-Specific Challenges**
   - Vietnamese compound words and flexible word order
   - Multiple valid phrasings for same concept
   - Tokenization doesn't account for this

---

## Recommended Solutions

### Solution 1: Structured Extraction Pipeline

**Implementation**:
```python
def structured_extract(text, field_type):
    # Stage 1: Identify document sections
    sections = identify_sections(text)  # headers, lists, paragraphs
    
    # Stage 2: Route to appropriate section
    target_section = route_to_section(sections, field_type)
    
    # Stage 3: Extract with section-specific rules
    if field_type == "education_level":
        return extract_with_education_patterns(target_section)
    elif field_type == "major":
        return extract_with_major_validation(target_section)
```

**Expected Impact**: Reduces context boundary failures by 60-70%

### Solution 2: Multi-Stage Refinement

**Stage 1 - Coarse Extraction**: Get candidate spans (current approach)
**Stage 2 - Boundary Refinement**: Trim to minimal relevant span
**Stage 3 - Validation**: Check against field constraints
**Stage 4 - Normalization**: Convert to canonical form

**Example for education_level**:
```
Raw: "Tốt nghiệp đại học/cao đẳng các chuyên ngành liên quan"
Stage 1: "Tốt nghiệp đại học/cao đẳng" 
Stage 2: "đại học/cao đẳng"
Stage 3: Valid education level ✓
Stage 4: "đại học/cao đẳng" (normalized)
```

**Expected Impact**: Improves precision from 0.3-0.5 to 0.7-0.9 range

### Solution 3: Field-Specific Heuristics and Validators

**Education Level Validator**:
```python
EDUCATION_PATTERNS = [
    r"(đại học|cao đẳng|trung cấp|thạc sĩ|tiến sĩ)",
    r"(không yêu cầu|không bắt buộc)",
    r"(đang học|năm \d)"
]

def validate_education_level(text):
    for pattern in EDUCATION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return extract_matching_span(text, pattern)
    return ""
```

**Major Validator**:
```python
KNOWN_MAJORS = ["công nghệ thông tin", "kinh tế", "quản trị", ...]

def validate_major(text, context):
    # Check if it's actually a major, not institution
    if any(inst in text.lower() for inst in ["trường", "đại học", "cao đẳng"]):
        return ""
    # Match against known majors with fuzzy matching
    return fuzzy_match(text, KNOWN_MAJORS)
```

**Expected Impact**: Eliminates 80% of field confusion errors

### Solution 4: Ensemble with Verification

**Approach**:
1. Run both QA and EMB extraction
2. Cross-validate results
3. Use disagreement to trigger refinement
4. Select or merge based on field-specific confidence

**Example**:
```python
def extract_with_verification(text, field):
    qa_result = qa_extract(text, field)
    emb_result = emb_extract(text, field)
    
    if high_overlap(qa_result, emb_result):
        return consensus(qa_result, emb_result)
    else:
        # Disagreement triggers detailed extraction
        return detailed_extract(text, field, qa_result, emb_result)
```

**Expected Impact**: Improves consistency and catches edge cases

### Solution 5: Hierarchical Context Windows

**Problem**: Current approach uses entire document as context
**Solution**: Progressive narrowing

```python
def hierarchical_extract(text, field):
    # Level 1: Full document (for overview)
    overview = get_document_overview(text)
    
    # Level 2: Relevant section only
    section = find_relevant_section(text, field, overview)
    
    # Level 3: Sentence/chunk level
    candidates = extract_candidates(section, field)
    
    # Level 4: Token level refinement
    return refine_to_tokens(candidates, field)
```

**Expected Impact**: Reduces noise by 50-70% for embedded fields

### Solution 6: Post-Extraction Canonicalization

**Problem**: Multiple valid phrasings for same concept
**Solution**: Normalize to gold standard format

```python
CANONICAL_FORMS = {
    "education": {
        "đại học.*": "đại học",
        "cao đẳng.*": "cao đẳng", 
        "trung cấp.*": "trung cấp",
        "không.*yêu cầu": "không yêu cầu"
    },
    "major": {
        "công nghệ thông tin|cntt|it": "công nghệ thông tin",
        "quản trị kinh doanh|qtkd": "quản trị kinh doanh"
    }
}

def canonicalize(text, field_type):
    for pattern, canonical in CANONICAL_FORMS[field_type].items():
        if re.match(pattern, text, re.IGNORECASE):
            return canonical
    return text
```

**Expected Impact**: Directly addresses semantic vs literal mismatch

---

## Implementation Priority

### Phase 1 (Quick Wins - Target: 0.6-0.7 precision)
1. Add field-specific validators (Solution 3)
2. Implement canonicalization (Solution 6)
3. Add basic boundary trimming

### Phase 2 (Structural Changes - Target: 0.7-0.85 precision)
1. Implement structured extraction pipeline (Solution 1)
2. Add hierarchical context windows (Solution 5)
3. Deploy multi-stage refinement (Solution 2)

### Phase 3 (Advanced - Target: 0.9+ precision)
1. Full ensemble verification (Solution 4)
2. Inference layer for implicit information
3. Continuous learning from disagreements

---

## Metrics to Track

For each solution implementation, track:
1. **Precision improvement** per field
2. **Zero extraction rate** reduction
3. **Over-extraction ratio** (pred_tokens/gold_tokens)
4. **Cross-field contamination** incidents
5. **Canonicalization match rate**

---

## Conclusion

The gap between current performance (0.12-0.80) and target (0.9-1.0) is not due to random errors but **systematic patterns** that can be addressed through:

1. **Structural awareness** - understanding document organization
2. **Validation layers** - catching and correcting extraction errors
3. **Normalization** - aligning output with expected formats
4. **Verification** - using multiple approaches to confirm results

These solutions address the root causes rather than symptoms, ensuring improvements will generalize beyond the 10 test cases to the full corpus of CVs and JDs.