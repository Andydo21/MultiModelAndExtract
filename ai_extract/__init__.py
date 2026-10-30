# AI Extract & Summarize App

# Application-owned Neo4j types; model output never supplies Cypher identifiers.
SEMANTIC_RELATIONS = {name: name.upper() for name in (
    'references', 'defines', 'obligates', 'depends_on', 'part_of', 'describes',
    'applies_to', 'has_amount', 'has_deadline', 'pays_to',
    'located_in', 'designed_by', 'works_for', 'supports', 'integrates_with')}
SEMANTIC_EDGE_TYPES = tuple(SEMANTIC_RELATIONS.values())
RETRIEVAL_EDGE_TYPES = ('RELATED_PASSAGE',)
VISUAL_SOURCE_EDGE_TYPES = ('HAS_FIGURE_TEXT','FIGURE_CAPTION_CANDIDATE','FIGURE_NEAR_TEXT_CANDIDATE')
VISUAL_SOURCE_EDGE_TYPES = ('HAS_FIGURE_TEXT','FIGURE_CAPTION_CANDIDATE','FIGURE_NEAR_TEXT_CANDIDATE')
STRUCTURE_REVIEW_SCHEMA = 'surya-gliner-review-v3-direct-pairs'
