import re
from google.adk.agents import LlmAgent

OUT_OF_SCOPE_KEYWORDS = [
    r'\bflight(s)?\b', r'\bbook(ing)?\b', r'\btravel\b', r'\bhotel(s)?\b', r'\bairline(s)?\b',
    r'\bweather\b', r'\btemperature\b', r'\brain\b', r'\bforecast weather\b', r'\bsunny\b',
    r'\brecipe(s)?\b', r'\bfood\b', r'\bcook(ing)?\b', r'\bdinner\b', r'\brestaurant\b',
    r'\bjoke(s)?\b', r'\bmovie(s)?\b', r'\bsong(s)?\b', r'\bmusic\b', r'\bgame(s)?\b', r'\bsports\b',
    r'\bpersonal\b', r'\bvacation\b', r'\bholiday\b', r'\bchat\b', r'\bwho are you\b'
]

def semantic_validation(query: str) -> dict:
    """
    Validates a user query against enterprise data scope and safety guardrails.
    Returns dict containing validation status and response message.
    """
    query_lower = query.lower().strip()
    
    for pattern in OUT_OF_SCOPE_KEYWORDS:
        if re.search(pattern, query_lower):
            return {
                "is_valid": False,
                "rejection_message": "[Semantic Validation Guardrail] Request Denied: Out of Scope. This enterprise decision support system is restricted strictly to organizational data analysis, operational metrics, and predictive forecasting."
            }
            
    business_keywords = [
        'revenue', 'impact', 'sla', 'breach', 'kpi', 'department', 'transaction', 'amount',
        'query', 'processing', 'time', 'cost', 'savings', 'anomaly', 'score', 'priority',
        'risk', 'escalation', 'outcome', 'review', 'agent', 'accuracy', 'customer', 'data',
        'count', 'average', 'mean', 'max', 'min', 'sum', 'forecast', 'predict', 'trend',
        'chart', 'plot', 'distribution', 'complexity', 'record', 'column', 'value'
    ]
    
    has_business_term = any(term in query_lower for term in business_keywords)
    
    if not has_business_term and len(query_lower.split()) > 2 and not any(w in query_lower for w in ['show', 'get', 'calculate', 'what', 'how', 'list', 'display']):
        return {
            "is_valid": False,
            "rejection_message": "[Semantic Validation Guardrail] Request Denied: Out of Scope. This enterprise decision support system is restricted strictly to organizational data analysis, operational metrics, and predictive forecasting."
        }
        
    return {
        "is_valid": True,
        "validation_log": f"[Semantic Validation Guardrail] Query '{query}' validated within enterprise scope. Proceeding to quantitative analysis."
    }

semantic_agent = LlmAgent(
    name="SemanticValidationAgent",
    model="gemini-3.5-flash",
    description="Validates user business queries and enforces out-of-bounds safety guardrails.",
    instruction="""You are the Semantic Validation and Safety Guardrail Agent. Your primary job is to inspect user queries BEFORE any data analysis or tool execution occurs. 
If a query is unrelated to enterprise data, corporate analytics, key performance indicators (KPIs), operational metrics, or predictive forecasting (e.g., booking travel, checking weather, personal tasks, general trivia), you MUST immediately reject it. 
Return a clean, professional refusal message prefixed with '[Semantic Validation Guardrail]': 'Request Denied: Out of Scope. This enterprise decision support system is restricted strictly to organizational data analysis, operational metrics, and predictive forecasting.' Never execute tools or pass out-of-scope queries to downstream analytical agents.""",
    tools=[semantic_validation]
)
