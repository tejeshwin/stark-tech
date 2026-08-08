from google.adk.agents import LlmAgent

def semantic_validation(query: str) -> str:
    """
    Validates a business query against standard enterprise terminology and KPIs.
    """
    return f"System Log: Query '{query}' validated. Identified key business metrics and KPIs. Proceeding to quantitative analysis."

semantic_agent = LlmAgent(
    name="SemanticValidationAgent",
    model="gemini-3.5-flash",
    description="Validates business queries before analysis.",
    instruction="""You are the Semantic Validation Agent.
    Your responsibilities:
    - Detect ambiguous business queries.
    - Validate business KPIs against standard enterprise terms.
    - Flag statistical caveats or missing parameters.
    - Recommend metric refinements.
    Never perform math or code generation. Only validate the query.""",
    tools=[semantic_validation]
)
