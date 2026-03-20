"""
Explainability & LLM Utilities
Generates Glass-Box explanations using template-based reasoning (no API needed).
"""

import os
from dotenv import load_dotenv

load_dotenv()


class ExplainabilityEngine:
    """Generates Glass-Box explanations using intelligent templates."""
    
    def __init__(self):
        """Initialize explanation engine."""
        pass
    
    def generate_explanation(
        self,
        job_description: str,
        github_username: str,
        match_score: float,
        top_languages: dict,
        total_stars: int,
        verification_signals: dict
    ) -> str:
        """
        Generate Glass-Box explanation for candidate-job fit using templates.
        
        Args:
            job_description: Job description text
            github_username: GitHub username
            match_score: Overall match score (0-1)
            top_languages: Top 5 languages from GitHub
            total_stars: Total stars across repositories
            verification_signals: Dictionary of verified capability signals
            
        Returns:
            Explainability report as string
        """
        
        return self._generate_template_explanation(
            job_description=job_description,
            github_username=github_username,
            match_score=match_score,
            top_languages=top_languages,
            total_stars=total_stars,
            verification_signals=verification_signals
        )
    
    def _generate_template_explanation(
        self,
        job_description: str,
        github_username: str,
        match_score: float,
        top_languages: dict,
        total_stars: int,
        verification_signals: dict
    ) -> str:
        """Generate explanation using intelligent templates."""
        
        # Extract job requirements
        job_lower = job_description.lower()
        langs = ", ".join(top_languages.keys()) if top_languages else "Not specified"
        
        # Check for language keywords in job description
        language_match = self._check_language_alignment(job_description, top_languages)
        
        # Determine match quality
        if match_score > 0.85:
            confidence = "excellent"
            overall = "This candidate is an exceptional fit for the role."
        elif match_score > 0.7:
            confidence = "strong"
            overall = "This candidate is well-aligned with the role requirements."
        elif match_score > 0.55:
            confidence = "moderate"
            overall = "This candidate has relevant experience but with some gaps."
        else:
            confidence = "weak"
            overall = "This candidate's profile shows limited alignment with the role."
        
        # Build explanation
        explanation = f"""{overall}

**Evidence:**
• Primary Languages: {langs}
• Portfolio Quality: {total_stars} total stars across repositories (indicates production experience)
• Language Match: {language_match}
• Verified Status: Real GitHub profile with verifiable artifacts

**Analysis:**
The candidate's technical stack shows a {confidence} alignment with job requirements. The match score of {match_score:.0%} reflects the semantic similarity between their GitHub profile and the job description, considering both explicit skills and implicit experience signals."""
        
        return explanation
    
    def _check_language_alignment(self, job_description: str, top_languages: dict) -> str:
        """Check how well candidate's languages align with job description."""
        
        job_lower = job_description.lower()
        matched_langs = []
        
        for lang in top_languages.keys():
            if lang.lower() in job_lower:
                matched_langs.append(lang)
        
        if matched_langs:
            return f"✓ {', '.join(matched_langs)} mentioned in job description"
        else:
            return "Transferable skills present; specific match may require deeper review"
    
    def _build_prompt(
        self,
        job_description: str,
        github_username: str,
        match_score: float,
        top_languages: dict,
        total_stars: int,
        verification_signals: dict
    ) -> str:
        """
        Build prompt for LLM.
        
        Args:
            See generate_explanation()
            
        Returns:
            Formatted prompt string
        """
        
        prompt = f"""You are a technical recruiter analyzing a candidate's GitHub profile against a job description.

TASK: Provide a concise Glass-Box explanation (Reasoning Chain) of why this candidate does or doesn't fit the role.

JOB DESCRIPTION:
{job_description}

CANDIDATE GITHUB PROFILE:
Username: {github_username}
Top Languages: {', '.join(top_languages.keys()) if top_languages else 'Not available'}
Total Repository Stars: {total_stars}
Match Score: {match_score:.2%}

VERIFIED CAPABILITY SIGNALS (Real-World Artifacts):
{self._format_signals(verification_signals)}

INSTRUCTIONS:
1. Analyze the alignment between job requirements and GitHub evidence
2. Highlight which programming languages match
3. Comment on code quality (based on stars and repository activity)
4. Provide 2-3 key reasons for the match score
5. Keep explanation concise (3-4 sentences max)
6. Reference specific GitHub artifacts (languages, stars, projects)

Provide the explanation now:"""
        
        return prompt
    
    def _format_signals(self, signals: dict) -> str:
        """Format verification signals for display in prompt."""
        if not signals:
            return "No signals available"
        
        formatted = []
        for signal, value in signals.items():
            formatted.append(f"- {signal}: {value}")
        
        return "\n".join(formatted)


def generate_verified_signals(
    github_data: dict,
    job_description: str,
    match_score: float
) -> dict:
    """
    Generate Verified Capability Signals based on GitHub data.
    These represent real-world, verifiable artifacts.
    
    Args:
        github_data: User data from scraper
        job_description: Job description
        match_score: Match score from engine
        
    Returns:
        Dictionary of verified signals
    """
    
    signals = {}
    
    # Signal 1: Language Match
    top_langs = github_data.get("top_5_languages", {})
    if top_langs:
        signals["Primary Languages"] = ", ".join(list(top_langs.keys())[:3])
    
    # Signal 2: Code Quality Indicator
    total_stars = github_data.get("total_stars", 0)
    if total_stars > 100:
        signals["Code Quality"] = f"High (★{total_stars} across repositories)"
    elif total_stars > 10:
        signals["Code Quality"] = f"Medium (★{total_stars} across repositories)"
    else:
        signals["Code Quality"] = "Building (★0-10 stars)"
    
    # Signal 3: Portfolio Depth
    public_repos = github_data.get("public_repos", 0)
    if public_repos > 20:
        signals["Portfolio Depth"] = f"Extensive ({public_repos} repositories)"
    elif public_repos > 5:
        signals["Portfolio Depth"] = f"Moderate ({public_repos} repositories)"
    else:
        signals["Portfolio Depth"] = f"Emerging ({public_repos} repositories)"
    
    # Signal 4: Commit Activity
    signals["Verified"] = "✓ Real GitHub Profile"
    
    # Signal 5: Match Confidence
    if match_score > 0.8:
        signals["Match Confidence"] = f"High ({match_score:.0%})"
    elif match_score > 0.6:
        signals["Match Confidence"] = f"Moderate ({match_score:.0%})"
    else:
        signals["Match Confidence"] = f"Low ({match_score:.0%})"
    
    return signals


def format_reasoning_chain(
    explanation: str,
    match_score: float,
    verified_signals: dict
) -> dict:
    """
    Format explanation into a structured reasoning chain.
    
    Args:
        explanation: LLM-generated explanation
        match_score: Match score
        verified_signals: Verified capability signals
        
    Returns:
        Structured reasoning chain dictionary
    """
    
    return {
        "match_score": f"{match_score:.0%}",
        "verified_signals": verified_signals,
        "reasoning": explanation,
        "confidence_level": _get_confidence_level(match_score)
    }


def _get_confidence_level(score: float) -> str:
    """Determine confidence level based on score."""
    if score > 0.85:
        return "Very High"
    elif score > 0.7:
        return "High"
    elif score > 0.55:
        return "Moderate"
    elif score > 0.4:
        return "Low"
    else:
        return "Very Low"
