"""
Embedding & Matching Engine
Uses Sentence-Transformers and FAISS for cosine similarity matching between
job descriptions and GitHub profile data.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class EmbeddingEngine:
    """Creates embeddings and performs FAISS-based similarity matching."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name for sentence transformers
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.embeddings = None
        self.documents = []
    
    def create_embeddings(self, texts: list) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def build_index(self, documents: list):
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of text documents to index
        """
        self.documents = documents
        embeddings = self.create_embeddings(documents)
        
        # Convert embeddings to float32 for FAISS
        embeddings = embeddings.astype(np.float32)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.embeddings = embeddings
    
    def search(self, query: str, k: int = 5) -> list:
        """
        Search for top-k most similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of tuples: (document_text, similarity_score)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_embedding = self.create_embeddings([query])[0]
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search FAISS index (returns L2 distances)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            # Convert L2 distance to similarity score (0-1)
            similarity = 1 / (1 + distance)
            results.append({
                "document": self.documents[idx],
                "similarity": float(similarity),
                "distance": float(distance)
            })
        
        return results
    
    def compute_match_score(self, job_description: str, github_profile_text: str) -> float:
        """
        Compute overall match score between job description and GitHub profile.
        
        Args:
            job_description: Job description text
            github_profile_text: Extracted GitHub profile text
            
        Returns:
            Match score between 0 and 1
        """
        # Create embeddings for both
        jd_embedding = self.create_embeddings([job_description])[0]
        profile_embedding = self.create_embeddings([github_profile_text])[0]
        
        # Convert to float32
        jd_embedding = jd_embedding.astype(np.float32)
        profile_embedding = profile_embedding.astype(np.float32)
        
        # Compute cosine similarity
        similarity = np.dot(jd_embedding, profile_embedding) / (
            np.linalg.norm(jd_embedding) * np.linalg.norm(profile_embedding)
        )
        
        # Normalize to 0-1 range (cosine similarity ranges from -1 to 1)
        match_score = (similarity + 1) / 2
        
        return float(match_score)


def create_profile_text(github_data: dict = None, resume_data: dict = None) -> str:
    """
    Convert data dictionary to searchable text format.
    Supports both GitHub data and resume CSV data.
    
    Args:
        github_data: Dictionary from GitHub scraper (optional)
        resume_data: Dictionary from resume CSV (optional)
        
    Returns:
        Formatted text for embedding
    """
    text_parts = []
    
    # Handle GitHub data
    if github_data:
        if github_data.get("bio"):
            text_parts.append(f"Bio: {github_data['bio']}")
        
        if github_data.get("company"):
            text_parts.append(f"Company: {github_data['company']}")
        
        languages = github_data.get("top_5_languages", {})
        if languages:
            lang_text = ", ".join(languages.keys())
            text_parts.append(f"Languages: {lang_text}")
        
        text_parts.append(f"Total Stars: {github_data.get('total_stars', 0)}")
        text_parts.append(f"Public Repositories: {github_data.get('public_repos', 0)}")
        
        readmes = github_data.get("readme_content", [])
        if readmes:
            readme_text = " ".join([r["content"] for r in readmes[:3]])
            text_parts.append(f"Projects: {readme_text[:500]}")
    
    # Handle resume data
    if resume_data:
        if resume_data.get("name"):
            text_parts.append(f"Name: {resume_data['name']}")
        
        if resume_data.get("skills"):
            text_parts.append(f"Skills: {resume_data['skills']}")
        
        if resume_data.get("experience_years"):
            text_parts.append(f"Experience: {resume_data['experience_years']} years")
        
        if resume_data.get("education"):
            text_parts.append(f"Education: {resume_data['education']}")
        
        if resume_data.get("resume_text"):
            text_parts.append(f"Background: {resume_data['resume_text']}")
    
    return " ".join(text_parts)


def match_candidate_to_jobs(
    candidate_text: str,
    jobs: list,
    top_k: int = 3
) -> list:
    """
    Match a candidate's profile to multiple jobs and return top matches.
    
    Args:
        candidate_text: Candidate profile text
        jobs: List of job dictionaries
        top_k: Number of top matches to return
        
    Returns:
        List of job matches with scores
    """
    engine = EmbeddingEngine()
    results = []
    
    for job in jobs:
        job_description = f"{job.get('job_title')} at {job.get('company')}. {job.get('description')}"
        match_score = engine.compute_match_score(job_description, candidate_text)
        
        results.append({
            "job_id": job.get("id"),
            "job_title": job.get("job_title"),
            "company": job.get("company"),
            "description": job.get("description"),
            "required_skills": job.get("required_skills"),
            "experience_level": job.get("experience_level"),
            "match_score": match_score
        })
    
    # Sort by match score descending
    results.sort(key=lambda x: x["match_score"], reverse=True)
    
    return results[:top_k]
