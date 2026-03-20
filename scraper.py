"""
GitHub Profile Scraper Module & Dataset Loader
Fetches user's top 5 languages, total stars, and README content from non-forked repos.
Also loads resume data from CSV datasets.
"""

import os
import csv
from github import Github
from github.GithubException import GithubException
from dotenv import load_dotenv
from collections import Counter

load_dotenv()


class GitHubScraper:
    """Scrapes GitHub profile data for skill verification."""
    
    def __init__(self):
        """Initialize GitHub API client."""
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_TOKEN not found in .env file")
        self.github = Github(github_token)
    
    def fetch_user_data(self, username: str) -> dict:
        """
        Fetch comprehensive user data from GitHub.
        
        Args:
            username: GitHub username
            
        Returns:
            Dictionary containing user data, top languages, stars, and README content
        """
        try:
            user = self.github.get_user(username)
            
            # Basic user info
            user_data = {
                "username": user.login,
                "name": user.name,
                "bio": user.bio,
                "public_repos": user.public_repos,
                "followers": user.followers,
                "public_gists": user.public_gists,
                "company": user.company,
                "location": user.location,
                "email": user.email,
                "blog": user.blog,
                "created_at": user.created_at,
                "updated_at": user.updated_at,
            }
            
            # Fetch repository data
            repos_data = self._fetch_repos_data(user)
            user_data.update(repos_data)
            
            return user_data
            
        except GithubException as e:
            raise ValueError(f"Failed to fetch GitHub user '{username}': {str(e)}")
    
    def _fetch_repos_data(self, user) -> dict:
        """
        Fetch top 5 languages, total stars, and README content from non-forked repos.
        
        Args:
            user: GitHub User object
            
        Returns:
            Dictionary with languages, stars, and README content
        """
        language_counter = Counter()
        total_stars = 0
        readme_content = []
        
        try:
            repos = user.get_repos(type="owner")
            
            for repo in repos:
                # Skip forked repositories
                if repo.fork:
                    continue
                
                # Accumulate stars
                total_stars += repo.stargazers_count
                
                # Count languages
                if repo.language:
                    language_counter[repo.language] += 1
                
                # Fetch README content from non-forked repos
                try:
                    readme = repo.get_readme()
                    readme_content.append({
                        "repo": repo.name,
                        "content": readme.decoded_content.decode('utf-8')[:500]  # First 500 chars
                    })
                except:
                    pass  # Repo might not have README
        
        except Exception as e:
            print(f"Warning: Error fetching repository data: {str(e)}")
        
        # Get top 5 languages
        top_5_languages = dict(language_counter.most_common(5))
        
        return {
            "top_5_languages": top_5_languages,
            "total_stars": total_stars,
            "readme_content": readme_content,
        }


def get_github_profile(username: str) -> dict:
    """
    Convenience function to fetch GitHub profile data.
    
    Args:
        username: GitHub username
        
    Returns:
        User data dictionary
    """
    scraper = GitHubScraper()
    return scraper.fetch_user_data(username)


def load_resumes_from_csv(filepath: str = "data/Resume.csv") -> list:
    """
    Load resume data from CSV file with Resume_str and Category columns.
    
    Args:
        filepath: Path to Resume.csv
        
    Returns:
        List of resume dictionaries
    """
    resumes = []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                resumes.append({
                    "id": row.get("ID"),
                    "resume_text": row.get("Resume_str"),
                    "category": row.get("Category"),
                    # Resume_html is intentionally ignored
                })
    except FileNotFoundError:
        print(f"Warning: {filepath} not found")
    
    return resumes


def load_jobs_from_csv(filepath: str = "data/jobs.csv") -> list:
    """
    Load job postings from CSV file.
    
    Args:
        filepath: Path to jobs.csv
        
    Returns:
        List of job dictionaries
    """
    jobs = []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                jobs.append({
                    "id": row.get("id"),
                    "job_title": row.get("job_title"),
                    "company": row.get("company"),
                    "description": row.get("description"),
                    "required_skills": row.get("required_skills"),
                    "experience_level": row.get("experience_level"),
                })
    except FileNotFoundError:
        print(f"Warning: {filepath} not found")
    
    return jobs
