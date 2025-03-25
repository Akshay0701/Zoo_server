import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter

def preprocess_research_interests(profiles):
    """
    Converts lists of research interests into text strings suitable for vectorization.
    
    Args:
        profiles (dict): Dictionary with researcher names as keys and research topics as values.
    
    Returns:
        tuple: List of profile names and list of research texts.
    """
    profile_names = list(profiles.keys())
    research_texts = [" ".join(interests) for interests in profiles.values()]
    return profile_names, research_texts

def vectorize_texts(texts):
    """
    Converts research interest texts into TF-IDF vectors.
    
    Args:
        texts (list): List of research interest strings.
    
    Returns:
        scipy.sparse.csr_matrix: TF-IDF vectorized representation of the texts.
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    return vectors

def determine_optimal_clusters(vectors, max_team_size=4):
    """
    Determines the optimal number of clusters using silhouette scores.
    
    Args:
        vectors (scipy.sparse.csr_matrix): Vectorized research interests.
        max_team_size (int): Maximum number of members per team (default is 4).
    
    Returns:
        int: Optimal number of clusters.
    """
    n_samples = vectors.shape[0]
    if n_samples < 2:
        return 1
    cluster_range = range(2, min(n_samples, 10))
    best_k = 2
    best_silhouette = -1

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        silhouette_avg = silhouette_score(vectors, labels)
        if silhouette_avg > best_silhouette:
            best_k = k
            best_silhouette = silhouette_avg

    return best_k

def form_teams(profiles, max_team_size=4):
    """
    Forms teams by clustering researchers based on their research interests.
    
    Args:
        profiles (dict): Dictionary with researcher names as keys and research topics as values.
        max_team_size (int): Maximum number of members per team (default is 4).
    
    Returns:
        list: List of teams, where each team is a list of member names.
    """
    profile_names, research_texts = preprocess_research_interests(profiles)
    vectors = vectorize_texts(research_texts)

    n_samples = len(profile_names)
    if n_samples == 0:
        return []
    elif n_samples == 1:
        return [[profile_names[0]]]

    num_teams = determine_optimal_clusters(vectors, max_team_size)
    kmeans = KMeans(n_clusters=num_teams, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)

    clusters = {i: [] for i in range(num_teams)}
    for i, profile in enumerate(profile_names):
        clusters[labels[i]].append(profile)

    final_teams = []
    for members in clusters.values():
        for i in range(0, len(members), max_team_size):
            final_teams.append(members[i:i + max_team_size])

    return final_teams

def extract_main_research_areas(profiles, teams):
    """
    Extracts the main research areas for each team and lists member fields.
    
    Args:
        profiles (dict): Dictionary with researcher names as keys and research topics as values.
        teams (list): List of teams, where each team is a list of member names.
    
    Returns:
        dict: Dictionary with team IDs as keys and research area details as values.
    """
    team_research_areas = {}
    for team_id, members in enumerate(teams):
        research_fields = []
        member_fields = {}
        for member in members:
            member_fields[member] = profiles[member]
            research_fields.extend(profiles[member])
        # Use Counter to get the top 5 most common fields
        most_common_fields = [field for field, _ in Counter(research_fields).most_common(5)]
        team_research_areas[team_id] = {"team_fields": most_common_fields, "member_fields": member_fields}
    return team_research_areas