import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class NSFProjectChain:
    def __init__(self, excel_path='AwardsMRSEC.xls'):
        self.chat = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("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"),
            model_name="llama-3.3-70b-versatile"
        )
        self.excel_path = excel_path
        self.existing_abstracts = self._load_existing_abstracts()
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def _load_existing_abstracts(self):
        try:
            df = pd.read_excel(self.excel_path)
            if 'Abstract' not in df.columns:
                raise ValueError("Excel file must contain 'abstract' column")
            return df['Abstract'].tolist()
        except FileNotFoundError:
            print(f"Warning: Excel file {self.excel_path} not found.")
            return []
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return []

    def _calculate_similarity(self, new_abstracts):
        if not self.existing_abstracts:
            return {abstract: 0 for abstract in new_abstracts}
        
        # Combine existing and new abstracts
        all_texts = self.existing_abstracts + new_abstracts
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Split matrix into existing and new
        existing_matrix = tfidf_matrix[:len(self.existing_abstracts)]
        new_matrix = tfidf_matrix[len(self.existing_abstracts):]
        
        # Calculate cosine similarity
        similarity_scores = cosine_similarity(new_matrix, existing_matrix)
        
        # Get maximum similarity for each new abstract
        max_similarities = similarity_scores.max(axis=1)
        return dict(zip(new_abstracts, max_similarities))

    def generate_project_proposals(self, team):
        team_info = (
            f"Members: {', '.join(team.get('members', []))}. "
            f"Research Areas: {', '.join(team.get('team_research_areas', []))}. "
            f"Member Details: {team.get('member_fields', {})}"
        )

        # Create a prompt template to request 5 NSF project abstract recommendations.
        prompt_template = """
        ### TEAM MEMBER PROFILE AND RESEARCH INTERESTS:
        {team_info}
        
        ### INSTRUCTION:
        Generate 5 detailed project abstract recommendations for an NSF project proposal based on the team's profiles and research interests.
        Each project abstract should be at least 3 sentences long, outlining the project scope, objectives, and potential impact.
        
        Return the output in valid JSON format with a single key "project_proposals" mapping to a list of 5 abstract strings.
        Only return valid JSON without extra text.
        """

        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.chat

        response = chain.invoke(input={"team_info": team_info})
        try:
            json_parser = JsonOutputParser()
            result = json_parser.parse(response.content)
        except OutputParserException:
            raise OutputParserException("Unable to parse project proposals from Groq response.")
        
        proposals = result.get("project_proposals", [])
        
        if proposals:
            similarity_scores = self._calculate_similarity(proposals)
            # Sort by least similar first (ascending order)
            sorted_proposals = sorted(proposals, key=lambda x: similarity_scores[x])
            return sorted_proposals
        
        return proposals