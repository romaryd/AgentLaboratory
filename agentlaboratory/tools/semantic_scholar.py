from semanticscholar import SemanticScholar


class SemanticScholarSearch:
    def __init__(self):
        self.sch_engine = SemanticScholar(retry=False)

    def find_papers_by_str(self, query, N=10):
        paper_sums = list()
        results = self.sch_engine.search_paper(
            query, limit=N, min_citation_count=3, open_access_pdf=True
        )
        for _i in range(len(results)):
            paper_sum = f"Title: {results[_i].title}\n"
            paper_sum += f"Abstract: {results[_i].abstract}\n"
            paper_sum += f"Citations: {results[_i].citationCount}\n"
            paper_sum += f"Release Date: year {results[_i].publicationDate.year}, month {results[_i].publicationDate.month}, day {results[_i].publicationDate.day}\n"
            paper_sum += f"Venue: {results[_i].venue}\n"
            paper_sum += f"Paper ID: {results[_i].externalIds['DOI']}\n"
            paper_sums.append(paper_sum)
        return paper_sums

    def retrieve_full_paper_text(self, query):
        pass
