
import time
import arxiv
import os
from pypdf import PdfReader

class ArxivSearch:
    def __init__(self):
        # Construct the default API client.
        self.sch_engine = arxiv.Client()

    def find_papers_by_str(self, query, N=20):
        search = arxiv.Search(
            query="abs:" + query,
            max_results=N,
            sort_by=arxiv.SortCriterion.Relevance)

        paper_sums = list()
        # `results` is a generator; you can iterate over its elements one by one...
        for r in self.sch_engine.results(search):
            paperid = r.pdf_url.split("/")[-1]
            pubdate = str(r.published).split(" ")[0]
            paper_sum = f"Title: {r.title}\n"
            paper_sum += f"Summary: {r.summary}\n"
            paper_sum += f"Publication Date: {pubdate}\n"
            paper_sum += f"Categories: {' '.join(r.categories)}\n"
            paper_sum += f"arXiv paper ID: {paperid}\n"
            paper_sums.append(paper_sum)
        time.sleep(2.0)
        return "\n".join(paper_sums)

    def retrieve_full_paper_text(self, query):
        pdf_text = str()
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[query])))
        # Download the PDF to the PWD with a custom filename.
        paper.download_pdf(filename="downloaded-paper.pdf")
        # creating a pdf reader object
        reader = PdfReader('downloaded-paper.pdf')
        # Iterate over all the pages
        for page_number, page in enumerate(reader.pages, start=1):
            # Extract text from the page
            try:
                text = page.extract_text()
            except Exception as e:
                os.remove("downloaded-paper.pdf")
                time.sleep(2.0)
                return "EXTRACTION FAILED"

            # Do something with the text (e.g., print it)
            pdf_text += f"--- Page {page_number} ---"
            pdf_text += text
            pdf_text += "\n"
        os.remove("downloaded-paper.pdf")
        time.sleep(2.0)
        return pdf_text