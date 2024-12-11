import os
import json
import time
import requests
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
import openai
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm

nltk.download('punkt', quiet=True)

# Set your OpenAI API key if needed
os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.getenv("OPENAI_API_KEY")




def get_gene_id(symbol):
    """Look up GeneID based on gene symbol."""
    file_path = 'data/Homo_sapiens.gene_info'  # Update with your actual file path
    df = pd.read_csv(file_path, sep='\t')
    result = df[df['Symbol'] == symbol]
    if not result.empty:
        return result['GeneID'].values[0]
    else:
        return "Symbol not found"


def get_entity_info(entity):
    """
    Fetch information from NCBI. For SNP: returns HTML page from /snp/{entity}.
    For Gene: fetches full_report text from /gene/{gene_id}.
    """
    # Try SNP URL first
    url = f"https://www.ncbi.nlm.nih.gov/snp/{entity}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    elif response.status_code == 404:
        # Try gene info by looking up GeneID
        gene_id = get_gene_id(entity)
        if gene_id != "Symbol not found":
            url = f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}?report=full_report&format=text"
            response = requests.get(url)
            if response.status_code == 200:
                return response.text
            else:
                return f"Failed to retrieve the gene information for {entity}. Status code: {response.status_code}"
        else:
            return f"Entity {entity} not found."
    else:
        return f"Failed to retrieve the webpage for {entity}. Status code: {response.status_code}"


def extract_gene_sections_and_content(text):
    """
    Extract sections from a gene full report using regex.
    """
    section_pattern = re.compile(r'^(?P<section>[A-Z\s]+)\n-+$', re.MULTILINE)
    matches = list(section_pattern.finditer(text))
    sections_data = {}

    for i, match in enumerate(matches):
        section_name = match.group('section').strip()
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        section_content = text[start:end].strip()
        sections_data[section_name] = section_content

    return sections_data


# --- SNP Extraction Code ---
# Insert the code you provided for SNP parsing here
# For brevity, we show placeholders. Replace these with the actual functions from your snippet.
def extract_snp_data(html):
    """
    Extract SNP data sections from the provided HTML using BeautifulSoup.
    Returns a dictionary with keys representing SNP 'sections'.
    """
    soup = BeautifulSoup(html, 'html.parser')
    result = {}

    # From your snippet (shortened):
    # top_summary
    def extract_top_summary(soup):
        data = {}
        # rsID
        rs_match = soup.find('h2', id='refsnp_id')
        if rs_match:
            data['rsid'] = rs_match.get_text(strip=True)

        # Current Build and Released date
        accession = soup.find('div', class_='accession')
        if accession:
            # Format: 
            #  <p>Current Build <span>156</span></p>
            #  <p>Released <span>September 21, 2022</span></p>
            build_p = accession.find('p', text=re.compile('Current Build'))
            if build_p and build_p.find('span'):
                data['build'] = build_p.find('span').get_text(strip=True)
            release_p = accession.find('p', text=re.compile('Released'))
            if release_p and release_p.find('span'):
                data['release_date'] = release_p.find('span').get_text(strip=True)

        # summary-box has Organism, Position, Alleles, Variation Type, Frequency, Clinical significance, Gene:Consequence, Publications
        summary_box = soup.find('div', class_='summary-box')
        if summary_box:
            dt_dd = {}
            dts = summary_box.find_all('dt')
            for dt in dts:
                dd = dt.find_next('dd')
                if dd:
                    dt_text = dt.get_text(strip=True)
                    dd_text = dd.get_text(" ", strip=True)
                    dt_dd[dt_text] = dd_text

            # Organism
            if 'Organism' in dt_dd:
                data['organism'] = dt_dd['Organism']
            # Position
            if 'Position' in dt_dd:
                data['position'] = dt_dd['Position']
            # Alleles
            if 'Alleles' in dt_dd:
                data['alleles'] = dt_dd['Alleles']
            # Variation Type
            if 'Variation Type' in dt_dd:
                data['variation_type'] = dt_dd['Variation Type']
            # Frequency (brief from summary)
            if 'Frequency' in dt_dd:
                data['summary_frequency'] = dt_dd['Frequency']
            # Clinical Significance
            if 'Clinical Significance' in dt_dd:
                data['clinical_significance_summary'] = dt_dd['Clinical Significance']
            # Gene : Consequence
            if 'Gene : Consequence' in dt_dd:
                data['gene_consequence'] = dt_dd['Gene : Consequence']
            # Publications count
            if 'Publications' in dt_dd:
                # Something like "439 citations"
                data['publications_count'] = dt_dd['Publications']

        return data

    # frequency
    def extract_frequency_tab(soup):
        data = {}
        # frequency_tab is #frequency_tab
        freq_section = soup.find('div', id='frequency_tab')
        if not freq_section:
            return data

        # There's a large table with id dbsnp_freq_datatable
        dbsnp_table = freq_section.find('table', id='dbsnp_freq_datatable')
        if dbsnp_table:
            freq_entries = []
            headers = [th.get_text(strip=True) for th in dbsnp_table.find_all('th')]
            for row in dbsnp_table.find_all('tr')[1:]:
                cells = [c.get_text(strip=True) for c in row.find_all('td')]
                if cells and len(cells) == len(headers):
                    freq_entries.append(dict(zip(headers, cells)))
            data['dbsnp_frequency_table'] = freq_entries

        # There's also a popfreq_datatable (ALFA Allele Frequency table)
        popfreq_table = freq_section.find('table', id='popfreq_datatable')
        if popfreq_table:
            alfa_entries = []
            headers = [th.get_text(strip=True) for th in popfreq_table.find_all('th')]
            for row in popfreq_table.find_all('tr')[1:]:
                cells = [c.get_text(strip=True) for c in row.find_all('td')]
                if cells and len(cells) == len(headers):
                    alfa_entries.append(dict(zip(headers, cells)))
            data['alfa_popfreq_table'] = alfa_entries

        return data

    def extract_variant_details(soup):
        details = {}
        variant_section = soup.find('div', id='variant_details')
        if not variant_section:
            return details

        # Genomic Placements
        gp_table = variant_section.find('table', id='genomics_placements_table')
        if gp_table:
            gp_entries = []
            headers = [th.get_text(strip=True) for th in gp_table.find_all('th')]
            for row in gp_table.find_all('tr')[1:]:
                cells = [c.get_text(strip=True) for c in row.find_all('td')]
                if cells:
                    gp_entries.append(dict(zip(headers, cells)))
            details['genomic_placements'] = gp_entries

        # Transcript Annotations
        ta_table = variant_section.find('table', class_='trans_anno_allele_datatable')
        if ta_table:
            ta_entries = []
            headers = [th.get_text(strip=True) for th in ta_table.find('thead').find_all('th')]
            for row in ta_table.find('tbody').find_all('tr'):
                cells = [c.get_text(" ", strip=True) for c in row.find_all('td')]
                if cells and len(cells) == len(headers):
                    ta_entries.append(dict(zip(headers, cells)))
            details['transcript_annotation'] = ta_entries

        return details

    def extract_clinical_significance(soup):
        data = []
        cs_section = soup.find('div', id='clinical_significance')
        if cs_section:
            table = cs_section.find('table', id='clinical_significance_datatable')
            if table:
                headers = [th.get_text(strip=True) for th in table.find_all('th')]
                for row in table.find_all('tr')[1:]:
                    cells = [c.get_text(" ", strip=True) for c in row.find_all('td')]
                    if len(cells) == len(headers):
                        entry = dict(zip(headers, cells))
                        data.append(entry)
        return {"clinical_significance": data}

    def extract_hgvs(soup):
        data = []
        hgvs_section = soup.find('div', id='hgvs_tab')
        if hgvs_section:
            table = hgvs_section.find('table', id='alliases_alleles_datatable')
            if table:
                # First row is headers: "Placement", "C=", "T"
                headers = [th.get_text(strip=True) for th in table.find_all('th')]
                # The table might have multiple alleles. We'll parse each row
                for row in table.find_all('tr')[1:]:
                    cells = [c.get_text(strip=True) for c in row.find_all('td')]
                    if cells and len(cells) == len(headers):
                        data.append(dict(zip(headers, cells)))
        return {"hgvs": data}

    def extract_submissions(soup):
        data = []
        subs_section = soup.find('div', id='submissions')
        if subs_section:
            table = subs_section.find('table', id='submission_datatable')
            if table:
                headers = [th.get_text(strip=True) for th in table.find_all('th')]
                for row in table.find_all('tr')[1:]:
                    cells = [c.get_text(" ", strip=True) for c in row.find_all('td')]
                    if cells:
                        data.append(dict(zip(headers, cells)))
        return {"submissions": data}

    def extract_history(soup):
        data = {}
        hist_section = soup.find('div', id='history')
        if hist_section:
            # Associated IDs
            up_hist_datatable = hist_section.find('table', id='up_hist_datatable')
            if up_hist_datatable:
                rows = []
                headers = [th.get_text(strip=True) for th in up_hist_datatable.find_all('th')]
                for row in up_hist_datatable.find_all('tr')[1:]:
                    cells = [c.get_text(strip=True) for c in row.find_all('td')]
                    if cells:
                        rows.append(dict(zip(headers, cells)))
                data['history_associated_ids'] = rows

            # Observation present table
            obs_present_table = hist_section.find('table', id='obs_present_table')
            if obs_present_table:
                obs_rows = []
                headers = [th.get_text(strip=True) for th in obs_present_table.find_all('th')]
                for row in obs_present_table.find_all('tr')[1:]:
                    cells = [c.get_text(" ", strip=True) for c in row.find_all('td')]
                    if cells:
                        obs_rows.append(dict(zip(headers, cells)))
                data['history_observations'] = obs_rows
        return data

    def extract_publications(soup):
        data = {}
        pub_section = soup.find('div', id='publications')
        if pub_section:
            # count is already extracted
            # let's get the table with PMIDs:
            pub_table = pub_section.find('table', id='publication_datatable')
            if pub_table:
                headers = [th.get_text(strip=True) for th in pub_table.find_all('th')]
                publications = []
                for row in pub_table.find_all('tr')[1:]:
                    cells = [c.get_text(" ", strip=True) for c in row.find_all('td')]
                    if len(cells) == len(headers):
                        entry = dict(zip(headers, cells))
                        publications.append(entry)
                data['publications'] = publications

        return data

    # Now call them:
    result.update(extract_top_summary(soup))
    result.update(extract_frequency_tab(soup))
    result.update(extract_variant_details(soup))
    result.update(extract_clinical_significance(soup))
    result.update(extract_hgvs(soup))
    result.update(extract_submissions(soup))
    result.update(extract_history(soup))
    result.update(extract_publications(soup))

    return result
# --- End SNP Extraction Code ---


def load_section_descriptions(entity):
    """
    Load the appropriate section descriptions JSON depending on whether
    we have a gene or an SNP entity.
    """
    if entity.lower().startswith('rs'):
        # SNP
        fname = "data/section_descriptions_snp.json"
    else:
        # Gene
        fname = "data/section_descriptions_ncbi.json"

    with open(fname, "r", encoding="utf-8") as f:
        return json.load(f)


def select_relevant_sections(query, sections_data, section_descriptions):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that selects the most relevant sections from a record "
                "based on a user's query. You will receive a query and a set of sections, each "
                "with a description. You must return a JSON array of the keys of the sections "
                "that are most relevant. Limit output to a JSON list."
            )
        },
        {
            "role": "user",
            "content": (
                f"User query: {query}\n\n"
                "Sections and their descriptions (JSON):\n"
                f"{json.dumps(section_descriptions, indent=4)}\n\n"
                "Please select the keys of the sections most relevant to the query "
                "and return them as a JSON list."
            )
        }
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0
    )

    assistant_message = response.choices[0].message.content.strip()
    # print(f"Relevant sections for query ({query}): {assistant_message}")

    try:
        relevant_sections = json.loads(assistant_message)
        relevant_sections = [sec for sec in relevant_sections if sec in sections_data]
        if not relevant_sections:
            relevant_sections = list(sections_data.keys())
    except json.JSONDecodeError:
        relevant_sections = list(sections_data.keys())

    return relevant_sections


def create_sections_string(relevant_sections, sections_data, section_descriptions):
    parts = []
    for section in relevant_sections:
        desc = section_descriptions.get(section, "")
        content = sections_data.get(section, "")
        section_str = f"Section: {section}\nDescription: {desc}\nContent:\n{content}\n"
        parts.append(section_str)
    return "\n".join(parts)


def prune_text_based_on_query(text, query, max_tokens=5000, encoding_model="cl100k_base"):
    model = SentenceTransformer('all-mpnet-base-v2')
    sentences = sent_tokenize(text)
    
    sentence_embeddings = model.encode(sentences)
    query_embedding = model.encode([query])[0]

    similarities = np.dot(sentence_embeddings, query_embedding) / (
        np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    sentence_similarity_pairs = list(zip(sentences, similarities))
    sorted_sentences = sorted(sentence_similarity_pairs, key=lambda x: x[1], reverse=True)

    current_text = []
    token_count = 0
    encoding = tiktoken.get_encoding(encoding_model)

    for sentence, sim in sorted_sentences:
        sentence_tokens = len(encoding.encode(sentence))
        if token_count + sentence_tokens <= max_tokens:
            current_text.append(sentence)
            token_count += sentence_tokens
        else:
            break

    pruned_text = " ".join(current_text)
    return pruned_text


def num_tokens_from_messages(messages, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = 0

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if content is None:
                content = ""
        else:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")
            if content is None:
                content = ""

        total_tokens += len(encoding.encode(role + ": " + content))

    return total_tokens


def run_ncbi_query(query):
    tools = [
       {
            "type": "function",
            "function": {
                "name": "get_entity_info",
                "description": "Get information about a gene or SNP from NCBI.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "description": "The name of the gene or SNP, e.g., 'BRCA1' or 'rs7412'."
                        }
                    },
                    "required": ["entity"],
                    "additionalProperties": False
                }
            }
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can provide information about genes and SNPs from NCBI. "
                       "You have access to the tool get_entity_info to get information about an entity. You HAVE to call this function."
        },
        {"role": "user", "content": query}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        temperature=0,
        max_tokens=500
    )

    assistant_message = response.choices[0].message

    if assistant_message.tool_calls:
        tool_results = []
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_arguments = json.loads(tool_call.function.arguments)
            if function_name == 'get_entity_info':
                entity = function_arguments.get('entity')
                entity_info = get_entity_info(entity)
                # Save full text
                sanitized_query = re.sub(r'\W+', '_', query)
                os.makedirs(sanitized_query, exist_ok=True)
                with open(f"{sanitized_query}/{entity}-info-full.txt", "w", encoding="utf-8") as file:
                    file.write(entity_info)

                # Check if entity is SNP or gene
                if entity.lower().startswith('rs'):
                    # SNP extraction
                    sections_data = extract_snp_data(entity_info)
                else:
                    # Gene extraction
                    sections_data = extract_gene_sections_and_content(entity_info)

                # Save sections JSON
                sanitized_query = re.sub(r'\W+', '_', query)
                with open(f"{sanitized_query}/{entity}-sections.json", "w", encoding="utf-8") as json_file:
                    json.dump(sections_data, json_file, indent=4, ensure_ascii=False)

                # Load appropriate section descriptions
                section_descriptions = load_section_descriptions(entity)
                relevant_sections = select_relevant_sections(query, sections_data, section_descriptions)
                relevant_sections_string = create_sections_string(relevant_sections, sections_data, section_descriptions)

                tool_call_result_message = {
                    "role": "tool",
                    "name": "get_entity_info",
                    "content": relevant_sections_string,
                    "tool_call_id": tool_call.id
                }
                tool_results.append(tool_call_result_message)

        messages.append(assistant_message)
        messages.extend(tool_results)

        # Check token count and prune if needed
        total_tokens = num_tokens_from_messages(messages, model="gpt-4o")
        MAX_TOKENS_ALLOWED = 100000
        if total_tokens > MAX_TOKENS_ALLOWED:
            for tr in tool_results:
                if total_tokens <= MAX_TOKENS_ALLOWED:
                    break
                original_content = tr["content"]
                prune_target = 5000
                pruned_content = prune_text_based_on_query(original_content, query, max_tokens=prune_target)
                tr["content"] = pruned_content
                total_tokens = num_tokens_from_messages(messages, model="gpt-4o")

        final_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            max_tokens=500
        )

        return final_response.choices[0].message.content
    else:
        
        if isinstance(assistant_message, dict):
            content = assistant_message.get("content", "")
            if content is None:
                content = ""
        else:
            content = getattr(assistant_message, "content", "")
            if content is None:
                content = ""
        return content


if __name__ == "__main__":
    query = ""
    start_time = time.time()
    answer = run_ncbi_query(query)
    end_time = time.time()
    print("FINAL ANSWER:\n", answer)
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    