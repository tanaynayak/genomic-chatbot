# genomic-insight-extractor

A Python tool for querying and extracting detailed SNP and gene information from NCBI, with section parsing and relevance-based summarization.

## Features

- Fetch SNP and gene details from NCBI.
- Parse and extract sections using BeautifulSoup.
- Use GPT models to identify relevant sections based on queries.
- Summarize and prune text for efficient output.

## Requirements

Ensure you have Python 3.8 or later installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Set up your OpenAI API Key**:  
   Update your OpenAI API key in the `agent.py` file or set it as an environment variable.

   ```python
   os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
   ```

2. **Update the Query**:  
   Replace the placeholder query with your desired query in the `agent.py` file:

   ```python
   query = "<YOUR_QUERY>"
   ```

3. **Run the Agent**:  
   Execute the script to fetch and process data:

   ```bash
   python agent.py
   ```

4. **Output**:  
   The results will be saved to the appropriate directory and displayed in the terminal.

## Notes

- Ensure you have access to the internet for API calls and data fetching.
- Modify file paths and configurations as needed for your environment.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
