# Unstructured Financial Data Extraction API

## Overview

This FastAPI application provides a complete solution for processing, extracting, and managing financial documents such as statements, capital calls, and distributions. The system leverages Azure OpenAI's LLM capabilities to extract structured information from unstructured PDF documents and provides a comprehensive API for document management.

## Features

- **Document Processing Pipeline**: Process PDF documents through a complete workflow from upload to storage
- **Intelligent Data Extraction**: Use Azure OpenAI to extract key information from financial documents
- **Document Classification**: Automatically classify documents into predefined or new document types
- **Configurable Document Types**: Add, update, and manage document type configurations
- **File Management**: Complete file handling across different processing stages
- **Database Integration**: Store and retrieve document metadata and extracted information
- **Filtering Capabilities**: Advanced filtering and pagination for document retrieval
- **CORS Support**: Support for cross-origin requests

## Technical Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL (accessed via SQLAlchemy and Databases library)
- **AI Models**:
  - Azure OpenAI GPT-3.5 Turbo 16K
  - Azure OpenAI Text Embedding 3 Large
- **PDF Processing**: pdfplumber, PyPDF2
- **Vector Search**: LlamaIndex

## Setup

### Prerequisites

- Python 3.8+
- PostgreSQL database
- Azure OpenAI account and API keys

### Environment Configuration

Update the following variables in the code with your own credentials:

```python
api_key = "your_azure_openai_api_key"
azure_endpoint = "your_azure_endpoint"
api_version = "your_api_version"

llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="your_deployment_name",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-3-large",
    deployment_name="your_embedding_deployment_name",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

DATABASE_URL = "postgresql://username:password@host:port/database"
```

### Directory Structure

Ensure the following directory structure exists:

```
/home/removed/Processes/drop/         # Initial drop folder for documents
/home/removed/filehandler/            # Processing folder
/home/removed/Processes/completed/    # Folder for completed documents
/home/removed/Processes/incomplete/   # Folder for incomplete documents
```

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install fastapi uvicorn sqlalchemy databases pydantic pdfplumber PyPDF2 llama-index python-multipart
   ```
3. Start the server:
   ```
   uvicorn main:app --reload
   ```

## Database Schema

The application uses the following tables:

1. **documents**: Stores information about processed documents
2. **new_documents**: Stores information about documents with new document types
3. **configurations**: Stores document type configurations
4. **extracted**: Stores extracted key-value pairs from documents

## API Endpoints

### Document Processing

- `POST /json`: Process a PDF file and extract structured JSON data
- `POST /doc_details`: Extract document details from a PDF file
- `POST /store_key_value_pairs`: Store extracted key-value pairs
- `GET /check_drop_folder/`: Check and process files in the drop folder
- `POST /move_to_completed/{file_name}`: Move a file to the completed folder
- `POST /move_to_incomplete/{file_name}`: Move a file to the incomplete folder

### Configuration Management

- `POST /add_configuration`: Add a new document type configuration
- `POST /update_configuration`: Update an existing document type configuration
- `GET /get_configuration/{doc_type}`: Get configuration for a document type
- `POST /transfer_document`: Transfer a document from new_documents to documents

### Document Filtering and Retrieval

- `POST /documents/filter`: Filter and paginate documents
- `POST /new_documents/filter`: Filter and paginate new documents
- `GET /dropdown/fund_names`: Get all distinct fund names
- `GET /dropdown/firm_names`: Get all distinct firm names
- `GET /dropdown/account_names`: Get all distinct account names
- `GET /dropdown/document_types`: Get all distinct document types

### Field Handling

- `GET /checklist_fields`: Get all field names from a JSON file
- `POST /filter_json_fields`: Filter fields from a JSON file

## Document Processing Workflow

1. PDF files are placed in the drop folder
2. Files are moved to the processing folder
3. Document details are extracted and stored in the database
4. Document type is determined:
   - If the document type exists in configurations, the document is added to the documents table
   - If the document type is new, the document is added to the new_documents table
5. Structured data is extracted from the document and stored
6. Files are moved to either the completed or incomplete folder

## Usage Examples

### Process a new document

```python
import requests

response = requests.get("http://localhost:8000/check_drop_folder/")
print(response.json())  # Shows files found and moved to processing

file_name = "example_document"  # Without extension
response = requests.post(f"http://localhost:8000/doc_details?file_name={file_name}")
print(response.json())  # Shows extracted document details

response = requests.post(f"http://localhost:8000/json?file_name={file_name}")
print(response.json())  # Shows extracted structured data

response = requests.post(
    "http://localhost:8000/store_key_value_pairs",
    params={"doc_id": file_name, "json_file_name": file_name}
)
print(response.json())  # Confirms key-value pairs stored

response = requests.post(f"http://localhost:8000/move_to_completed/{file_name}")
print(response.json())  # Confirms file moved to completed folder
```

### Add a new document type configuration

```python
import requests
import json

config = {
    "doc_type": "Distribution Notice",
    "fields": ["Entity_Name", "Fund_Name", "Distribution_Amount", "Distribution_Date"]
}

response = requests.post(
    "http://localhost:8000/add_configuration",
    json=config
)
print(response.json())  # Confirms configuration added
```

### Filter documents

```python
import requests
import json

filter_params = {
    "filter": {
        "doc_type": "Statement",
        "fund_name": "Example Fund"
    },
    "pagination": {
        "page": 1,
        "order": "desc",
        "order_by": "date_time"
    }
}

response = requests.post(
    "http://localhost:8000/documents/filter?page_size=5",
    json=filter_params
)
print(response.json())  # Shows filtered documents
```
## Acknowledgments

We would like to express our deepest gratitude to all those who contributed to this project. Their valuable input, support, and expertise made this project possible.

### Contributors

- Bhavya Chanana - [Github](https://github.com/bhavya-chanana) - [Linkedin](https://www.linkedin.com/in/bhavya-chanana)
- Govind Anjan - [Github](https://github.com/trimax420) - [Linkedin](https://www.linkedin.com/in/anjancs/)
