from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, TIMESTAMP, update, Float, desc, asc, func, select
from sqlalchemy.dialects.postgresql import ARRAY
from databases import Database
import shutil
import pdfplumber
import json
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
import os
from PyPDF2 import PdfReader
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods, you can specify ["GET", "POST", etc.] if needed
    allow_headers=["*"],  # Allows all headers, you can specify specific headers if needed
)

api_key = ""
azure_endpoint = ""
api_version = ""

llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-3-large",
    deployment_name="",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

DATABASE_URL = ""

database = Database(DATABASE_URL)
metadata = MetaData()

documents = Table(
    "documents",
    metadata,
    Column("doc_id", String(255), primary_key=True),
    Column("account_name", String(255)),
    Column("firm_id", String(255)),
    Column("firm_name", String(255)),
    Column("fund_name", String(255)),
    Column("doc_type", String(255)),
    Column("file_name", String(255)),
    Column("file_type", String(255)),
    Column("num_pages", Integer),
    Column("date_time", TIMESTAMP),
    Column("status", String(255)),
    Column("confidence", String(255)),
)

# Configuration table schema
configurations = Table(
    "configurations",
    metadata,
    Column("doc_type", String(255), primary_key=True),
    Column("fields", ARRAY(String)),
)

# New documents table schema (same as documents table)
new_documents = Table(
    "new_documents",
    metadata,
    Column("doc_id", String(255), primary_key=True),
    Column("account_name", String(255)),
    Column("firm_id", String(255)),
    Column("firm_name", String(255)),
    Column("fund_name", String(255)),
    Column("doc_type", String(255)),
    Column("file_name", String(255)),
    Column("file_type", String(255)),
    Column("num_pages", Integer),
    Column("date_time", TIMESTAMP),
    Column("status", String(255)),
    Column("confidence", String(255)),
)

# Extracted table schema
extracted = Table(
    "extracted",
    metadata,
    Column("doc_id", String(255), primary_key=True),
    Column("kv_pairs", ARRAY(String)),
    Column("keys", ARRAY(String)),
)

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/json")
async def json_out(file_name: str):
    file_name = file_name.split('.')[0]  # Remove any extension
    file_path = os.path.join("/home/removed/filehandler/", f"{file_name}.pdf")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # Extract text from the PDF file
    with pdfplumber.open(file_path) as pdf:
        documents_text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
    
    # Convert the extracted text into a Document object
    document = Document(text=documents_text)
    
    # Now process the document as before
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model
    )
    index = VectorStoreIndex.from_documents([document], service_context=service_context)
    query_engine = index.as_query_engine()
    response = query_engine.query("""
        Extract all important details from the given document, including text and tables, and present them in a structured JSON format. Include details such as investor name, fund name, document type (categorized as 'Statement', 'Capital Call', 'Distribution', or 'New Document Type'), and other key-value pairs, along with their respective confidence scores as percentages. Ensure the JSON structure is correct and free from delimiter issues. The output should be in the following format:

        {
            "Investor_Name": {"value": "Extracted value", "confidence": "Confidence score as percentage"},
            "Fund_Name": {"value": "Extracted value", "confidence": "Confidence score as percentage"},
            "Document_Type": {"value": "Statement/Capital Call/Distribution/New Document Type", "confidence": "Confidence score as percentage"},
            ...
        }
        No other response is required, only the structured JSON output as specified above.
    """)
    
    logger.info(f"Raw response: {response.response}")

    try:
        json_output = json.loads(response.response)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON decode error: {e.msg}")

    # Save the JSON output to a file
    json_file_path = os.path.join("/home/removed/filehandler/", f"{file_name}.json")
    with open(json_file_path, "w") as json_file:
        json.dump(json_output, json_file, indent=4)

    # Update the database with the status
    query = update(documents).where(documents.c.file_name == f"{file_name}.pdf").values(status='Extracted')
    await database.execute(query)

    return {"json_output": json_output}

class DocumentUploadRequest(BaseModel):
    doc_id: str
    status: str
    confidence: float

@app.post("/doc_details")
async def doc_details(file_name: str):
    file_name = file_name.split('.')[0]  # Remove any extension
    file_path = os.path.join("/home/removed/filehandler/", f"{file_name}.pdf")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    file_type = 'application/pdf'
    num_pages = None

    try:
        with open(file_path, 'rb') as f:
            pdf = PdfReader(f)
            num_pages = len(pdf.pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read PDF file: {str(e)}")

    # Extract text from the PDF file
    with pdfplumber.open(file_path) as pdf:
        documents_text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())

    # Convert the extracted text into a Document object
    document_obj = Document(text=documents_text)

    # Process the document to extract details
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model
    )
    index = VectorStoreIndex.from_documents([document_obj], service_context=service_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(f"""
        Extract all important details from the given document, including text and tables, and present them in a structured JSON format. Include details such as:
        - Investor Name (as 'Entity_Name'): The name of the investor or entity mentioned in the document
        - Fund Name (as 'Fund_Name'): The name of the fund mentioned in the document
        - Firm Name (as 'Firm_Name'): The name of the firm mentioned in the document
        - Document Type (as 'Document_Type'): Categorize it as 'Statement', 'Capital Call', 'Distribution', or a relevant type based on the document's content
        - Number of Pages (as 'Num_Pages'): The total number of pages in the document
        - Confidence (as 'Confidence'): The confidence score as a percentage

        The JSON structure should be as follows:
        {{
            "Entity_Name": "Extracted value",
            "Fund_Name": "Extracted value",
            "Firm_Name": "Extracted value",
            "Document_Type": "Statement/Capital Call/Distribution",
            "Num_Pages": {num_pages},
            "Confidence": "Confidence score as percentage"
        }}
        Ensure the JSON structure is correct and free from delimiter issues. No other response is required.
    """)

    logger.info(f"Raw response: {response.response}")

    try:
        json_output = json.loads(response.response)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON decode error: {e.msg}")

    # Determine if the document type exists in the configuration table
    query = select(configurations).where(configurations.c.doc_type == json_output.get("Document_Type"))
    config_result = await database.fetch_one(query)

    current_datetime = datetime.now()

    if config_result:
        # Insert into the documents table
        query = documents.insert().values(
            doc_id=file_name,
            account_name=json_output.get("Entity_Name"),
            firm_id=None,  # Adjust as needed
            firm_name=json_output.get("Firm_Name"),
            fund_name=json_output.get("Fund_Name"),
            doc_type=json_output.get("Document_Type"),
            file_name=f"{file_name}.pdf",
            file_type=file_type,
            num_pages=num_pages,
            date_time=current_datetime,  # Add current datetime
            status='Received',  # Initial status
            confidence=json_output.get("Confidence"),
        )
        await database.execute(query)
    else:
        # Insert into the new_documents table
        query = new_documents.insert().values(
            doc_id=file_name,
            account_name=json_output.get("Entity_Name"),
            firm_id=None,  # Adjust as needed
            firm_name=json_output.get("Firm_Name"),
            fund_name=json_output.get("Fund_Name"),
            doc_type=json_output.get("Document_Type"),
            file_name=f"{file_name}.pdf",
            file_type=file_type,
            num_pages=num_pages,
            date_time=current_datetime,  # Add current datetime
            status='Received',  # Initial status
            confidence=json_output.get("Confidence"),
        )
        await database.execute(query)

    return {"doc_details": json_output}


@app.post("/store_key_value_pairs")
async def store_key_value_pairs(doc_id: str, json_file_name: str):
    file_path = os.path.join("/home/removed/filehandler/", f"{json_file_name}.json")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="JSON file not found")
    
    with open(file_path, "r") as file:
        json_data = json.load(file)
    
    kv_pairs = [f"{key}: {value}" for key, value in json_data.items()]
    keys = list(json_data.keys())
    
    query = extracted.insert().values(
        doc_id=doc_id,
        kv_pairs=kv_pairs,
        keys=keys
    )
    await database.execute(query)
    
    return {"message": "Key-value pairs and keys stored successfully"}

class Configuration(BaseModel):
    doc_type: str
    fields: List[str]

@app.post("/add_configuration")
async def add_configuration(configuration: Configuration):
    # Check if the document type already exists in the configurations table
    query = select(configurations).where(configurations.c.doc_type == configuration.doc_type)
    config_result = await database.fetch_one(query)

    if config_result:
        raise HTTPException(status_code=400, detail="Document type already exists in the configuration table")

    # Insert the new document type and fields into the configurations table
    query = configurations.insert().values(
        doc_type=configuration.doc_type,
        fields=configuration.fields
    )
    await database.execute(query)

    return {"message": "Configuration added successfully"}

class UpdateConfigurationRequest(BaseModel):
    doc_type: str
    add_fields: List[str] = []
    remove_fields: List[str] = []

@app.post("/update_configuration")
async def update_configuration(request: UpdateConfigurationRequest):
    # Check if the document type exists in the configurations table
    query = select(configurations).where(configurations.c.doc_type == request.doc_type)
    config_result = await database.fetch_one(query)

    if not config_result:
        raise HTTPException(status_code=404, detail="Document type not found in the configuration table")

    existing_fields = set(config_result["fields"])
    add_fields = set(request.add_fields)
    remove_fields = set(request.remove_fields)

    # Add only fields that are not already present
    fields_to_add = add_fields - existing_fields
    # Remove only fields that are currently present
    fields_to_remove = remove_fields & existing_fields

    # Calculate the new set of fields
    new_fields = existing_fields.union(fields_to_add).difference(fields_to_remove)

    # Update the document type with the new fields
    query = update(configurations).where(configurations.c.doc_type == request.doc_type).values(fields=list(new_fields))
    await database.execute(query)

    return {"message": "Configuration updated successfully"}

@app.post("/transfer_document")
async def transfer_document(doc_id: str):
    # Fetch the document from new_documents that matches the given document ID
    select_query = select(new_documents).where(new_documents.c.doc_id == doc_id)
    new_doc = await database.fetch_one(select_query)

    if not new_doc:
        raise HTTPException(status_code=404, detail="No document found for the given document ID")

    # Insert the fetched document into the documents table
    insert_query = documents.insert().values(
        doc_id=new_doc["doc_id"],
        account_name=new_doc["account_name"],
        firm_id=new_doc["firm_id"],
        firm_name=new_doc["firm_name"],
        fund_name=new_doc["fund_name"],
        doc_type=new_doc["doc_type"],
        file_name=new_doc["file_name"],
        file_type=new_doc["file_type"],
        num_pages=new_doc["num_pages"],
        date_time=new_doc["date_time"],
        status=new_doc["status"],
        confidence=new_doc["confidence"],
    )
    await database.execute(insert_query)

    # Delete the transferred document from the new_documents table
    delete_query = new_documents.delete().where(new_documents.c.doc_id == doc_id)
    await database.execute(delete_query)

    return {"message": "Document transferred successfully"}

@app.get("/get_configuration/{doc_type}")
async def get_configuration(doc_type: str):
    # Retrieve the configuration for the given document type
    query = select(configurations).where(configurations.c.doc_type == doc_type)
    config_result = await database.fetch_one(query)

    return config_result

DROP_FOLDER = "/home/removed/Processes/drop/"
PROCESSING_FOLDER = "/home/removed/filehandler/"
COMPLETED_FOLDER = "/home/removed/Processes/completed/"
INCOMPLETE_FOLDER = "/home/removed/Processes/incomplete/"

def move_file(src, dst):
    try:
        shutil.move(str(src), str(dst))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to move file: {e}")

@app.get("/check_drop_folder/")
async def check_drop_folder():
    if not os.path.exists(DROP_FOLDER):
        raise HTTPException(status_code=404, detail="Drop folder not found")

    files = list(os.listdir(DROP_FOLDER))
    file_count = len(files)
    file_names = files

    if not files:
        return {"message": "No files found in drop folder", "file_count": file_count, "files": file_names}

    for file in files:
        move_file(os.path.join(DROP_FOLDER, file), os.path.join(PROCESSING_FOLDER, file))

    return {"message": "Files moved to processing folder", "file_count": file_count, "files": file_names}

@app.post("/move_to_completed/{file_name}")
async def move_to_completed(file_name: str):
    file_path = os.path.join(PROCESSING_FOLDER, f"{file_name}.pdf")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found in processing folder")

    move_file(file_path, os.path.join(COMPLETED_FOLDER, f"{file_name}.pdf"))
    return {"message": f"{file_name} moved to completed folder"}

@app.post("/move_to_incomplete/{file_name}")
async def move_to_incomplete(file_name: str):
    file_path = os.path.join(PROCESSING_FOLDER, f"{file_name}.pdf")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found in processing folder")

    move_file(file_path, os.path.join(INCOMPLETE_FOLDER, f"{file_name}.pdf"))
    return {"message": f"{file_name} moved to incomplete folder"}

# Database filter endpoint
class DocumentFilter(BaseModel):
    doc_type: Optional[str] = None
    fund_name: Optional[str] = None
    account_name: Optional[str] = None
    firm_name: Optional[str] = None

class PaginationParams(BaseModel):
    page: Optional[int] = 1
    order: Optional[str] = "desc"
    order_by: Optional[str] = "date_time"

class DocumentQuery(BaseModel):
    filter: Optional[DocumentFilter] = None
    pagination: Optional[PaginationParams] = None

@app.post("/documents/filter")
async def filter_documents(
    query_params: DocumentQuery = Body(...),
    page_size: int = Query(10, ge=1, le=100)
):
    filter = query_params.filter
    pagination = query_params.pagination

    selected_columns = []
    if filter:
        if filter.fund_name:
            selected_columns.append(documents.c.fund_name)
        if filter.account_name:
            selected_columns.append(documents.c.account_name)
        if filter.firm_name:
            selected_columns.append(documents.c.firm_name)
        if filter.doc_type:
            selected_columns.append(documents.c.doc_type)
    if not selected_columns:
        selected_columns = [
            documents.c.doc_id,
            documents.c.account_name,
            documents.c.firm_id,
            documents.c.firm_name,
            documents.c.fund_name,
            documents.c.doc_type,
            documents.c.file_name,
            documents.c.file_type,
            documents.c.num_pages,
            documents.c.date_time,
            documents.c.status,
            documents.c.confidence,
        ]

    query = select(*selected_columns)

    if filter:
        if filter.fund_name:
            query = query.where(documents.c.fund_name == filter.fund_name)
        if filter.account_name:
            query = query.where(documents.c.account_name == filter.account_name)
        if filter.firm_name:
            query = query.where(documents.c.firm_name == filter.firm_name)
        if filter.doc_type:
            query = query.where(documents.c.doc_type == filter.doc_type)

    # Always prioritize the latest documents first
    query = query.order_by(desc(documents.c.date_time))

    # If pagination payload is provided
    if pagination:
        # Secondary ordering by specified column and direction
        order_column = getattr(documents.c, pagination.order_by, None)
        if order_column is not None:
            if pagination.order.lower() == "asc":
                query = query.order_by(asc(order_column))
            else:
                query = query.order_by(desc(order_column))

    # Create the subquery to get the total count
    total_documents_query = select(func.count()).select_from(query.alias())
    total_documents = await database.fetch_val(total_documents_query)

    page = pagination.page if pagination and pagination.page else 1
    query = query.limit(page_size).offset((page - 1) * page_size)
    results = await database.fetch_all(query)

    return {
        "total_documents": total_documents,
        "page": page,
        "page_size": page_size,
        "documents": results,
    }

@app.get("/dropdown/fund_names")
async def get_fund_names():
    query = select(documents.c.fund_name).distinct()
    results = await database.fetch_all(query)
    fund_names = [result["fund_name"] for result in results]
    return {"fund_names": fund_names}

@app.get("/dropdown/firm_names")
async def get_firm_names():
    query = select(documents.c.firm_name).distinct()
    results = await database.fetch_all(query)
    firm_names = [result["firm_name"] for result in results]
    return {"firm_names": firm_names}

@app.get("/dropdown/account_names")
async def get_account_names():
    query = select(documents.c.account_name).distinct()
    results = await database.fetch_all(query)
    account_names = [result["account_name"] for result in results]
    return {"account_names": account_names}

@app.get("/dropdown/document_types")
async def get_document_types():
    query = select(documents.c.doc_type).distinct()
    results = await database.fetch_all(query)
    document_types = [result["doc_type"] for result in results]
    return {"document_types": document_types}

# New documents filter endpoint
@app.post("/new_documents/filter")
async def filter_new_documents(
    query_params: DocumentQuery = Body(...),
    page_size: int = Query(10, ge=1, le=100)
):
    filter = query_params.filter
    pagination = query_params.pagination

    selected_columns = []
    if filter:
        if filter.fund_name:
            selected_columns.append(new_documents.c.fund_name)
        if filter.account_name:
            selected_columns.append(new_documents.c.account_name)
        if filter.firm_name:
            selected_columns.append(new_documents.c.firm_name)
        if filter.doc_type:
            selected_columns.append(new_documents.c.doc_type)
    if not selected_columns:
        selected_columns = [
            new_documents.c.doc_id,
            new_documents.c.account_name,
            new_documents.c.firm_id,
            new_documents.c.firm_name,
            new_documents.c.fund_name,
            new_documents.c.doc_type,
            new_documents.c.file_name,
            new_documents.c.file_type,
            new_documents.c.num_pages,
            new_documents.c.date_time,
            new_documents.c.status,
            new_documents.c.confidence,
        ]

    query = select(*selected_columns)

    if filter:
        if filter.fund_name:
            query = query.where(new_documents.c.fund_name == filter.fund_name)
        if filter.account_name:
            query = query.where(new_documents.c.account_name == filter.account_name)
        if filter.firm_name:
            query = query.where(new_documents.c.firm_name == filter.firm_name)
        if filter.doc_type:
            query = query.where(new_documents.c.doc_type == filter.doc_type)

    # Always prioritize the latest documents first
    query = query.order_by(desc(new_documents.c.date_time))

    # If pagination payload is provided
    if pagination:
        # Secondary ordering by specified column and direction
        order_column = getattr(new_documents.c, pagination.order_by, None)
        if order_column is not None:
            if pagination.order.lower() == "asc":
                query = query.order_by(asc(order_column))
            else:
                query = query.order_by(desc(order_column))

    # Create the subquery to get the total count
    total_documents_query = select(func.count()).select_from(query.alias())
    total_documents = await database.fetch_val(total_documents_query)

    page = pagination.page if pagination and pagination.page else 1
    query = query.limit(page_size).offset((page - 1) * page_size)
    results = await database.fetch_all(query)
    
    return {
        "total_documents": total_documents,
        "page": page,
        "page_size": page_size,
        "documents": results,
    }

# Endpoint to retrieve all field names in a JSON file
class ChecklistFieldsResponse(BaseModel):
    fields: List[str]

@app.get("/checklist_fields", response_model=ChecklistFieldsResponse)
async def checklist_fields(json_file_name: str):
    file_path = os.path.join("/home/removed/filehandler/", f"{json_file_name}.json")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="JSON file not found")
    
    with open(file_path, "r") as file:
        json_data = json.load(file)
    
    field_names = list(json_data.keys())
    
    return ChecklistFieldsResponse(fields=field_names)

class FilterFieldsRequest(BaseModel):
    json_file_name: str
    fields: Optional[List[str]] = None

@app.post("/filter_json_fields")
async def filter_json_fields(request: FilterFieldsRequest):
    file_path = os.path.join("/home/removed/filehandler/", f"{request.json_file_name}.json")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="JSON file not found")
    
    with open(file_path, "r") as file:
        json_data = json.load(file)
    
    if request.fields:
        filtered_data = {key: json_data[key] for key in request.fields if key in json_data}
        if not filtered_data:
            raise HTTPException(status_code=404, detail="None of the specified fields were found in the JSON file")
        return filtered_data
    
    return json_data

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/json_test")
async def json_out(file_name: str):
    file_name = file_name.split('.')[0]  # Remove any extension
    file_path = os.path.join("/home/removed/filehandler/", f"{file_name}.pdf")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # Extract text from the PDF file
    with pdfplumber.open(file_path) as pdf:
        documents_text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
    
    # Convert the extracted text into a Document object
    document = Document(text=documents_text)
    
    # Now process the document as before
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model
    )
    index = VectorStoreIndex.from_documents([document], service_context=service_context)
    query_engine = index.as_query_engine()
    response = query_engine.query("""
        Extract all important details from the given document, including text and tables, and present them in a structured JSON format. 
        Ensure the JSON structure is correct and free from delimiter issues. 
        No other response is required, only the structured JSON output as specified above.
    """)
    
    logger.info(f"Raw response: {response.response}")

    try:
        json_output = json.loads(response.response)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON decode error: {e.msg}")

    # # Save the JSON output to a file
    # json_file_path = os.path.join("/home/removed/filehandler/", f"{file_name}.json")
    # with open(json_file_path, "w") as json_file:
    #     json.dump(json_output, json_file, indent=4)

    # # Update the database with the status
    # query = update(documents).where(documents.c.file_name == f"{file_name}.pdf").values(status='Extracted')
    # await database.execute(query)

    return {"json_output": json_output}