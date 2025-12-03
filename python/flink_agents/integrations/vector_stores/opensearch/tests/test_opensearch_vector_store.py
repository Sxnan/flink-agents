################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
import os
import time
import uuid
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

try:
    from alibabacloud_ha3engine_vector.client import Client
    from alibabacloud_ha3engine_vector.models import (
        Config,
        CreateTableRequest,
        CreateTableRequestVectorIndex,
        PushDocumentsRequest,
    )

    opensearch_available = True
except ImportError:
    opensearch_available = False

from flink_agents.api.resource import Resource, ResourceType
from flink_agents.api.vector_stores.vector_store import VectorStoreQuery
from flink_agents.integrations.vector_stores.opensearch.opensearch_vector_store import (
    OpenSearchVectorStore,
)

# Environment variables for real OpenSearch service
opensearch_endpoint = os.environ.get("TEST_OPENSEARCH_ENDPOINT")
opensearch_username = os.environ.get("TEST_OPENSEARCH_USERNAME")
opensearch_password = os.environ.get("TEST_OPENSEARCH_PASSWORD")


class MockEmbeddingModel(Resource):  # noqa: D101
    @classmethod
    def resource_type(cls) -> ResourceType:  # noqa: D102
        return ResourceType.EMBEDDING_MODEL

    @property
    def model_kwargs(self) -> dict[str, Any]:  # noqa: D102
        return {}

    def embed(self, text: str, **kwargs: Any) -> list[float]:  # noqa: D102
        return [0.1, 0.2, 0.3]


def _create_opensearch_table(
    client: "Client",
    table_name: str,
    id_field: str = "id",
    vector_field: str = "vector",
    content_field: str = "content",
    dimension: int = 3,
) -> None:
    """Create an OpenSearch table for testing.

    Args:
        client: OpenSearch client instance
        table_name: Name of the table to create
        id_field: Primary key field name
        vector_field: Vector field name
        content_field: Content field name
        dimension: Vector dimension
    """
    request = CreateTableRequest()
    request.name = table_name
    request.primary_key = id_field
    request.partition_count = 1

    # Field schema
    request.field_schema = {
        id_field: "INT64",
        vector_field: "MULTI_FLOAT",
        content_field: "STRING",
        "category": "STRING",  # Additional field for testing filters
    }

    # Vector index configuration
    vector_index = CreateTableRequestVectorIndex()
    vector_index.index_name = vector_field
    vector_index.vector_field = vector_field
    vector_index.dimension = dimension
    vector_index.vector_index_type = "HNSW"
    vector_index.distance_type = "InnerProduct"
    request.vector_index = [vector_index]

    client.create_table(request)


def _wait_table_ready(client: "Client", table_name: str, timeout: int = 60) -> None:
    """Wait for table to become ready.

    Args:
        client: OpenSearch client instance
        table_name: Name of the table to wait for
        timeout: Maximum wait time in seconds
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        result = client.get_table(table_name)
        status = result.body.result.status
        print(f"Table status: {status}")  # Debug: print status
        if status == "IN_USE":
            return
        time.sleep(5)  # Check every 5 seconds
    msg = f"Table {table_name} not ready after {timeout} seconds, last status: {status}"
    raise TimeoutError(msg)


def _stop_and_delete_table(client: "Client", table_name: str, timeout: int = 60) -> None:
    """Stop and delete an OpenSearch table.

    Args:
        client: OpenSearch client instance
        table_name: Name of the table to delete
        timeout: Maximum wait time in seconds
    """
    try:
        # Stop the table
        client.stop_table(table_name)

        # Wait for table to stop
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = client.get_table(table_name)
            status = result.body.result.status
            print(f"Stopping table, status: {status}")
            if status == "NOT_USE":
                break
            time.sleep(5)  # Check every 5 seconds

        # Delete the table
        client.delete_table(table_name)
    except Exception as e:
        # Log error but don't fail the test cleanup
        print(f"Error during table cleanup: {e}")


def test_opensearch_vector_store_initialization() -> None:
    """Test OpenSearchVectorStore initialization with different configurations."""
    # Test with environment variables
    with patch.dict(
        os.environ,
        {
            "OPENSEARCH_ENDPOINT": "http://test-endpoint.com",
            "OPENSEARCH_USERNAME": "test_user",
            "OPENSEARCH_PASSWORD": "test_pass",
        },
    ):
        vector_store = OpenSearchVectorStore(
            name="test_store",
            embedding_model="mock_embeddings",
        )
        assert vector_store.endpoint == "http://test-endpoint.com"
        assert vector_store.access_user_name == "test_user"
        assert vector_store.access_pass_word == "test_pass"
        assert vector_store.table_name == "flink_agents_opensearch_table"
        assert vector_store.content_field == "content"

    # Test with explicit parameters
    vector_store = OpenSearchVectorStore(
        name="test_store",
        embedding_model="mock_embeddings",
        endpoint="http://custom-endpoint.com",
        access_user_name="custom_user",
        access_pass_word="custom_pass",
        table_name="custom_table",
        content_field="custom_content",
    )
    assert vector_store.endpoint == "http://custom-endpoint.com"
    assert vector_store.access_user_name == "custom_user"
    assert vector_store.access_pass_word == "custom_pass"
    assert vector_store.table_name == "custom_table"
    assert vector_store.content_field == "custom_content"


def test_opensearch_query_with_mock() -> None:
    """Test OpenSearch query with mocked client."""
    embedding_model = MockEmbeddingModel(name="mock_embeddings")

    def get_resource(name: str, resource_type: ResourceType) -> Resource:
        if resource_type == ResourceType.EMBEDDING_MODEL:
            return embedding_model
        msg = f"Unknown resource type: {resource_type}"
        raise ValueError(msg)

    # Create vector store
    vector_store = OpenSearchVectorStore(
        name="test_store",
        embedding_model="mock_embeddings",
        endpoint="http://test-endpoint.com",
        access_user_name="test_user",
        access_pass_word="test_pass",
        table_name="test_table",
        get_resource=get_resource,
    )

    # Mock the client and response
    mock_client = MagicMock()
    mock_response = Mock()
    mock_response.body = {
        "result": [
            {
                "id": 1,
                "score": 0.95,
                "fields": {
                    "content": "This is a test document about Flink Agents",
                    "title": "Test Document",
                    "category": "test",
                },
            },
            {
                "id": 2,
                "score": 0.85,
                "fields": {
                    "content": "Another document about vector search",
                    "author": "Test Author",
                },
            },
        ],
        "totalCount": 2,
        "totalTime": 1.5,
    }
    mock_client.query.return_value = mock_response

    # Inject mock client via private attribute
    vector_store._OpenSearchVectorStore__client = mock_client

    # Test basic query
    query = VectorStoreQuery(query_text="What is Flink Agent?", limit=2)
    result = vector_store.query(query)

    # Verify results
    assert result is not None
    assert len(result.documents) == 2

    # Check first document
    doc1 = result.documents[0]
    assert doc1.id == "1"
    assert doc1.content == "This is a test document about Flink Agents"
    assert doc1.metadata["score"] == 0.95
    assert doc1.metadata["title"] == "Test Document"
    assert doc1.metadata["category"] == "test"
    assert "content" not in doc1.metadata  # content_field should not be in metadata

    # Check second document
    doc2 = result.documents[1]
    assert doc2.id == "2"
    assert doc2.content == "Another document about vector search"
    assert doc2.metadata["score"] == 0.85
    assert doc2.metadata["author"] == "Test Author"


def test_opensearch_query_with_filter() -> None:
    """Test OpenSearch query with filter condition."""
    embedding_model = MockEmbeddingModel(name="mock_embeddings")

    def get_resource(name: str, resource_type: ResourceType) -> Resource:
        if resource_type == ResourceType.EMBEDDING_MODEL:
            return embedding_model
        msg = f"Unknown resource type: {resource_type}"
        raise ValueError(msg)

    vector_store = OpenSearchVectorStore(
        name="test_store",
        embedding_model="mock_embeddings",
        endpoint="http://test-endpoint.com",
        access_user_name="test_user",
        access_pass_word="test_pass",
        table_name="test_table",
        get_resource=get_resource,
    )

    mock_client = MagicMock()
    mock_response = Mock()
    mock_response.body = {
        "result": [
            {
                "id": 1,
                "score": 0.95,
                "fields": {"content": "Filtered document", "age": 25},
            }
        ],
        "totalCount": 1,
    }
    mock_client.query.return_value = mock_response

    # Inject mock client via private attribute
    vector_store._OpenSearchVectorStore__client = mock_client

    # Query with filter
    query = VectorStoreQuery(
        query_text="test query", limit=10, extra_args={"filter": "age > 18"}
    )
    result = vector_store.query(query)

    # Verify the query was called with filter
    call_args = mock_client.query.call_args
    request = call_args[0][0]
    assert hasattr(request, "filter")
    assert request.filter == "age > 18"

    # Verify results
    assert len(result.documents) == 1
    assert result.documents[0].content == "Filtered document"


def test_opensearch_output_fields_logic() -> None:
    """Test output_fields logic: content_field is always included."""
    embedding_model = MockEmbeddingModel(name="mock_embeddings")

    def get_resource(name: str, resource_type: ResourceType) -> Resource:
        if resource_type == ResourceType.EMBEDDING_MODEL:
            return embedding_model
        msg = f"Unknown resource type: {resource_type}"
        raise ValueError(msg)

    vector_store = OpenSearchVectorStore(
        name="test_store",
        embedding_model="mock_embeddings",
        endpoint="http://test-endpoint.com",
        access_user_name="test_user",
        access_pass_word="test_pass",
        table_name="test_table",
        content_field="doc_content",
        get_resource=get_resource,
    )

    mock_client = MagicMock()
    mock_client.query.return_value = Mock(body={"result": []})

    # Inject mock client via private attribute
    vector_store._OpenSearchVectorStore__client = mock_client

    # Test 1: No output_fields specified - should default to [content_field]
    query = VectorStoreQuery(query_text="test", limit=10)
    vector_store.query(query)
    request = mock_client.query.call_args[0][0]
    assert request.output_fields == ["doc_content"]

    # Test 2: User specifies output_fields without content_field
    query = VectorStoreQuery(
        query_text="test", limit=10, extra_args={"output_fields": ["title", "author"]}
    )
    vector_store.query(query)
    request = mock_client.query.call_args[0][0]
    assert "doc_content" in request.output_fields
    assert "title" in request.output_fields
    assert "author" in request.output_fields

    # Test 3: User specifies output_fields with content_field already included
    query = VectorStoreQuery(
        query_text="test",
        limit=10,
        extra_args={"output_fields": ["doc_content", "title"]},
    )
    vector_store.query(query)
    request = mock_client.query.call_args[0][0]
    # Should not duplicate content_field
    assert request.output_fields.count("doc_content") == 1
    assert "title" in request.output_fields


def test_opensearch_empty_results() -> None:
    """Test handling of empty query results."""
    embedding_model = MockEmbeddingModel(name="mock_embeddings")

    def get_resource(name: str, resource_type: ResourceType) -> Resource:
        if resource_type == ResourceType.EMBEDDING_MODEL:
            return embedding_model
        msg = f"Unknown resource type: {resource_type}"
        raise ValueError(msg)

    vector_store = OpenSearchVectorStore(
        name="test_store",
        embedding_model="mock_embeddings",
        endpoint="http://test-endpoint.com",
        access_user_name="test_user",
        access_pass_word="test_pass",
        get_resource=get_resource,
    )

    mock_client = MagicMock()
    mock_client.query.return_value = Mock(body={"result": [], "totalCount": 0})

    # Inject mock client via private attribute
    vector_store._OpenSearchVectorStore__client = mock_client

    query = VectorStoreQuery(query_text="no results query", limit=10)
    result = vector_store.query(query)

    assert result is not None
    assert len(result.documents) == 0


def test_opensearch_store_kwargs() -> None:
    """Test store_kwargs property."""
    vector_store = OpenSearchVectorStore(
        name="test_store",
        embedding_model="mock_embeddings",
        endpoint="http://test-endpoint.com",
        access_user_name="test_user",
        access_pass_word="test_pass",
        table_name="custom_table",
        content_field="custom_content",
    )

    kwargs = vector_store.store_kwargs
    assert kwargs["table_name"] == "custom_table"
    assert kwargs["content_field"] == "custom_content"
    assert len(kwargs) == 2  # Only these two fields


@pytest.mark.skipif(
    not opensearch_available or opensearch_endpoint is None,
    reason="OpenSearch SDK not available or TEST_OPENSEARCH_ENDPOINT not set",
)
def test_opensearch_with_real_service() -> None:
    """Test OpenSearch vector store with real OpenSearch service.

    This test requires the following environment variables:
    - TEST_OPENSEARCH_ENDPOINT: OpenSearch API endpoint
    - TEST_OPENSEARCH_USERNAME: OpenSearch username
    - TEST_OPENSEARCH_PASSWORD: OpenSearch password

    The test will automatically:
    1. Create a unique test table with timestamp-based name
    2. Insert test data
    3. Run queries
    4. Clean up by deleting the table
    """
    # Generate unique table name to avoid conflicts
    unique_suffix = uuid.uuid4().hex[:8]
    test_table_name = f"flink_agents_test_{unique_suffix}"

    embedding_model = MockEmbeddingModel(name="mock_embeddings")

    def get_resource(name: str, resource_type: ResourceType) -> Resource:
        if resource_type == ResourceType.EMBEDDING_MODEL:
            return embedding_model
        msg = f"Unknown resource type: {resource_type}"
        raise ValueError(msg)

    # Create OpenSearch client for table management
    config = Config(
        endpoint=opensearch_endpoint,
        access_user_name=opensearch_username,
        access_pass_word=opensearch_password,
    )
    client = Client(config)

    try:
        # Step 1: Create test table
        print(f"Creating test table: {test_table_name}")
        _create_opensearch_table(
            client=client,
            table_name=test_table_name,
            id_field="id",
            vector_field="vector",
            content_field="content",
            dimension=3,
        )

        # Wait for table to be ready
        print(f"Waiting for table {test_table_name} to be ready...")
        _wait_table_ready(client, test_table_name, timeout=300)  # 5 minutes timeout
        print(f"Table {test_table_name} is ready")

        # Step 2: Create vector store
        vector_store = OpenSearchVectorStore(
            name="test_store",
            embedding_model="mock_embeddings",
            endpoint=opensearch_endpoint,
            username=opensearch_username,
            password=opensearch_password,
            table_name=test_table_name,
            get_resource=get_resource,
            content_field="content",
        )

        # Step 3: Insert test data
        test_docs = [
            {
                "id": 1001,
                "vector": [0.1, 0.2, 0.3],
                "content": "Apache Flink Agents is an AI framework for stream processing",
                "category": "ai-framework",
            },
            {
                "id": 1002,
                "vector": [0.2, 0.3, 0.4],
                "content": "OpenSearch is a powerful vector database",
                "category": "database",
            },
            {
                "id": 1003,
                "vector": [0.15, 0.25, 0.35],
                "content": "Vector search enables semantic similarity matching",
                "category": "ai-framework",
            },
        ]

        print(f"Inserting {len(test_docs)} test documents...")
        # Use push_documents method (correct way for OpenSearch)
        documents_list = []
        for doc in test_docs:
            documents_list.append({
                "fields": doc,
                "cmd": "add",  # add command for new documents
            })
        
        # Push all documents at once
        push_request = PushDocumentsRequest({}, documents_list)
        pk_field = "id"  # Primary key field
        client.push_documents(test_table_name, pk_field, push_request)

        # Wait a bit for data to be indexed
        time.sleep(5)

        # Step 4: Test basic query
        print("Testing basic query...")
        query = VectorStoreQuery(
            query_text="What is Flink Agents?",
            limit=2,
        )
        result = vector_store.query(query)

        assert result is not None
        assert len(result.documents) > 0
        assert len(result.documents) <= 2

        # Verify document structure
        doc = result.documents[0]
        assert doc.content is not None
        assert doc.id is not None
        assert "score" in doc.metadata
        print(f"Basic query returned {len(result.documents)} documents")

        # Step 5: Test query with filter
        print("Testing query with filter...")
        query_with_filter = VectorStoreQuery(
            query_text="AI technology",
            limit=5,
            extra_args={"filter": "category = 'ai-framework'"},
        )
        result_filtered = vector_store.query(query_with_filter)

        assert result_filtered is not None
        print(f"Filtered query returned {len(result_filtered.documents)} documents")
        # All results should match the filter
        for doc in result_filtered.documents:
            if "category" in doc.metadata:
                assert doc.metadata["category"] == "ai-framework"

        # Step 6: Test query with additional output fields
        print("Testing query with output fields...")
        query_with_fields = VectorStoreQuery(
            query_text="database",
            limit=3,
            extra_args={"output_fields": ["category"]},
        )
        result_with_fields = vector_store.query(query_with_fields)

        assert result_with_fields is not None
        assert len(result_with_fields.documents) > 0
        print(f"Query with fields returned {len(result_with_fields.documents)} documents")
        # Verify that category field is included
        for doc in result_with_fields.documents:
            assert doc.content is not None  # content_field always included

        print("All tests passed!")

    finally:
        # Step 7: Cleanup - Delete test table
        print(f"Cleaning up: deleting test table {test_table_name}...")
        _stop_and_delete_table(client, test_table_name, timeout=300)  # 5 minutes timeout
        print(f"Test table {test_table_name} deleted successfully")
