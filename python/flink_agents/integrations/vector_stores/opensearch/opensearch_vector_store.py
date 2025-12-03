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
import json
import os
from typing import Any

from alibabacloud_ha3engine_vector.client import Client
from alibabacloud_ha3engine_vector.models import Config, QueryRequest
from pydantic import Field

from flink_agents.api.vector_stores.vector_store import (
    BaseVectorStore,
    Document,
)

DEFAULT_TABLE = "flink_agents_opensearch_table"
DEFAULT_CONTENT_FIELD = "content"


class OpenSearchVectorStore(BaseVectorStore):
    """Alibaba Cloud OpenSearch vector store for semantic search.

    Visit https://help.aliyun.com/zh/open-search/vector-search-edition/ for OpenSearch
    documentation.

    This implementation uses Alibaba Cloud HA3 Engine Vector SDK to perform
    vector similarity search on OpenSearch instances.

    Attributes:
    ----------
    endpoint : str
        API endpoint for OpenSearch instance. Can be found in instance details page.
        Example: "http://ha-cn-xxxxx.public.ha.aliyuncs.com"
    access_user_name : str
        Username for OpenSearch API authentication.
    access_pass_word : str
        Password for OpenSearch API authentication.
    table_name : str
        Name of the OpenSearch table to query (default: flink_agents_opensearch_table).
    content_field : str
        Field name for document content in search results (default: "content").
    """

    # Connection configuration
    endpoint: str = Field(
        default_factory=lambda: os.getenv("OPENSEARCH_ENDPOINT", ""),
        description="API endpoint for OpenSearch instance.",
    )
    access_user_name: str = Field(
        default_factory=lambda: os.getenv("OPENSEARCH_USERNAME", ""),
        description="Username for OpenSearch API authentication.",
    )
    access_pass_word: str = Field(
        default_factory=lambda: os.getenv("OPENSEARCH_PASSWORD", ""),
        description="Password for OpenSearch API authentication.",
    )

    # Query configuration
    table_name: str = Field(
        default=DEFAULT_TABLE,
        description="Name of the OpenSearch table to query.",
    )
    content_field: str = Field(
        default=DEFAULT_CONTENT_FIELD,
        description="Field name for document content in search results.",
    )

    __client: Client | None = None

    def __init__(
        self,
        *,
        embedding_model: str,
        endpoint: str | None = None,
        access_user_name: str | None = None,
        access_pass_word: str | None = None,
        table_name: str = DEFAULT_TABLE,
        content_field: str = DEFAULT_CONTENT_FIELD,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenSearchVectorStore.

        Args:
            embedding_model: Name of the embedding model resource to use
            endpoint: API endpoint (overrides OPENSEARCH_ENDPOINT env var)
            access_user_name: Username (overrides OPENSEARCH_USERNAME env var)
            access_pass_word: Password (overrides OPENSEARCH_PASSWORD env var)
            table_name: Table name to query
            content_field: Field name for document content
            **kwargs: Additional parameters
        """
        # Use provided values or fall back to environment variables
        if endpoint is None:
            endpoint = os.getenv("OPENSEARCH_ENDPOINT", "")
        if access_user_name is None:
            access_user_name = os.getenv("OPENSEARCH_USERNAME", "")
        if access_pass_word is None:
            access_pass_word = os.getenv("OPENSEARCH_PASSWORD", "")

        super().__init__(
            embedding_model=embedding_model,
            endpoint=endpoint,
            access_user_name=access_user_name,
            access_pass_word=access_pass_word,
            table_name=table_name,
            content_field=content_field,
            **kwargs,
        )

    @property
    def client(self) -> Client:
        """Return OpenSearch client, creating it if necessary."""
        if self.__client is None:
            config = Config(
                endpoint=self.endpoint,
                access_user_name=self.access_user_name,
                access_pass_word=self.access_pass_word,
            )
            self.__client = Client(config)
        return self.__client

    @property
    def store_kwargs(self) -> dict[str, Any]:
        """Return OpenSearch-specific setup settings."""
        return {
            "table_name": self.table_name,
            "content_field": self.content_field,
        }

    def query_embedding(
        self, embedding: list[float], limit: int = 10, **kwargs: Any
    ) -> list[Document]:
        """Perform vector search using pre-computed embedding.

        Args:
            embedding: Pre-computed embedding vector for semantic search
            limit: Maximum number of results to return (default: 10)
            **kwargs: OpenSearch-specific parameters:
                - table_name: Table to query (overrides default)
                - content_field: Field name for content (overrides default)
                - output_fields: Additional fields to return (optional)
                - filter: Filter condition string (e.g., "age > 18")

        Returns:
            List of documents matching the search criteria
        """
        # Extract OpenSearch-specific parameters
        table_name = kwargs.get("table_name", self.table_name)
        content_field = kwargs.get("content_field", self.content_field)
        user_output_fields = kwargs.get("output_fields")  # Optional additional fields
        filter_condition = kwargs.get("filter")  # Optional filter string

        # Prepare output fields: always include content_field
        if user_output_fields is not None:
            # User specified fields, ensure content_field is included
            if content_field not in user_output_fields:
                output_fields = [content_field] + list(user_output_fields)
            else:
                output_fields = list(user_output_fields)
        else:
            # Default: only return content_field to minimize data transfer
            output_fields = [content_field]

        # Build query request
        request = QueryRequest(
            table_name=table_name,
            vector=embedding,
            include_vector=False,  # We don't need vectors in results
            output_fields=output_fields,  # Always specify to control returned data
            top_k=limit,
        )

        # Add optional filter
        if filter_condition is not None:
            request.filter = filter_condition

        # Perform query
        result = self.client.query(request)

        # Convert to Document objects
        documents = []
        if hasattr(result, "body") and result.body:
            # Parse JSON string to dict
            result_data = json.loads(result.body)
            if result_data.get("result"):
                for item in result_data["result"]:
                    # Extract document content from fields
                    doc_content = ""
                    metadata = {}

                    if item.get("fields"):
                        fields = item["fields"]
                        # Get content field
                        if content_field in fields:
                            doc_content = str(fields[content_field])

                        # Store all other fields as metadata
                        metadata = {
                            k: v for k, v in fields.items() if k != content_field
                        }

                    # Extract ID and score
                    doc_id = str(item.get("id")) if "id" in item else None
                    if "score" in item:
                        metadata["score"] = item["score"]

                    documents.append(
                        Document(
                            content=doc_content,
                            id=doc_id,
                            metadata=metadata,
                        )
                    )

        return documents
