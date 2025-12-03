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
from typing import Any

from alibabacloud_ha3engine_vector.client import Client
from alibabacloud_ha3engine_vector.models import Config, QueryRequest
from pydantic import Field

from flink_agents.api.vector_stores.vector_store import (
    BaseVectorStore,
    Document,
)

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
    username : str
        Username for OpenSearch API authentication.
    password : str
        Password for OpenSearch API authentication.
    table_name : str
        Name of the OpenSearch table to query (default: flink_agents_opensearch_table).
    content_field : str
        Field name for document content in search results (default: "content").
    """

    # Connection configuration
    endpoint: str = Field(
        description="API endpoint for OpenSearch instance.",
    )
    username: str = Field(
        description="Username for OpenSearch API authentication.",
    )
    password: str = Field(
        description="Password for OpenSearch API authentication.",
    )

    # Query configuration
    table_name: str = Field(
        description="Name of the OpenSearch table to query.",
    )
    content_field: str = Field(
        description="Field name for document content in search results.",
    )

    __client: Client | None = None

    def __init__(
        self,
        *,
        embedding_model: str,
        endpoint: str,
        username: str,
        password: str,
        table_name: str,
        content_field: str,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenSearchVectorStore.

        Args:
            embedding_model: Name of the embedding model resource to use
            endpoint: API endpoint
            username: Username
            password: Password
            table_name: Table name to query
            content_field: Field name for document content
            **kwargs: Additional parameters
        """
        super().__init__(
            embedding_model=embedding_model,
            endpoint=endpoint,
            username=username,
            password=password,
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
                access_user_name=self.username,
                access_pass_word=self.password,
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
