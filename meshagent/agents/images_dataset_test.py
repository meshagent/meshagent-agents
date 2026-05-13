from typing import cast

import pyarrow as pa
import pytest

from meshagent.api import RoomClient
from meshagent.agents.images_dataset import ImageDatasetClient, ImagesDataset


def test_images_dataset_marks_data_column_as_image_content() -> None:
    schema = ImagesDataset(room=cast(RoomClient, None))._schema()

    assert pa.types.is_large_binary(schema.field("data").type)
    assert schema.field("data").metadata == {b"content-type": b"image/*"}


class _FakeDatasets:
    def __init__(self) -> None:
        self.schema = pa.schema(
            [
                pa.field("id", pa.string(), nullable=False),
                pa.field("data", pa.binary(), nullable=False),
            ]
        )
        self.create_calls = 0
        self.added_columns: dict[str, pa.Field] | None = None
        self.index_calls = 0
        self.search_calls: list[dict] = []
        self.search_rows: list[dict] = []

    async def inspect(self, *, table: str) -> pa.Schema:
        assert table == "images"
        return self.schema

    async def create_table_with_schema(
        self,
        *,
        name: str,
        schema: pa.Schema,
        mode: str,
    ) -> None:
        del name
        del schema
        del mode
        self.create_calls += 1
        raise AssertionError("existing images table should not be recreated")

    async def add_columns(
        self,
        *,
        table: str,
        new_columns: dict[str, pa.Field],
    ) -> None:
        assert table == "images"
        self.added_columns = new_columns
        self.schema = pa.schema([*self.schema, *new_columns.values()])

    async def create_index(self, *, table: str, config) -> None:
        del config
        assert table == "images"
        self.index_calls += 1

    async def search(
        self,
        *,
        table: str,
        namespace=None,
        where=None,
        limit=None,
        select=None,
    ) -> pa.Table:
        self.search_calls.append(
            {
                "table": table,
                "namespace": namespace,
                "where": where,
                "limit": limit,
                "select": select,
            }
        )
        return pa.Table.from_pylist(self.search_rows)


class _FakeRoom:
    def __init__(self) -> None:
        self.datasets = _FakeDatasets()


@pytest.mark.asyncio
async def test_images_dataset_uses_existing_table_with_different_schema() -> None:
    room = _FakeRoom()
    dataset = ImagesDataset(room=cast(RoomClient, room))

    await dataset._ensure_ready()

    assert room.datasets.create_calls == 0
    assert room.datasets.added_columns is not None
    assert "mime_type" in room.datasets.added_columns
    assert "created_at" in room.datasets.added_columns
    assert "created_by" in room.datasets.added_columns
    assert "annotations" in room.datasets.added_columns
    assert room.datasets.index_calls == 1


def test_image_dataset_client_parses_dataset_image_uri() -> None:
    assert ImageDatasetClient.dataset_uri_reference("dataset://images?id=image-1") == (
        "images",
        None,
        "image-1",
    )
    assert ImageDatasetClient.dataset_uri_reference(
        "dataset://agents/demo/images?id=image-2"
    ) == ("images", ["agents", "demo"], "image-2")


@pytest.mark.asyncio
async def test_image_dataset_client_reads_dataset_uri_record() -> None:
    room = _FakeRoom()
    room.datasets.search_rows = [
        {"id": "image-1", "data": bytearray(b"image-bytes"), "mime_type": "image/png"}
    ]
    client = ImageDatasetClient(room=cast(RoomClient, room))

    record = await client.read_record_from_uri(
        "dataset://agents/demo/images?id=image-1"
    )

    assert record is not None
    assert record.data == b"image-bytes"
    assert record.mime_type == "image/png"
    assert room.datasets.search_calls == [
        {
            "table": "images",
            "namespace": ["agents", "demo"],
            "where": {"id": "image-1"},
            "limit": 1,
            "select": ["data", "mime_type"],
        }
    ]
