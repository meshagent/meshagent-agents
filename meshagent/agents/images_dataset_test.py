from typing import cast

import pyarrow as pa
import pytest

from meshagent.api import RoomClient
from meshagent.agents.images_dataset import ImagesDataset


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
