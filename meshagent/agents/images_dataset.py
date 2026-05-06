import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import pyarrow as pa

from meshagent.api import RoomClient
from meshagent.api.room_server_client import DatasetIndexConfig, DatasetStruct

logger = logging.getLogger("images_dataset")
_IMAGE_SAVE_MAX_RETRIES = 6
_IMAGE_SAVE_RETRY_BASE_DELAY_SECONDS = 0.2


@dataclass(frozen=True)
class SavedImage:
    id: str
    mime_type: str
    created_at: str
    created_by: str
    annotations: dict[str, str]


class ImagesDataset:
    TABLE_NAME = "images"
    _METADATA_COLUMNS = ["id", "mime_type", "created_at", "created_by", "annotations"]

    def __init__(self, room: RoomClient):
        self._room = room
        self._ready = False
        self._lock = asyncio.Lock()

    def _schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.string(), nullable=False),
                pa.field(
                    "data",
                    pa.large_binary(),
                    nullable=False,
                    metadata={"content-type": "image/*"},
                ),
                pa.field("mime_type", pa.string(), nullable=False),
                pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("created_by", pa.string(), nullable=False),
                pa.field(
                    "annotations",
                    pa.list_(
                        pa.struct(
                            [
                                pa.field("key", pa.string(), nullable=False),
                                pa.field("value", pa.string()),
                            ]
                        )
                    ),
                ),
            ]
        )

    @staticmethod
    def _normalize_annotations(
        annotations: Optional[dict[str, str]],
    ) -> dict[str, str]:
        if annotations is None:
            return {}

        normalized: dict[str, str] = {}
        for key, value in annotations.items():
            normalized[str(key)] = str(value)
        return normalized

    @staticmethod
    def _encode_annotations(annotations: dict[str, str]) -> list[DatasetStruct]:
        return [
            DatasetStruct({"key": key, "value": value})
            for key, value in annotations.items()
        ]

    @staticmethod
    def _decode_annotations(value: Any) -> dict[str, str]:
        if value is None:
            return {}

        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}

        if not isinstance(value, list):
            return {}

        decoded: dict[str, str] = {}
        for item in value:
            if isinstance(item, DatasetStruct):
                item = item.fields
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            if not isinstance(key, str):
                continue
            item_value = item.get("value")
            decoded[key] = "" if item_value is None else str(item_value)
        return decoded

    @staticmethod
    def _to_saved_image(record: dict[str, Any]) -> SavedImage:
        return SavedImage(
            id=str(record.get("id") or ""),
            mime_type=str(record.get("mime_type") or "application/octet-stream"),
            created_at=str(record.get("created_at") or ""),
            created_by=str(record.get("created_by") or ""),
            annotations=ImagesDataset._decode_annotations(record.get("annotations")),
        )

    @staticmethod
    def _is_commit_conflict_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "commit conflict" in message
            and "concurrent transaction" in message
            and "lance" in message
        )

    async def _ensure_ready(self) -> None:
        if self._ready:
            return

        async with self._lock:
            if self._ready:
                return

            schema = self._schema()
            await self._room.datasets.create_table_with_schema(
                name=self.TABLE_NAME,
                schema=schema,
                mode="create_if_not_exists",
            )

            existing_schema = await self._room.datasets.inspect(table=self.TABLE_NAME)
            existing_names = set(existing_schema.names)
            missing_columns = {
                field.name: field
                for field in schema
                if field.name not in existing_names
            }
            if len(missing_columns) > 0:
                await self._room.datasets.add_columns(
                    table=self.TABLE_NAME,
                    new_columns=missing_columns,
                )

            try:
                await self._room.datasets.create_index(
                    table=self.TABLE_NAME,
                    config=DatasetIndexConfig(column="id", index_type="BTREE"),
                )
            except Exception:
                logger.debug(
                    "unable to create images.id scalar index; continuing",
                    exc_info=True,
                )

            self._ready = True

    async def save(
        self,
        *,
        data: bytes,
        mime_type: str,
        created_by: str,
        annotations: Optional[dict[str, str]] = None,
        image_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> SavedImage:
        await self._ensure_ready()

        normalized_annotations = self._normalize_annotations(annotations=annotations)
        normalized_id = image_id or str(uuid.uuid4())
        normalized_created_at = created_at or datetime.now(
            timezone.utc
        ).isoformat().replace("+00:00", "Z")

        record = {
            "id": normalized_id,
            "data": data,
            "mime_type": mime_type,
            "created_at": normalized_created_at,
            "created_by": created_by,
            "annotations": self._encode_annotations(normalized_annotations),
        }

        for attempt in range(_IMAGE_SAVE_MAX_RETRIES):
            try:
                await self._room.datasets.insert(
                    table=self.TABLE_NAME,
                    records=[record],
                )
                break
            except Exception as ex:
                if not self._is_commit_conflict_error(ex):
                    raise
                if attempt >= _IMAGE_SAVE_MAX_RETRIES - 1:
                    raise

                retry_delay = _IMAGE_SAVE_RETRY_BASE_DELAY_SECONDS * (2**attempt)
                logger.warning(
                    "images insert hit commit conflict; retrying (%d/%d) in %.1fs",
                    attempt + 1,
                    _IMAGE_SAVE_MAX_RETRIES,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)

        return SavedImage(
            id=normalized_id,
            mime_type=mime_type,
            created_at=normalized_created_at,
            created_by=created_by,
            annotations=normalized_annotations,
        )

    async def read(self, *, image_id: str) -> Optional[SavedImage]:
        await self._ensure_ready()

        rows = await self._room.datasets.search(
            table=self.TABLE_NAME,
            where={"id": image_id},
            limit=1,
            select=self._METADATA_COLUMNS,
        )
        rows = rows.to_pylist()
        if len(rows) == 0:
            return None

        return self._to_saved_image(rows[0])

    async def search(
        self,
        *,
        where: Optional[str | dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> list[SavedImage]:
        await self._ensure_ready()

        rows = await self._room.datasets.search(
            table=self.TABLE_NAME,
            where=where,
            limit=limit,
            select=self._METADATA_COLUMNS,
        )
        rows = rows.to_pylist()
        return [self._to_saved_image(row) for row in rows]

    async def read_data(self, *, image_id: str) -> Optional[bytes]:
        await self._ensure_ready()

        rows = await self._room.datasets.search(
            table=self.TABLE_NAME,
            where={"id": image_id},
            limit=1,
            select=["data"],
        )
        rows = rows.to_pylist()
        if len(rows) == 0:
            return None

        data = rows[0].get("data")
        if isinstance(data, bytearray):
            return bytes(data)
        if isinstance(data, bytes):
            return data
        return None

    async def optimize(self) -> None:
        await self._ensure_ready()
        await self._room.datasets.optimize(table=self.TABLE_NAME)
