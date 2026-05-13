import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

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


@dataclass(frozen=True)
class ImageDatasetRecord:
    data: bytes
    mime_type: str


class ImageDatasetClient:
    TABLE_NAME = "images"

    def __init__(self, room: RoomClient):
        self._room = room

    @staticmethod
    def dataset_uri_reference(
        uri: str | None,
    ) -> tuple[str, list[str] | None, str] | None:
        if not isinstance(uri, str):
            return None
        parsed = urlparse(uri.strip())
        if parsed.scheme != "dataset":
            return None
        values = parse_qs(parsed.query).get("id")
        if not values:
            return None
        image_id = values[0].strip()
        if image_id == "":
            return None

        path_parts = [
            part.strip()
            for part in ([parsed.netloc] if parsed.netloc.strip() != "" else [])
            + list(parsed.path.split("/"))
            if part.strip() != ""
        ]
        if len(path_parts) == 0:
            return None
        table_name = path_parts[-1]
        namespace = path_parts[:-1] or None
        return table_name, namespace, image_id

    async def read_record(
        self,
        *,
        image_id: str,
        table: str | None = None,
        namespace: list[str] | None = None,
        fallback_mime_type: str | None = None,
    ) -> ImageDatasetRecord | None:
        normalized_image_id = image_id.strip()
        if normalized_image_id == "":
            return None
        rows = await self._room.datasets.search(
            table=table or self.TABLE_NAME,
            namespace=namespace,
            where={"id": normalized_image_id},
            limit=1,
            select=["data", "mime_type"],
        )
        values = rows.to_pylist()
        if len(values) == 0:
            return None

        row = values[0]
        data = row.get("data")
        if isinstance(data, bytearray):
            data = bytes(data)
        if not isinstance(data, bytes):
            return None

        mime_type = row.get("mime_type")
        if not isinstance(mime_type, str) or mime_type.strip() == "":
            mime_type = fallback_mime_type or "image/png"
        return ImageDatasetRecord(data=data, mime_type=mime_type.strip())

    async def read_record_from_uri(
        self,
        uri: str | None,
        *,
        fallback_mime_type: str | None = None,
    ) -> ImageDatasetRecord | None:
        reference = self.dataset_uri_reference(uri)
        if reference is None:
            return None
        table, namespace, image_id = reference
        return await self.read_record(
            image_id=image_id,
            table=table,
            namespace=namespace,
            fallback_mime_type=fallback_mime_type,
        )


class ImagesDataset(ImageDatasetClient):
    _METADATA_COLUMNS = ["id", "mime_type", "created_at", "created_by", "annotations"]

    def __init__(self, room: RoomClient):
        super().__init__(room)
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
            try:
                existing_schema = await self._room.datasets.inspect(
                    table=self.TABLE_NAME
                )
            except Exception:
                try:
                    await self._room.datasets.create_table_with_schema(
                        name=self.TABLE_NAME,
                        schema=schema,
                        mode="create_if_not_exists",
                    )
                except Exception:
                    # Another writer may have created the shared images table
                    # while this client was starting up.
                    logger.debug(
                        "unable to create images table; inspecting existing table",
                        exc_info=True,
                    )
                existing_schema = await self._room.datasets.inspect(
                    table=self.TABLE_NAME
                )

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
