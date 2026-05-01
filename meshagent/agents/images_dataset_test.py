from typing import cast

from meshagent.api import RoomClient
from meshagent.agents.images_dataset import ImagesDataset


def test_images_dataset_marks_data_column_as_image_content() -> None:
    schema = ImagesDataset(room=cast(RoomClient, None))._schema()

    assert schema.field("data").metadata == {b"content-type": b"image/*"}
