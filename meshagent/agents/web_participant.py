from meshagent.api import Participant, RemoteParticipant


class WebParticipant(RemoteParticipant):
    def __init__(self, *, participant: Participant, connection_id: str) -> None:
        attributes = dict(participant.attributes)
        attributes["base_participant_id"] = participant.id
        attributes["websocket_connection_id"] = connection_id
        super().__init__(
            id=connection_id,
            role="user",
            attributes=attributes,
            online=True,
        )
