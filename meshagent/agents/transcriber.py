import logging
import asyncio

from meshagent.api.schema_document import Element,Text
from meshagent.api.room_server_client import RoomClient
from meshagent.api.websocket_protocol import WebSocketClientProtocol


import asyncio
import os
import logging
import json

from livekit import api

from livekit.agents import stt, transcription, utils
from livekit.plugins import openai, silero
from livekit import rtc
from livekit.rtc import TranscriptionSegment
from livekit.agents import utils
from livekit.agents import stt as speech_to_text

from meshagent.api.runtime import RuntimeDocument

from typing import Optional

from meshagent.api.schema import MeshSchema

from meshagent.api.schema import MeshSchema, ElementType, ChildProperty, ValueProperty

from meshagent.agents.agent import AgentCallContext
from meshagent.agents import TaskRunner

logger = logging.getLogger("transcriber")
logger.setLevel(logging.INFO)

transcription_schema = MeshSchema(
    root_tag_name="transcript",
    elements=[
        ElementType(tag_name="transcript", description="a transcript", properties=[
            ChildProperty(name="transcriptions", description="the transcript entries", child_tag_names=[ "speech" ])
        ]),
        ElementType(tag_name="speech", description="transcribed speech", properties=[
            ValueProperty(name="text", description="the transcribed text", type="string"),
            ValueProperty(name="startTime", description="the time of the start of this speech", type="number"),
            ValueProperty(name="endTime", description="the time of th end of this speech", type="number"),
            ValueProperty(name="participantId", description="the identity of the participant", type="string"),
            ValueProperty(name="participantName", description="the name of the participant", type="string")
        ])
    ]
)

class Transcriber(TaskRunner):

    def __init__(self, *, livekit_url: Optional[str] = None, livekit_api_key: Optional[str] = None, livekit_api_secret: Optional[str] = None, livekit_identity: Optional[str] = None):
        super().__init__(
            name="livekit.transcriber",
            title="transcriber",
            description="connects to a livekit room and transcribes the conversation",
            input_schema={
                "type" : "object",
                "additionalProperties" : False,
                "required" : [ "room_name", "path" ],
                "properties" : {
                    "room_name" : {
                        "type": "string"
                    },
                    "path" : {
                        "type" : "string"
                    }
                }
            },
            output_schema={
                "type" : "object",
                "additionalProperties" : False,
                "required" : [],
                "properties" : {
                }
            }
        )
        self._livekit_url = livekit_url
        self._livekit_api_key = livekit_api_key
        self._livekit_api_secret = livekit_api_secret
        self._livekit_identity = livekit_identity

    async def _transcribe_participant(self, doc: RuntimeDocument, room: rtc.Room, participant: rtc.RemoteParticipant,
            stt_stream: stt.SpeechStream, stt_forwarder: transcription.STTSegmentsForwarder
        ):
            logger.info("transcribing participant %s", participant.sid)
            """Forward the transcription to the client and log the transcript in the console"""
            async for ev in stt_stream:
                logger.info("event from participant %s %s", participant.sid, ev)
                             
                if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    logger.info("transcript: %s", ev.alternatives[0].text)
                    if len(ev.alternatives) > 0:
                        alt = ev.alternatives[0]                                                
                        doc.root.append_child(tag_name="speech", attributes={ "text":  alt.text, "startTime": alt.start_time, "endTime" : alt.end_time, "participantId" : participant.identity, "participantName" : participant.name })

            logger.info("done forwarding %s", participant.sid)

    def should_transcribe(self, p: rtc.Participant) -> bool:
        # don't transcribe other agents
        # todo: maybe have a better way to detect
        return ".agent" not in p.identity
    
    async def _wait_for_disconnect(self, room: rtc.Room):
        disconnected = asyncio.Future()
        def on_disconnected(_):
            disconnected.set_result(True)
        room.on("disconnected", on_disconnected)

        logger.info("waiting for disconnection")
        await disconnected                 


    async def ask(self, *, context: AgentCallContext, arguments: dict):
        logger.info("Transcriber connecting to %s", arguments)
        output_path = arguments["path"]
        room_name = arguments["room_name"]

        client = context.room        
        doc = await client.sync.open(path=output_path)
        try:

            vad = silero.VAD.load()                
            utils.http_context._new_session_ctx()              
            
            pending_tasks = list()
            participantNames = dict[str,str]()

            sst_provider = openai.STT()
            #sst_provider = fal.WizperSTT()

            room_options = rtc.RoomOptions(auto_subscribe=False)

            room = rtc.Room()

            url = self._livekit_url if self._livekit_url is not None else os.getenv('LIVEKIT_URL')
            api_key = self._livekit_api_key if self._livekit_api_key is not None else os.getenv('LIVEKIT_API_KEY')
            api_secret = self._livekit_api_secret if self._livekit_api_secret is not None else os.getenv('LIVEKIT_API_SECRET')
            identity = self._livekit_identity if self._livekit_identity is not None else os.getenv('AGENT_IDENTITY')

            token = api.AccessToken(api_key=api_key, api_secret=api_secret) \
                .with_identity(identity) \
                .with_name("Agent") \
                .with_kind("agent") \
                .with_grants(api.VideoGrants(
                    can_update_own_metadata=True,
                    room_join=True,
                    room=room_name,
                    agent=True
                ))
            
            jwt = token.to_jwt()
            
            await room.connect(url=url, token=jwt, options=room_options)
            

            logger.info(
                "connected to room: %s",
                room_name
            )
        
            audio_streams = list[rtc.AudioStream]()

            async def transcribe_track(participant: rtc.RemoteParticipant, track: rtc.Track):
                audio_stream = rtc.AudioStream(track)
                stt_forwarder = transcription.STTSegmentsForwarder(
                    room=room, participant=participant, track=track
                )

                audio_streams.append(audio_stream)
                
                stt = sst_provider
                if not sst_provider.capabilities.streaming:                
                    stt = speech_to_text.StreamAdapter(
                        stt=stt,
                        vad=vad,
                    )
                
                stt_stream = stt.stream()

                pending_tasks.append(asyncio.create_task(self._transcribe_participant(doc, room, participant, stt_stream, stt_forwarder)))

                async for ev in audio_stream:
                    stt_stream.push_frame(ev.frame)

            def subscribe_if_needed(pub: rtc.RemoteTrackPublication):
                if pub.kind == rtc.TrackKind.KIND_AUDIO:
                    pub.set_subscribed(True)
                
            for p in room.remote_participants.values():
                participantNames[p.identity] = p.name
                if self.should_transcribe(p):
                    for pub in p.track_publications.values():
                        subscribe_if_needed(pub)
            
            first_parts = dict[str, rtc.Participant]()
        
            def on_transcript_event(segments: list[TranscriptionSegment], part: rtc.Participant | None, pub: rtc.TrackPublication | None = None) -> None:
                nonlocal room
                logger.info("Got transcription segment %s %s %s", segments, part, pub)
                for segment in segments:
                    if segment.id not in first_parts and part != None:
                        first_parts[segment.id] = part

                    if segment.final:
                        if part == None and segment.id in first_parts:
                            part = first_parts[segment.id]
                            first_parts.pop(segment.id)
                        
                        if part != None:
                            doc.root.append_child(tag_name="speech", attributes={ "text":  segment.text, "startTime": segment.start_time, "endTime" : segment.end_time, "participantId" : part.identity, "participantName" : part.name })
                        else:
                            logger.warning("transcription was missing participant information")

            def on_participant_connected(p: rtc.RemoteParticipant):
                participantNames[p.identity] = p.name

            def on_track_published(pub: rtc.RemoteTrackPublication, p: rtc.RemoteParticipant):
                if self.should_transcribe(p):
                    subscribe_if_needed(pub)

            subscriptions = dict()
            
            def on_track_unpublished(pub: rtc.RemoteTrackPublication, p: rtc.RemoteParticipant):
                if pub in subscriptions:
                    logger.info("track unpublished, stopping transcription")
                    # todo: maybe could be more graceful
                    subscriptions[pub].cancel()
                    subscriptions.pop(pub)
                
            
            def on_track_subscribed(
                track: rtc.Track,
                publication: rtc.TrackPublication,
                participant: rtc.RemoteParticipant,
            ):
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info("transcribing track %s", track.sid)
                    track_task = asyncio.create_task(transcribe_track(participant, track))
                    def on_transcription_done(t):
                        try:
                            t.result()
                        except Exception as e:
                            logger.error("Transcription failed", exc_info=e)


                    track_task.add_done_callback(on_transcription_done)
                    pending_tasks.append(track_task)
                    subscriptions[publication] = track_task
            

            for p in room.remote_participants.values():
                on_participant_connected(p)
                            
            room.on("participant_connected", on_participant_connected)

            room.on("track_published", on_track_published)
            room.on("track_unpublished", on_track_unpublished)
            room.on("track_subscribed", on_track_subscribed)
            room.on("transcription_received", on_transcript_event)

            await self._wait_for_disconnect(room)
                        
            logger.info("waited for termination")
            await room.disconnect()
            
            logger.info("closing audio streams")
            
            for stream in audio_streams:
                await stream.aclose()

    
            logger.info("waiting for pending tasks")
            gather_future = asyncio.gather(*pending_tasks)

            gather_future.cancel()
            try:
                await gather_future
            except Exception as e:
                if isinstance(e, asyncio.CancelledError) == False:
                    logger.warning("Did not shut down cleanly", exc_info=e)
                pass

            print("done")
        except Exception as e:
            logger.info("Transcription failed", exc_info=e)
        finally:

            await utils.http_context._close_http_ctx()
            logger.info("Transcription done")

            await asyncio.sleep(5)
            await client.sync.close(path=output_path)
            
            return {}
