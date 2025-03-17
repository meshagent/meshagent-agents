from meshagent.agents.agent import AgentCallContext, AgentChatContext, AgentException
from meshagent.api import WebSocketClientProtocol, RequiredToolkit, RequiredSchema, Requirement
from meshagent.api.schema_document import Document
from meshagent.api.room_server_client import RoomClient
from meshagent.tools.toolkit import Toolkit, TextResponse, Tool, ToolContext
from meshagent.api.schema import MeshSchema
from meshagent.agents.writer import Writer, WriterContext
from meshagent.agents.adapter import LLMAdapter, ToolResponseAdapter
from meshagent.api.schema import MeshSchema, ElementType, ChildProperty, ValueProperty
from meshagent.api.schema_util import merge
from meshagent.tools.document_tools import build_tools, DocumentAuthoringToolkit
from meshagent.agents import TaskRunner
from copy import deepcopy
from typing import Optional

reasoning_rules = [
    "If an ask_user tool call is available, plans should include a series of questions to ask the user to help refine.",
    "If an ask_user tool call is available, ask a maximum of one question per step",
    "If an ask_user tool call is not available, you may not ask the user any questions",
    "You will be given a task. First formulate a plan for the task. Then execute the steps until you have a final answer.",
    "Do not use tool calls to write estimates of progress or plans",
    "You are a document generation service",
    "The user is asking for a document to be created",
    "You must use tool calls must to generate the answer to the user's question as document"
]

goto_next_step_message = """
     execute the next step, and provide the result of the step.
"""

def is_reasoning_done(*, context: AgentCallContext, response: dict) -> bool:
    parsed = response["response"]["data"][0]
    if "abort" in parsed:
        abort = parsed["abort"]
        reason = abort["reason"]
        raise AgentException(reason)
        
    elif "progress" in parsed:
        result = parsed["progress"]
                            
        if "done" not in result or result["done"]:
            logger.info("Done generating response %s", result)
            return True
        else:
            context.chat.append_user_message(goto_next_step_message)
            return False
        
    elif "plan" in parsed:
        plan = parsed["plan"]
        context.chat.append_user_message(goto_next_step_message)
        return False
    else:
        logger.info("recieved invalid response, %s", parsed)
        context.chat.append_user_message("this response did not conform to the schema")
        return False

def reasoning_schema(*, description: str, elements: Optional[list[ElementType]] = None, has_done_property: bool = True, has_abort: bool = True) -> MeshSchema:

    if elements == None:
        elements = []

    progress_properties = [
        ValueProperty(name="percentage", description="an estimate for how complete the task is so far", type="string"),
        ValueProperty(name="next", description="a very short description of the next step", type="string"),
    ]

    if has_done_property:
        progress_properties.append( ValueProperty(name="done", description="whether there is more work to do. the program will continue will continue to send messages to the LLM to refine the answer until done is set to true.", type="boolean") )

    elements = elements.copy()

    if has_abort:
        elements.append(ElementType(
                tag_name="abort",
                description="return if the task cannot completed because the user cancelled a request or errors could not be resolved",
                properties=[
                    ValueProperty(name="reason", description="the reason the task was aborted", type="string")
                ]
            )),

    return MeshSchema(
        root_tag_name="response",
        elements=[
            ElementType(
                tag_name="response",
                description="a response for a task",
                properties=[
                    ChildProperty(name="data", description="the response for a task, should contain a single item", child_tag_names=[
                        "plan","progress",
                        *map(lambda x: x.tag_name, elements)    
                    ])
                ]
            ),
            ElementType(
                tag_name="plan",
                description="a plan will be output for each task to describe the work that will be done, the work will be performed using tool calls.",
                properties=[
                    ChildProperty(name="steps", description="the steps for the plan", child_tag_names=["step"])
                ]
            ),
            ElementType(
                tag_name="step",
                description="a step in the plan",
                properties=[
                    ValueProperty(name="description", description="a short sentence description description of the work that will be performed to complete the user's request.", type="string")
                ],
            ),
            ElementType(
                tag_name="progress",
                description="the progress of the task",
                properties=progress_properties
            ),
            ElementType(
                tag_name="thinking",
                description="use to log information that will not be included in the final answer",
                properties=[
                    ValueProperty(name="text", description="used to log thoughts or progress", type="string"),
                ],
            ),
            *elements
        ])

from meshagent.api.schema_util import prompt_schema
import logging
logging.basicConfig()
logger = logging.getLogger("planning_agent")
logger.setLevel(logging.INFO)


class PlanningWriter(Writer):
    
    def __init__(self, *, name: str, llm_adapter: LLMAdapter, tool_adapter:  Optional[ToolResponseAdapter] = None, max_iterations : int = 100, toolkits: Optional[list[Tool]] = None, title: Optional[str] = None, description: Optional[str] = None, rules: Optional[list[str]] | None = None, requires: Optional[list[Requirement]] = None, supports_tools: Optional[bool] = None):
        
        super().__init__(
            name=name,
            description=description,
            title=title,
            input_schema=merge(
                schema=prompt_schema(description="use a prompt to generate content"), 
                additional_properties={
                "path" : { "type" : "string" }
            }),
            output_schema={
                "type" : "object",
                "additionalProperties" : False,
                "required" : [],
                "properties" : {
                }
            },
            requires = requires,
            supports_tools = supports_tools
        )

        if rules == None:
            rules = []

        self._rules = rules

        self._llm_adapter = llm_adapter
        self._tool_adapter = tool_adapter 
        self._max_iterations = max_iterations
        if toolkits == None:
            toolkits = []
        self.toolkits = toolkits

        self._planning_rules : list[str] = [
            *reasoning_rules,
            *rules
        ]  

    async def init_chat_context(self):
        chat = await super().init_chat_context()

        all_rules = self._planning_rules.copy()
        chat.append_rules(rules=all_rules)
        return chat
        
    async def write(self, writer_context: WriterContext, arguments: dict):

        writer_context.call_context.chat.append_rules(f"your are writing to the document at the path {writer_context.path}")

        arguments = arguments.copy()
        self.pop_path(arguments=arguments)
        
        execute = goto_next_step_message

        react = """
        based on what you know know, either execute the next task or formulate a new plan. If you have sufficient information to complete the task, return a final answer.
        """
    
        prompt = arguments["prompt"]

        writer_context.call_context.chat.append_user_message(message=prompt)

        rs = reasoning_schema(description="uses tools", elements=[]).to_json()

        i = 0
        while i < self._max_iterations:
            i += 1
            
            try:
                logger.info("COMPLETION STARTING: Step %s", i)
     
                base_args = arguments.copy()
                base_args.pop("path")
                
                toolkits = [
                    DocumentAuthoringToolkit(),
                    Toolkit(
                        name="meshagent.planning-writer.tools",
                        tools=build_tools(document_type="document", schema=writer_context.document.schema, documents={writer_context.path : writer_context.document})),
                    *self.toolkits,
                    *writer_context.call_context.toolkits
                ]

                responses = await self._llm_adapter.next(context=writer_context.call_context.chat, room=writer_context.room, toolkits=toolkits, tool_adapter=self._tool_adapter, output_schema=rs)

                logger.info("COMPLETION RESPONSE %s", responses)
            
            except Exception as e:
                logger.error("Unable to execute reasoning completion task", exc_info=e)
                # retry
                raise(e)
            
            parsed = responses["response"]["data"][0]
            if "abort" in parsed:
                abort = parsed["abort"]
                reason = abort["reason"]
                raise AgentException(reason)
                
            elif "progress" in parsed:
                result = parsed["progress"]

                                    
                if "done" not in result or result["done"]:
                    logger.info("Done generating response %s", result)
                    return {}
                else:
                    writer_context.call_context.chat.append_user_message(execute)
                    continue
            elif "plan" in parsed:
                plan = parsed["plan"]
                writer_context.call_context.chat.append_user_message(execute)
                continue
            else:
                logger.info("recieved invalid response, %s", parsed)
                writer_context.call_context.chat.append_user_message("this response did not conform to the schema")
                continue

 
class PlanningResponder(TaskRunner):

    def __init__(self, *, name:str, llm_adapter: LLMAdapter, tool_adapter:  Optional[ToolResponseAdapter] = None, output_schema: dict,  max_iterations : int = 100, toolkits: Optional[list[Toolkit]] = None, title: Optional[str] = None, description: Optional[str] = None, requires: Optional[list[Requirement]] = None, supports_tools : bool = True, input_prompt: bool = True, use_terminate_tool: bool = False, rules: Optional[list[str]] = None, labels: Optional[list[str]] = None):
        if isinstance(output_schema, dict) == False:
            raise Exception("schema must be a dict, got: {type}".format(type=type(output_schema)))

        self._input_prompt = input_prompt

        if rules == None:
            rules = []


        if input_prompt:
            input_schema = prompt_schema(description="use a prompt to generate content")
        else:
            input_schema = {
                "type" : "object",
                "additionalProperties" : False,
                "required" : [],
                "properties" : {
                }
            }

        super().__init__(
            name=name,
            title=title,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            requires=requires,
            supports_tools=supports_tools,
            labels=labels
        )
        
        self._max_iterations = max_iterations

        self._planning_rules : list[str] = [
            *rules,
            *reasoning_rules
        ]

        self._responses = dict()

        self._llm_adapter = llm_adapter
        self._tool_adapter = tool_adapter 

        if toolkits == None:
            toolkits = []
        
        self.toolkits = toolkits

        self._use_terminate_tool = use_terminate_tool

    async def init_chat_context(self):
        chat = self._llm_adapter.create_chat_context()

        all_rules = self._planning_rules.copy()
        chat.append_rules(rules=all_rules)
        return chat


    async def ask(self, context: AgentCallContext, arguments: dict):

        class ResponseTool(Tool):
            def __init__(self, output_schema: dict, context: AgentCallContext, parent:PlanningResponder):
                super().__init__(
                    name="respond",
                    title="respond", 
                    description="send the response to the user",     
                    input_schema=output_schema
                )
                self.parent = parent
                self.context = context 

            async def execute(self, *, context: ToolContext, **kwargs):
                self.parent._responses[self.context] = kwargs
                return TextResponse(text="the response was sent")
            
        terminated = False

        class ExitTool(Tool):
            def __init__(self):
                super().__init__(
                    name="terminate",
                    title="terminate", 
                    description="terminates the agent, the agent will no longer be available",     
                    input_schema={
                        "type" : "object",
                        "required" : [],
                        "additionalProperties" : False,
                        "properties" : {}
                    }
                )

            async def execute(self, *, context: ToolContext, **kwargs):
                nonlocal terminated
                terminated = True
                return TextResponse(text="the process was terminated")
        
        use_terminate_tool = self._use_terminate_tool
        

        class ResponseToolkit(Toolkit):
            def __init__(self, output_schema, context, parent):

                tools = [    
                    ResponseTool(output_schema=output_schema, context=context, parent=parent)
                ]

                if use_terminate_tool:
                    tools.append(ExitTool())

                super().__init__(
                    name="meshagent.responder",
                    title="responder",
                    description="tools for responding",
                    tools=tools
                )
                

        context.toolkits.append(ResponseToolkit(output_schema=self.output_schema, context=context, parent=self))
        
        execute = goto_next_step_message

        react = """
        based on what you know know, either execute the next task or formulate a new plan. If you have sufficient information to complete the task, return a final answer.
        """
    
        rs = reasoning_schema(description="uses tools", elements=[], has_done_property=self._use_terminate_tool == False, has_abort=self._use_terminate_tool == False).to_json()

        if self._input_prompt:
            prompt = arguments["prompt"]
            context.chat.append_user_message(message=prompt)

        room = context.room
        i = 0
        while i < self._max_iterations and not terminated:
            i += 1
            
            try:
                logger.info("COMPLETION STARTING: Step %s", i)
                
                toolkits = [
                    *self.toolkits,
                    *context.toolkits
                ]

                responses = await self._llm_adapter.next(context=context.chat, room=room, toolkits=toolkits, tool_adapter=self._tool_adapter, output_schema=rs)


                logger.info("COMPLETION RESPONSE %s", responses)
            
            except Exception as e:
                logger.error("Unable to execute reasoning completion task", exc_info=e)
                # retry
                raise(e)
            
            parsed = responses["response"]["data"][0]

            if "abort" in parsed:
                abort = parsed["abort"]
                reason = abort["reason"]
                raise AgentException(reason)
                
            elif "progress" in parsed:
                result = parsed["progress"]

                if "done" in result and result["done"]:                   
                    if context not in self._responses:
                        context.chat.append_user_message("you must call the respond tool")
                        continue
                    
                    final_answer = self._responses.pop(context)
                    logger.info("Done generating response %s", final_answer)
                    return final_answer
                
                else:
                    context.chat.append_user_message(execute)
                    continue
            elif "plan" in parsed:
                plan = parsed["plan"]
                context.chat.append_user_message(execute)
                continue
            else:
                logger.info("recieved invalid response, %s", parsed)
                context.chat.append_user_message("this response did not conform to the schema")
                continue

        
        return {}

class DynamicPlanningResponder(TaskRunner):

    def __init__(self, *, name: str, llm_adapter: LLMAdapter, tool_adapter:  Optional[ToolResponseAdapter] = None,  max_iterations : int = 100, toolkits: Optional[list[Toolkit]] = None, title: Optional[str] = None, description: Optional[str] = None):
        
        super().__init__(
            name=name,
            title=title,
            description=description,
            input_schema=merge(
                schema=prompt_schema(
                    description="use a prompt to generate content"),
                    additional_properties={
                    "output_schema" : { "type" : "object" }
                }
            ),
            output_schema=None)
            
        self._max_iterations = max_iterations

        self._planning_rules : list[str] = [
            *reasoning_rules
        ]

        self._responses = dict()

        self._llm_adapter = llm_adapter
        self._tool_adapter = tool_adapter 

        if toolkits == None:
            toolkits = []
        
        self.toolkits = toolkits
        

    async def init_chat_context(self):
        chat  = self._llm_adapter.create_chat_context()

        all_rules = self._planning_rules.copy()
        chat.append_rules(rules=all_rules)
        return chat


    async def ask(self, context: AgentCallContext, arguments: dict):

        dynamic_schema = arguments["output_schema"]

     
        class ResponseTool(Tool):
            def __init__(self, output_schema: dict, context: AgentCallContext, parent:PlanningResponder):

                output_schema = deepcopy(output_schema)
           
                schema = {
                    "type" : "object",
                    "additionalProperties" : False,
                    "required" : [ "summary", "data"], 
                    "properties" : {
                        "summary" :  {
                            "type" : "string",
                            "description" : "a summary of the data structure"
                        },
                        "data": output_schema
                    }
                }

                if "$defs" in output_schema:
                    schema["$defs"] = output_schema["$defs"]
                    del output_schema["$defs"]
                        
                super().__init__(
                    name="respond",
                    title="respond",
                    input_schema=schema,
                    description="send the response to the user",
                )
                self.parent = parent
                self.context = context 

            async def execute(self, *, context: ToolContext, **kwargs):
                self.parent._responses[self.context] = kwargs
                return TextResponse(text="the response was sent")

        class ResponseToolkit(Toolkit):
            def __init__(self, output_schema, context, parent):
                super().__init__(
                    name="meshagent.dynamic_response",
                    tools=[    
                        ResponseTool(output_schema=output_schema, context=context, parent=parent)
                    ]
                )

        context.toolkits.append(ResponseToolkit(output_schema=dynamic_schema, context=context, parent=self))
        
        execute = goto_next_step_message

        react = """
        based on what you know know, either execute the next task or formulate a new plan. If you have sufficient information to complete the task, return a final answer.
        """
    
        rs = reasoning_schema(description="uses tools", elements=[]).to_json()

        prompt = arguments["prompt"]

        context.chat.append_user_message(message=prompt)
        
        room = context.room
    
        i = 0
        while i < self._max_iterations:
            i += 1
            
            try:
                logger.info("COMPLETION STARTING: Step %s", i)
                
                toolkits = [
                    *self.toolkits,
                    *context.toolkits
                ]

                responses = await self._llm_adapter.next(context=context.chat, room=room, toolkits=toolkits, tool_adapter=self._tool_adapter, output_schema=rs)


                logger.info("COMPLETION RESPONSE %s", responses)
            
            except Exception as e:
                logger.error("Unable to execute reasoning completion task", exc_info=e)
                # retry
                raise(e)
            
            parsed = responses["response"]["data"][0]

            if "abort" in parsed:
                abort = parsed["abort"]
                reason = abort["reason"]
                raise AgentException(reason)
                
            elif "progress" in parsed:
                result = parsed["progress"]

                if "done" not in result or result["done"]:                   
                    if context not in self._responses:
                        context.chat.append_user_message("you must call the respond tool")
                        continue
                    
                    final_answer = self._responses.pop(context)
                    
                    logger.info("Done generating response %s", final_answer)
                    return final_answer["data"]
                
                else:
                    context.chat.append_user_message(execute)
                    continue
            elif "plan" in parsed:
                plan = parsed["plan"]
                context.chat.append_user_message(execute)
                continue
            else:
                logger.info("recieved invalid response, %s", parsed)
                context.chat.append_user_message("this response did not conform to the schema")
                continue

 
