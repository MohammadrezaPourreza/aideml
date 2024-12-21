import time
import logging
import os
import json
import vertexai

import google.api_core.exceptions
from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part, FunctionDeclaration, Tool
from vertexai.generative_models import HarmBlockThreshold, HarmCategory, ToolConfig
from google.generativeai.generative_models import generation_types

from funcy import once
from .utils import FunctionSpec, OutputType, backoff_create

logger = logging.getLogger("aide")

gdm_model = None  # type: ignore
generation_config = None  # type: ignore

GDM_TIMEOUT_EXCEPTIONS = (
    google.api_core.exceptions.RetryError,
    google.api_core.exceptions.TooManyRequests,
    google.api_core.exceptions.ResourceExhausted,
    google.api_core.exceptions.InternalServerError,
)

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}


@once
def _setup_gdm_client(model_name: str, temperature: float):
    global gdm_model
    global generation_config

    vertexai.init(project=os.environ["GCP_PROJECT"], location=os.environ["GCP_REGION"])
    gdm_model = GenerativeModel(model_name)
    generation_config = GenerationConfig(temperature=temperature)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    model = model_kwargs.pop("model")
    temperature = model_kwargs.pop("temperature", None)

    _setup_gdm_client(model, temperature)

    tools = []
    tool_config = None
    if func_spec is not None:
        tools = [Tool(function_declarations=[FunctionDeclaration(**func_spec.as_gemini_tool_dict)])]
        tool_config = ToolConfig(
            function_calling_config=ToolConfig.FunctionCallingConfig(
                mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                allowed_function_names=[func_spec.name],
            )
        )


    # Create a list of Content objects from the system and user messages
    messages = []
    if system_message:
        # Since system messages aren't directly supported outside beta, 
        # you might want to treat them as user messages or skip.
        # Here we keep them as system role messages if supported.
        messages.append(Content(role="user", parts=[Part.from_text(system_message)]))
    if user_message:
        messages.append(Content(role="user", parts=[Part.from_text(user_message)]))

    t0 = time.time()
    response: generation_types.GenerateContentResponse = backoff_create(
        gdm_model.generate_content,
        retry_exceptions=GDM_TIMEOUT_EXCEPTIONS,
        contents=messages,
        generation_config=generation_config,
        tools=tools,
        tool_config=tool_config,
        safety_settings=SAFETY_SETTINGS,
    )
    req_time = time.time() - t0

    if response.prompt_feedback.block_reason:
        output = str(response.prompt_feedback)
    else:
        if func_spec is None:
            output = response.text
        else:
            assert (
            response.candidates
            ), f"function_call is empty, it is not a function call: {response}"
            assert (
                response.candidates[0].function_calls
            ), f"function_call is empty, it is not a function call: {response}"
            assert (
                response.candidates[0].function_calls[0].name == func_spec.name
            ), "Function name mismatch"
            try:
                output = json.loads(response.candidates[0].function_calls[0].args)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Error decoding the function arguments: {response.candidates[0].function_calls[0].args}"
                )
                raise e
    in_tokens = response.usage_metadata.prompt_token_count
    out_tokens = response.usage_metadata.candidates_token_count
    info = {}  # this isn't used anywhere, but is an expected return value

    return output, req_time, in_tokens, out_tokens, info
