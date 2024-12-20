"""Backend for GDM Gemini API"""

import time
import logging
import os
import vertexai

import google.api_core.exceptions
# import google.generativeai as genai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.generative_models import HarmBlockThreshold
from vertexai.generative_models import HarmCategory
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

    if func_spec is not None:
        raise NotImplementedError(
            "GDM supports function calling but we won't use it for now."
        )

    # GDM gemini api doesnt support system messages outside of the beta
    messages = [
        {"role": "user", "parts": message}
        for message in [system_message, user_message]
        if message
    ]

    t0 = time.time()
    response: generation_types.GenerateContentResponse = backoff_create(
        gdm_model.generate_content,
        retry_exceptions=GDM_TIMEOUT_EXCEPTIONS,
        contents=messages,
        generation_config=generation_config,
        safety_settings=SAFETY_SETTINGS,
    )
    req_time = time.time() - t0

    if response.prompt_feedback.block_reason:
        output = str(response.prompt_feedback)
    else:
        output = response.text
    in_tokens = response.usage_metadata.prompt_token_count
    out_tokens = response.usage_metadata.candidates_token_count
    info = {}  # this isnt used anywhere, but is an expected return value

    # only `output` is actually used by scaffolding
    return output, req_time, in_tokens, out_tokens, info
