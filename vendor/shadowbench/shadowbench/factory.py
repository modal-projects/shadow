"""Certified books by default; the v3.1 native environment replays its recorded episodes."""

from pathlib import Path
from typing import Literal

from .native_environment import NativeEnvironment,SYSTEM_PROMPT as NATIVE_PROMPT,TOOL_SPECS as NATIVE_TOOLS,fingerprint as native_fingerprint
from .reviewed_environment import ReviewedEnvironment,SYSTEM_PROMPT as REVIEWED_PROMPT,TOOL_SPECS as REVIEWED_TOOLS,fingerprint as reviewed_fingerprint
from . import curation
from .targets import DATA,catalogue,load

HandModel=Literal["native","native_v31"]
Episode=NativeEnvironment


def task_directory(task_id:str,revision:str|None=None,*,archived=False):
    if revision is not None:
        return curation.revision_directory(task_id,revision)
    if archived:
        return DATA/task_id
    task=curation.load_revision(task_id)
    return curation.revision_directory(task_id,task.metadata["revision"])


def load_task(task_id:str,revision:str|None=None,*,archived=False):
    if revision is not None or not archived:
        return curation.load_revision(task_id,revision)
    return load(task_id,task_directory(task_id,archived=True).parent)


def directory_for_task(task):
    return task_directory(task.task_id,task.metadata.get("revision"),archived="revision" not in task.metadata)


def tasks_for(model:HandModel="native"):
    if model=="native":return curation.review_catalogue(certified_only=True)
    if model!="native_v31":raise ValueError("unknown hand model")
    return catalogue()


def create(task_id:str,log_dir:Path|None=None,model:HandModel="native",*,revision:str|None=None,preview=False,
           plan_required=True)->Episode:
    if model=="native":
        task=curation.load_revision(task_id,revision,require_certified=not preview)
        return ReviewedEnvironment(task,log_dir,preview=preview,plan_required=plan_required)
    if model!="native_v31":raise ValueError("unknown hand model")
    if revision is not None:raise ValueError("historical environments cannot reinterpret a new book revision")
    return NativeEnvironment(load_task(task_id,archived=True),log_dir)


def model_for_version(recorded:str)->HandModel:
    if recorded==reviewed_fingerprint():return "native"
    if recorded==native_fingerprint():return "native_v31"
    raise ValueError("environment changed; this replay requires its recorded version")


def protocol_for(env:Episode):
    if isinstance(env,ReviewedEnvironment):return REVIEWED_PROMPT,REVIEWED_TOOLS
    return NATIVE_PROMPT,NATIVE_TOOLS
