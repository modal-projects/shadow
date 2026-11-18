"""ShadowBench: a benchmark and RL environment for spatial reasoning about the body.

A model poses two articulated 3D hands between a point lamp and a screen, one tool call at a time, until their
shadow matches a hand-shadow plate from a nineteenth-century book.

    from shadowbench import create, tasks
    env = create("bursill_bunny")        # the certified target the published evaluation used
    image = env.observe()                # target | shadow | difference | hands, as the model first sees it
    out = env.call("plan", {"subject": "rabbit", "arrangement": "index and middle fingers as the ears"})
    out = env.call("set_fingers", {"hand": "right", "joints": {"ring.mcp": 40, "little.mcp": 40}})
    out.text, out.image                  # what the model receives
    env.metrics()["score"]               # the hidden score of the current pose (the model never sees it)
"""

from .agent import CONTINUE, USER_PROMPT
from .factory import create, load_task
from .reviewed_environment import SYSTEM_PROMPT, TOOL_SPECS, ReviewedEnvironment


def tasks() -> list[str]:
    """The 20 certified plates of the benchmark."""
    from .factory import tasks_for

    return [t["task_id"] for t in tasks_for()]


__all__ = ["CONTINUE", "SYSTEM_PROMPT", "TOOL_SPECS", "USER_PROMPT", "ReviewedEnvironment", "create", "load_task", "tasks"]
