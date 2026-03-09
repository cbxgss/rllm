import copy
import json
import logging
import os
import uuid
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.system_prompts import TOOL_SYSTEM_PROMPT
from rllm.parser import ToolParser, get_tool_parser
from rllm.tools.mcp_tool import MCPTool
from rllm.tools.multi_tool import MultiTool
from rllm.tools.tool_base import Tool

logger = logging.getLogger(__name__)


class ToolAgent(BaseAgent):
    """
    An tool agent that can use tools to interact with the environment,
    refactored to follow the BaseAgent abstraction.
    """

    def __init__(
        self,
        system_prompt=TOOL_SYSTEM_PROMPT,
        parser_name="qwen",
        tools: list[str] | None = None,
        tool_map: dict[str, type[Tool]] | None = None,
        log_dir: str | None = None,
        enable_logging: bool = False,
        # 5.4.1 Outlier Suppression: Break + 0 reward
        enable_outlier_suppression: bool = False,
        max_tool_calls_per_step: int = 10,
        check_duplicate_queries: bool = False,
    ):
        """
        Initialize the ToolAgent.

        Args:
            system_prompt: System prompt for the agent.
            parser_name: Name of the parser to use for tool calls.
            tools: List of tool names available to the agent (legacy behavior).
            tool_map: Dictionary mapping tool names to Tool classes (new behavior).
            log_dir: Directory to save trajectory logs.
            enable_logging: Whether to enable trajectory logging.
            enable_outlier_suppression: Whether to enable 5.4.1 outlier suppression (break + 0 reward).
            max_tool_calls_per_step: Maximum number of tool calls per step before breaking (for 5.4.1).
            check_duplicate_queries: Whether to check for duplicate queries (for 5.4.1).
        """
        if tool_map is not None and tools is not None:
            raise ValueError("Cannot specify both 'tools' and 'tool_map' parameters")

        self.system_prompt = system_prompt
        self.log_dir = log_dir
        self.enable_logging = enable_logging

        # 5.4.1 Outlier Suppression: Break + 0 reward 配置
        self.enable_outlier_suppression = enable_outlier_suppression
        self.max_tool_calls_per_step = max_tool_calls_per_step
        self.check_duplicate_queries = check_duplicate_queries

        # 初始化 MultiTool with either tools or tool_map
        if tool_map is not None:
            self.tools = MultiTool(tool_map=tool_map)
        elif tools is not None:
            self.tools = MultiTool(tools=tools)
        else:
            self.tools = MultiTool(tools=[])

        parser_class: type[ToolParser] = get_tool_parser(parser_name=parser_name)
        self.tool_parser = parser_class()

        self.tools_prompt = self.tool_parser.get_tool_prompt(json.dumps(self.tools.json, indent=2))

        # Initialize state according to BaseAgent
        self._trajectory = Trajectory()
        self.messages: list[dict[str, Any]] = []
        self.current_observation = None

        # Logging state
        self.current_uid: str | None = None
        self.step_count: int = 0
        self.log_data: dict = {}

        # 5.4.1 Outlier Suppression: 用于检测重复查询的历史记录
        self.previous_queries: list[str] = []

        self.reset()  # Call reset to set initial state

    def _format_observation_as_messages(self, obs: Any) -> list[dict]:
        """Helper to format observation into messages."""
        messages = []
        if isinstance(obs, dict):
            if "question" in obs:
                messages.append({"role": "user", "content": obs["question"]})
            elif "tool_outputs" in obs:
                # Format tool outputs from environment observation
                for tool_call_id, tool_output_str in obs["tool_outputs"].items():
                    messages.append(
                        {
                            "role": "tool",
                            "content": tool_output_str,
                            "tool_call_id": tool_call_id,
                        }
                    )
        elif isinstance(obs, str):
            messages.append({"role": "user", "content": obs})
        elif obs:
            messages.append({"role": "user", "content": str(obs)})

        return messages

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Updates the agent's state based on environment feedback.
        Formats observation and updates the trajectory.
        """

        # Format the observation for the next model call
        obs_messages = self._format_observation_as_messages(observation)
        self.messages.extend(obs_messages)
        self.current_observation = observation

        if self._trajectory.steps:
            self._trajectory.steps[-1].reward = reward
            self._trajectory.steps[-1].done = done
            self._trajectory.steps[-1].info = info

        # Log tool outputs if available
        if self.enable_logging and isinstance(observation, dict) and "tool_outputs" in observation:
            step_key = f"step_{self.step_count}"
            if step_key in self.log_data:
                self.log_data[step_key]["tool_outputs"] = observation["tool_outputs"]
                self.log_data[step_key]["reward"] = reward
                self.log_data[step_key]["done"] = done
                if info:
                    self.log_data[step_key]["info"] = info

            self.step_count += 1

    def _save_step_log(self, model_output: dict[str, Any], tool_calls_dict: list[dict], tool_outputs: dict[str, str] | None = None):
        """Save step information to log data."""
        if not self.enable_logging or not self.log_dir or not self.current_uid:
            return

        step_key = f"step_{self.step_count}"
        self.log_data[step_key] = {
            "model_output": model_output,
            "tool_calls": tool_calls_dict,
        }

        if tool_outputs:
            self.log_data[step_key]["tool_outputs"] = tool_outputs

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's state based on the model's response.
        Parses the response, updates messages, and the current step in the trajectory.
        
        5.4.1 Outlier Suppression: Break + 0 reward
        - 检测工具解析错误
        - 检测单步工具调用数量超限
        - 检测重复查询
        """
        tool_calls_dict = []
        assistant_content = response
        has_parse_error = False
        
        # Attempt to parse tool calls from string response
        try:
            tool_calls = self.tool_parser.parse(response)
            tool_calls_dict = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": tool_call.to_dict(),
                }
                for tool_call in tool_calls
            ]

        except Exception as e:
            logger.error(f"Failed to parse tool calls from string response: {e}")
            has_parse_error = True
            tool_calls_dict = []  # Indicate no valid tool calls parsed

        # 5.4.1 Outlier Suppression: 检查是否触发异常条件
        should_break = False
        abnormal_reason = None

        if self.enable_outlier_suppression:
            # 1. 检测工具解析错误
            if has_parse_error:
                should_break = True
                abnormal_reason = "tool_parse_error"
                logger.warning(f"Tool parse error detected, breaking trajectory with 0 reward")

            # 2. 检测单步工具调用数量超限
            elif len(tool_calls_dict) > self.max_tool_calls_per_step:
                should_break = True
                abnormal_reason = f"too_many_tool_calls ({len(tool_calls_dict)} > {self.max_tool_calls_per_step})"
                logger.warning(f"Too many tool calls detected ({len(tool_calls_dict)}), breaking trajectory with 0 reward")

            # 3. 检测重复查询（仅针对 search 工具）
            elif self.check_duplicate_queries:
                for call in tool_calls_dict:
                    func_name = call.get("function", {}).get("name", "")
                    if func_name == "search":
                        func_args = call.get("function", {}).get("arguments", "")
                        if isinstance(func_args, dict):
                            query = func_args.get("query", "")
                        else:
                            # 尝试从 JSON 字符串解析
                            try:
                                args_dict = json.loads(func_args) if isinstance(func_args, str) else {}
                                query = args_dict.get("query", "")
                            except:
                                query = ""

                        # 检查是否与之前的查询重复
                        if query and query in self.previous_queries:
                            should_break = True
                            abnormal_reason = f"duplicate_query: {query}"
                            logger.warning(f"Duplicate query detected: {query}, breaking trajectory with 0 reward")
                            break
                        # 记录当前查询
                        if query:
                            self.previous_queries.append(query)

        # 如果触发异常条件，标记当前 step
        if should_break:
            # 记录异常信息到 step 的 info 中
            if self._trajectory.steps:
                self._trajectory.steps[-1].info = self._trajectory.steps[-1].info or {}
                self._trajectory.steps[-1].info["abnormal_reason"] = abnormal_reason
                self._trajectory.steps[-1].info["should_break"] = True

        # Log step information
        self._save_step_log(
            model_output={
                "content": assistant_content,
            },
            tool_calls_dict=tool_calls_dict,
        )

        # Append assistant message to chat history
        assistant_message = {"role": "assistant", "content": assistant_content}
        if len(tool_calls_dict) > 0:
            # Ensure arguments within tool_calls_dict are strings if needed by downstream processing
            for call in tool_calls_dict:
                if isinstance(call.get("function", {}).get("arguments"), dict):
                    call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
        else:
            tool_calls_dict = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": {
                            "response": assistant_content,
                        },
                    },
                }
            ]

        self.messages.append(assistant_message)

        new_step = Step(chat_completions=copy.deepcopy(self.chat_completions), action=tool_calls_dict, model_response=response, observation=self.current_observation)
        self._trajectory.steps.append(new_step)

        return Action(action=tool_calls_dict)

    def save_trajectory_log(self, final_reward: float, metadata: dict | None = None):
        """
        Save the trajectory log to a YAML file.

        Args:
            final_reward: The final reward for the episode.
            metadata: Optional metadata dictionary to include in the log.
        """
        if not self.enable_logging or not self.log_dir or not self.current_uid:
            return

        import yaml

        class LiteralStr(str):
            """Helper class for YAML multiline strings."""
            pass

        def _literal_str_representer(dumper, data):
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")

        yaml.add_representer(LiteralStr, _literal_str_representer)

        def _convert_multiline_str(obj):
            """Recursively convert strings with newlines to LiteralStr."""
            if isinstance(obj, str):
                if "\n" in obj:
                    return LiteralStr(obj)
                return obj
            elif isinstance(obj, dict):
                return {k: _convert_multiline_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_convert_multiline_str(i) for i in obj]
            else:
                return obj

        # Create log directory structure
        epoch = metadata.get("epoch", 0) if metadata else 0
        step = metadata.get("step", 0) if metadata else 0
        uid_dir = os.path.join(self.log_dir, f"epoch_{epoch}", f"step_{step}", self.current_uid)
        os.makedirs(uid_dir, exist_ok=True)

        # Prepare log data
        log_all = {
            "steps": self.log_data,
            "final_reward": final_reward,
            "total_steps": self.step_count,
        }

        if metadata:
            log_all["metadata"] = metadata

        # Convert multiline strings and save
        log_all = _convert_multiline_str(log_all)
        log_path = os.path.join(uid_dir, "log.yaml")
        with open(log_path, "w", encoding="utf-8") as f:
            yaml.dump(
                log_all,
                f,
                allow_unicode=True,
                sort_keys=False,
                width=512,
                default_flow_style=False,
            )

        logger.info(f"Saved trajectory log to {log_path}")

    def reset(self, uid: str | None = None):
        """
        Resets the agent's state for a new episode.

        Args:
            uid: Optional unique identifier for this episode, used for logging.
        """
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt + self.tools_prompt}]

        # Reset logging state
        # Only set current_uid if logging is enabled and uid is provided
        if self.enable_logging and self.log_dir and uid:
            self.current_uid = uid
        else:
            self.current_uid = None

        self.step_count = 0
        self.log_data = {}
        # 5.4.1 Outlier Suppression: 重置重复查询历史
        self.previous_queries = []

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Returns the current message history for the model."""
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        """Returns the trajectory recorded so far."""
        return self._trajectory


class MCPToolAgent(ToolAgent):
    def __init__(self, system_prompt=TOOL_SYSTEM_PROMPT, parser_name="qwen", tool_map=list[MCPTool]):
        self.system_prompt = system_prompt
        self.tool_map = tool_map

        parser_class: type[ToolParser] = get_tool_parser(parser_name=parser_name)
        self.tool_parser = parser_class()

        tools_json = [tool.json for tool in self.tool_map.values()]
        self.tools_prompt = self.tool_parser.get_tool_prompt(json.dumps(tools_json, indent=2))

        self._trajectory = Trajectory()
        self.messages: list[dict[str, Any]] = []
        self.reset()
