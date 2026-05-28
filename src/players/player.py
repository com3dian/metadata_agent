"""
Unified Player class for the multi-agent system.

A Player is a self-contained agent that can:
1. Execute tasks using tools
2. Participate in debates (generate work, critique, revise)
3. Synthesize results from multiple sources

Each player has a role/persona defined by a prompt, and a set of tools
it can use to accomplish tasks.

Uses the unified ExecutionContext abstraction for all data access.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Type

from pydantic import BaseModel, ValidationError
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..config import (
    PLAYER_TEMPERATURE,
    create_llm,
    LLM_PROVIDER,
    PLAYER_TOOL_EXECUTION_MODE,
    PLAYER_MAX_TOOL_ITERATIONS,
)


def _stringify_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _serialize_tool_result(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, default=str)
    except TypeError:
        return str(obj)


def _tool_model_fields(tool: BaseTool) -> Dict[str, Any]:
    sch = getattr(tool, "args_schema", None)
    if sch is None:
        return {}
    return dict(getattr(sch, "model_fields", {}) or {})


def _tool_required_names(tool: BaseTool) -> frozenset[str]:
    fields = _tool_model_fields(tool)
    return frozenset(n for n, f in fields.items() if f.is_required())


def _normalize_tool_call_args(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _legacy_tool_needs_resource_loop(tool: BaseTool) -> bool:
    """Heuristic when a tool has no args_schema (unusual for @tool)."""
    name = tool.name.lower()
    return any(
        kw in name
        for kw in (
            "resource_info",
            "item_count",
            "field",
            "sample",
            "statistics",
            "missing",
            "unique",
        )
    )


class Player:
    """
    A unified player agent capable of executing tasks and participating in debates.

    Attributes:
        name: Unique identifier for this player instance
        role_prompt: The persona/role description that guides the player's behavior
        tools: List of tools available to this player
        llm: The language model instance for this player
    """

    def __init__(
        self,
        name: str,
        role_prompt: str,
        tools: Optional[List[BaseTool]] = None,
        model_name: str = None,
        temperature: float = None,
        provider: str = None,
        tool_execution_mode: Optional[str] = None,
        max_tool_iterations: Optional[int] = None,
        role_key: Optional[str] = None,
    ):
        """
        Initialize a Player with a role and tools.

        Args:
            name: Unique identifier for this player
            role_prompt: Description of the player's role/persona
            tools: List of LangChain tools available to this player
            model_name: The LLM model to use (default from config)
            temperature: LLM temperature (default from config)
            provider: LLM provider to use (default from config)
            tool_execution_mode: "llm" (bind_tools loop) or "eager" (deterministic invokes).
            max_tool_iterations: Cap for LLM tool rounds (default from config).
            role_key: PLAYER_CONFIGS role name (e.g. spatial_temporal_specialist), if known.
        """
        temperature = temperature if temperature is not None else PLAYER_TEMPERATURE
        provider = provider or LLM_PROVIDER

        self.name = name
        self.role_prompt = role_prompt
        self.tools = tools or []
        self.llm = create_llm(
            model_name=model_name,
            temperature=temperature,
            provider=provider,
        )
        self._output_parser = StrOutputParser()
        self.tool_execution_mode = (
            tool_execution_mode
            if tool_execution_mode is not None
            else PLAYER_TOOL_EXECUTION_MODE
        )
        self.max_tool_iterations = (
            max_tool_iterations
            if max_tool_iterations is not None
            else PLAYER_MAX_TOOL_ITERATIONS
        )
        self.role_key = role_key

    def get_tool_manifest(self) -> str:
        """
        Generates a string manifest of the tools available to this player.
        Used by the orchestrator for planning.
        """
        if not self.tools:
            return f"Player: {self.name}\n  Description: {self.role_prompt}\n  Tools: None"

        manifest = f"Player: {self.name}\n"
        manifest += f"  Description: {self.role_prompt}\n"
        tasks = [f"{tool.name}: {tool.description}" for tool in self.tools]
        manifest += f"  Tools:\n" + "\n".join([f"    - {task}" for task in tasks])
        return manifest

    def _build_task_context_block(
        self,
        context_key: str,
        context_info: Dict[str, Any],
        is_multi_csv: bool,
        resources: List[str],
        target_resources: List[str],
    ) -> str:
        if is_multi_csv:
            ctx_info = f"Multi-CSV Context: {context_info.get('name', 'context')}\n"
            ctx_info += f"Context type: {context_info.get('context_type', 'unknown')}\n"
            ctx_info += f"Resources: {', '.join(resources)}\n"
            if target_resources:
                ctx_info += (
                    f"Target resources for this step: {', '.join(target_resources)}\n"
                )
            ctx_info += f"\ncontext_key for tools (injected if omitted): {context_key}"
        else:
            resource_name = resources[0] if resources else "unknown"
            ctx_info = f"Context: {context_info.get('name', 'context')}\n"
            ctx_info += f"Context type: {context_info.get('context_type', 'unknown')}\n"
            ctx_info += f"Resource: {resource_name}\n"
            ctx_info += f"\ncontext_key for tools (injected if omitted): {context_key}"
        return ctx_info

    def _execute_eager_tools(
        self,
        context_key: str,
        resources_to_analyze: List[str],
        tool_trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Deterministic tool invokes: only tools whose required args fit context_key/resource."""
        tool_results: Dict[str, Any] = {}

        for tool in self.tools:
            fields = _tool_model_fields(tool)
            if fields:
                required = _tool_required_names(tool)
                unsatisfied = required - {"context_key", "resource"}
                if unsatisfied:
                    entry = {
                        "tool": tool.name,
                        "skipped": True,
                        "reason": f"eager cannot supply required args: {sorted(unsatisfied)}",
                        "source": "eager",
                    }
                    tool_trace.append(entry)
                    continue

                has_resource_param = "resource" in fields
                try:
                    if has_resource_param and resources_to_analyze:
                        for resource in resources_to_analyze:
                            args: Dict[str, Any] = {
                                "context_key": context_key,
                                "resource": resource,
                            }
                            if tool.args_schema is not None:
                                args = tool.args_schema.model_validate(args).model_dump()
                            result = tool.invoke(args)
                            key = f"{resource}:{tool.name}"
                            tool_results[key] = result
                            tool_trace.append(
                                {
                                    "tool": tool.name,
                                    "args": args,
                                    "result": result,
                                    "source": "eager",
                                }
                            )
                    else:
                        args = {"context_key": context_key}
                        if tool.args_schema is not None:
                            args = tool.args_schema.model_validate(args).model_dump()
                        result = tool.invoke(args)
                        tool_results[tool.name] = result
                        tool_trace.append(
                            {
                                "tool": tool.name,
                                "args": args,
                                "result": result,
                                "source": "eager",
                            }
                        )
                except Exception as e:
                    tool_results[tool.name] = f"Error: {str(e)}"
                    tool_trace.append(
                        {"tool": tool.name, "error": str(e), "source": "eager"}
                    )
                continue

            # Legacy: no schema — keyword heuristics
            try:
                if _legacy_tool_needs_resource_loop(tool):
                    for resource in resources_to_analyze:
                        try:
                            result = tool.invoke(
                                {"context_key": context_key, "resource": resource}
                            )
                            tool_results[f"{resource}:{tool.name}"] = result
                            tool_trace.append(
                                {
                                    "tool": tool.name,
                                    "args": {
                                        "context_key": context_key,
                                        "resource": resource,
                                    },
                                    "result": result,
                                    "source": "eager",
                                }
                            )
                        except Exception as e:
                            try:
                                result = tool.invoke({"context_key": context_key})
                                tool_results[tool.name] = result
                                tool_trace.append(
                                    {
                                        "tool": tool.name,
                                        "args": {"context_key": context_key},
                                        "result": result,
                                        "source": "eager",
                                    }
                                )
                                break
                            except Exception:
                                tool_results[f"{resource}:{tool.name}"] = f"Error: {str(e)}"
                                tool_trace.append(
                                    {
                                        "tool": tool.name,
                                        "error": str(e),
                                        "source": "eager",
                                    }
                                )
                else:
                    result = tool.invoke({"context_key": context_key})
                    tool_results[tool.name] = result
                    tool_trace.append(
                        {
                            "tool": tool.name,
                            "args": {"context_key": context_key},
                            "result": result,
                            "source": "eager",
                        }
                    )
            except Exception as e:
                tool_results[tool.name] = f"Error: {str(e)}"
                tool_trace.append({"tool": tool.name, "error": str(e), "source": "eager"})

        return tool_results

    def _spatial_temporal_eager_extras(
        self,
        context_key: str,
        resources_to_analyze: List[str],
        tool_results: Dict[str, Any],
        tool_trace: List[Dict[str, Any]],
        max_cols: int = 5,
    ) -> None:
        """Chained detect → analyze/extent for spatial_temporal specialist (eager / fallback)."""
        by_name = {t.name: t for t in self.tools}
        det_t = by_name.get("detect_temporal_columns")
        det_s = by_name.get("detect_spatial_columns")
        ana_t = by_name.get("analyze_temporal_column")
        ana_s = by_name.get("analyze_spatial_column")
        ext_t = by_name.get("get_temporal_extent")
        ext_s = by_name.get("get_spatial_extent")
        ext_tuple = by_name.get("get_spatial_extent_from_tuple_column")

        for resource in resources_to_analyze:
            if det_t:
                try:
                    d = det_t.invoke({"context_key": context_key, "resource": resource})
                    key = f"{resource}:detect_temporal_columns:extra"
                    tool_results[key] = d
                    tool_trace.append(
                        {
                            "tool": "detect_temporal_columns",
                            "args": {"context_key": context_key, "resource": resource},
                            "result": d,
                            "source": "eager_extra",
                        }
                    )
                    cols = (d or {}).get("temporal_columns") or {}
                    picked: List[str] = []
                    for col_name, info in cols.items():
                        if info.get("detected_type") or info.get("name_suggests_temporal"):
                            picked.append(col_name)
                        if len(picked) >= max_cols:
                            break
                    if not picked:
                        picked = list(cols.keys())[:max_cols]
                    for col in picked:
                        if ana_t:
                            try:
                                out = ana_t.invoke(
                                    {
                                        "context_key": context_key,
                                        "resource": resource,
                                        "column": col,
                                    }
                                )
                                rk = f"{resource}:analyze_temporal_column:{col}"
                                tool_results[rk] = out
                                tool_trace.append(
                                    {
                                        "tool": "analyze_temporal_column",
                                        "args": {
                                            "context_key": context_key,
                                            "resource": resource,
                                            "column": col,
                                        },
                                        "result": out,
                                        "source": "eager_extra",
                                    }
                                )
                            except Exception as e:
                                tool_trace.append(
                                    {
                                        "tool": "analyze_temporal_column",
                                        "error": str(e),
                                        "source": "eager_extra",
                                    }
                                )
                        if ext_t:
                            try:
                                out = ext_t.invoke(
                                    {
                                        "context_key": context_key,
                                        "resource": resource,
                                        "time_column": col,
                                    }
                                )
                                rk = f"{resource}:get_temporal_extent:{col}"
                                tool_results[rk] = out
                                tool_trace.append(
                                    {
                                        "tool": "get_temporal_extent",
                                        "args": {
                                            "context_key": context_key,
                                            "resource": resource,
                                            "time_column": col,
                                        },
                                        "result": out,
                                        "source": "eager_extra",
                                    }
                                )
                            except Exception as e:
                                tool_trace.append(
                                    {
                                        "tool": "get_temporal_extent",
                                        "error": str(e),
                                        "source": "eager_extra",
                                    }
                                )
                except Exception as e:
                    tool_trace.append(
                        {"tool": "detect_temporal_columns", "error": str(e), "source": "eager_extra"}
                    )

            if det_s:
                try:
                    ds = det_s.invoke({"context_key": context_key, "resource": resource})
                    key = f"{resource}:detect_spatial_columns:extra"
                    tool_results[key] = ds
                    tool_trace.append(
                        {
                            "tool": "detect_spatial_columns",
                            "args": {"context_key": context_key, "resource": resource},
                            "result": ds,
                            "source": "eager_extra",
                        }
                    )
                    pairs = (ds or {}).get("detected_coordinate_pairs") or []
                    for pair in pairs[:3]:
                        lat_c = pair.get("latitude")
                        lon_c = pair.get("longitude")
                        if ext_s and lat_c and lon_c:
                            try:
                                out = ext_s.invoke(
                                    {
                                        "context_key": context_key,
                                        "resource": resource,
                                        "lat_column": lat_c,
                                        "lon_column": lon_c,
                                    }
                                )
                                rk = f"{resource}:get_spatial_extent:{lat_c}:{lon_c}"
                                tool_results[rk] = out
                                tool_trace.append(
                                    {
                                        "tool": "get_spatial_extent",
                                        "args": {
                                            "context_key": context_key,
                                            "resource": resource,
                                            "lat_column": lat_c,
                                            "lon_column": lon_c,
                                        },
                                        "result": out,
                                        "source": "eager_extra",
                                    }
                                )
                            except Exception as e:
                                tool_trace.append(
                                    {
                                        "tool": "get_spatial_extent",
                                        "error": str(e),
                                        "source": "eager_extra",
                                    }
                                )
                    for tcc in (ds or {}).get("tuple_coord_columns") or []:
                        col_t = tcc.get("column")
                        order_t = tcc.get("tuple_order", "lon_lat")
                        if ext_tuple and col_t:
                            try:
                                tout = ext_tuple.invoke(
                                    {
                                        "context_key": context_key,
                                        "resource": resource,
                                        "column": col_t,
                                        "tuple_order": order_t,
                                    }
                                )
                                rk = f"{resource}:get_spatial_extent_from_tuple_column:{col_t}"
                                tool_results[rk] = tout
                                tool_trace.append(
                                    {
                                        "tool": "get_spatial_extent_from_tuple_column",
                                        "args": {
                                            "context_key": context_key,
                                            "resource": resource,
                                            "column": col_t,
                                            "tuple_order": order_t,
                                        },
                                        "result": tout,
                                        "source": "eager_extra",
                                    }
                                )
                            except Exception as e:
                                tool_trace.append(
                                    {
                                        "tool": "get_spatial_extent_from_tuple_column",
                                        "error": str(e),
                                        "source": "eager_extra",
                                    }
                                )
                    spatial_cols = (ds or {}).get("spatial_columns") or {}
                    for col_name, info in list(spatial_cols.items())[:max_cols]:
                        if ana_s and info.get("detected_type"):
                            try:
                                out = ana_s.invoke(
                                    {
                                        "context_key": context_key,
                                        "resource": resource,
                                        "column": col_name,
                                    }
                                )
                                rk = f"{resource}:analyze_spatial_column:{col_name}"
                                tool_results[rk] = out
                                tool_trace.append(
                                    {
                                        "tool": "analyze_spatial_column",
                                        "args": {
                                            "context_key": context_key,
                                            "resource": resource,
                                            "column": col_name,
                                        },
                                        "result": out,
                                        "source": "eager_extra",
                                    }
                                )
                            except Exception as e:
                                tool_trace.append(
                                    {
                                        "tool": "analyze_spatial_column",
                                        "error": str(e),
                                        "source": "eager_extra",
                                    }
                                )
                except Exception as e:
                    tool_trace.append(
                        {"tool": "detect_spatial_columns", "error": str(e), "source": "eager_extra"}
                    )

    def _run_llm_tool_loop(
        self,
        context_key: str,
        task: str,
        target_resources: str,
        input_context: str,
        ctx_info: str,
        tool_descriptions: str,
    ) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]], str]]:
        """
        Optional model-driven tool calling. Returns None if bind_tools is unavailable.
        """
        if not self.tools:
            return {}, [], ""

        tools_by_name = {t.name: t for t in self.tools}
        try:
            llm_tools = self.llm.bind_tools(self.tools)
        except Exception as e:
            logging.warning("bind_tools failed for player %s: %s", self.name, e)
            return None

        system = f"""You are {self.name}. {self.role_prompt}

You may call tools when they help complete the task. Tools are optional — skip them if not needed.
When you call a tool, pass correct argument names and values (e.g. resource, column, lat_column, lon_column, time_column).
The execution environment always provides context_key; if you omit context_key it will be filled in for you.

Available tools:
{tool_descriptions}

{ctx_info}
"""
        human = f"""Task: {task}

Target resources for this step: {target_resources}

Input context from previous steps:
{input_context}

Use tools only when necessary, then give a clear analysis covering approach, findings, and results."""

        messages: List[Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]] = [
            SystemMessage(content=system),
            HumanMessage(content=human),
        ]
        tool_results: Dict[str, Any] = {}
        tool_trace: List[Dict[str, Any]] = []

        for iteration in range(self.max_tool_iterations):
            ai_msg: AIMessage = llm_tools.invoke(messages)
            messages.append(ai_msg)
            tcalls = list(getattr(ai_msg, "tool_calls", None) or [])
            if not tcalls:
                text = _stringify_message_content(ai_msg.content)
                return tool_results, tool_trace, text

            for idx, tc in enumerate(tcalls):
                name = tc.get("name")
                if not name and isinstance(tc.get("function"), dict):
                    name = tc["function"].get("name")
                raw_args = tc.get("args")
                if raw_args is None and isinstance(tc.get("function"), dict):
                    raw_args = tc["function"].get("arguments")
                args = _normalize_tool_call_args(raw_args)
                tid = tc.get("id") or f"iter{iteration}_tc{idx}"

                tool = tools_by_name.get(name or "")
                if tool is None:
                    err = f"Unknown tool: {name}"
                    tool_trace.append(
                        {"tool": name, "args": args, "error": err, "source": "llm"}
                    )
                    messages.append(
                        ToolMessage(
                            content=_serialize_tool_result({"error": err}),
                            tool_call_id=tid,
                        )
                    )
                    continue

                merged = dict(args)
                merged["context_key"] = context_key
                try:
                    if tool.args_schema is not None:
                        merged = tool.args_schema.model_validate(merged).model_dump()
                    out = tool.invoke(merged)
                except (ValidationError, Exception) as ex:
                    out = {"error": str(ex)}
                    tool_trace.append(
                        {
                            "tool": tool.name,
                            "args": merged,
                            "error": str(ex),
                            "source": "llm",
                        }
                    )
                else:
                    tool_trace.append(
                        {
                            "tool": tool.name,
                            "args": merged,
                            "result": out,
                            "source": "llm",
                        }
                    )

                trace_key = f"llm:{iteration}:{tool.name}:{tid}"
                tool_results[trace_key] = out
                messages.append(
                    ToolMessage(
                        content=_serialize_tool_result(out),
                        tool_call_id=tid,
                    )
                )

        final_nudge = HumanMessage(
            content="Provide your final analysis for the task. Do not request more tools."
        )
        final_ai = self.llm.invoke(messages + [final_nudge])
        text = _stringify_message_content(final_ai.content)
        tool_trace.append(
            {"note": "max_tool_iterations reached; final summary without bind_tools", "source": "llm"}
        )
        return tool_results, tool_trace, text

    def execute_task(
        self,
        task: str,
        context_key: str,
        context_info: Dict[str, Any],
        workspace: Dict[str, Any],
        inputs: Dict[str, str],
        target_resources: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a specific task using available tools.

        tool_execution_mode:
        - "llm": model chooses optional tool calls via bind_tools (args from model).
        - "eager": invoke each tool deterministically when schema allows context_key/resource only.

        Returns tool_trace for debugging (optional tools / skips / errors).
        """
        resolved_inputs: Dict[str, Any] = {}
        for param_name, artifact_name in inputs.items():
            if artifact_name in workspace:
                resolved_inputs[param_name] = workspace[artifact_name]
            else:
                resolved_inputs[param_name] = f"[MISSING: {artifact_name}]"

        tool_descriptions = (
            "\n".join(
                f"- {tool.name}: {tool.description}" for tool in self.tools
            )
            if self.tools
            else "No tools available."
        )

        is_multi_csv = context_info.get("is_multi_csv", False)
        resources = list(context_info.get("resources", []) or [])
        target_resources = target_resources or []

        ctx_info = self._build_task_context_block(
            context_key, context_info, is_multi_csv, resources, target_resources
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are {self.name}. {self.role_prompt}

You have access to the following tools:
{tool_descriptions}

Your task is to analyze the context and provide a detailed response.

{ctx_info}

For multi-CSV contexts (multiple CSV resources), consider:
- How resources might relate to each other
- Common fields that could be foreign keys
- Data integrity across resources
""",
                ),
                (
                    "human",
                    """Task: {task}

Target resources for this step: {target_resources}

Input context from previous steps:
{input_context}

Execute this task and provide a comprehensive response. Include:
1. Your approach to the task
2. Any relevant observations or findings
3. The result of your analysis""",
                ),
            ]
        )

        input_context = (
            "\n".join(f"- {k}: {v}" for k, v in resolved_inputs.items())
            if resolved_inputs
            else "No inputs from previous steps."
        )

        chain = prompt | self.llm | self._output_parser

        if target_resources:
            resources_to_analyze = list(target_resources)
        else:
            resources_to_analyze = list(resources)

        tool_trace: List[Dict[str, Any]] = []
        llm_result: Optional[Tuple[Dict[str, Any], List[Dict[str, Any]], str]] = None

        mode = (self.tool_execution_mode or "eager").lower().strip()
        if mode == "llm" and self.tools:
            llm_result = self._run_llm_tool_loop(
                context_key=context_key,
                task=task,
                target_resources=", ".join(target_resources)
                if target_resources
                else ("All resources" if is_multi_csv else "N/A"),
                input_context=input_context,
                ctx_info=ctx_info,
                tool_descriptions=tool_descriptions,
            )

        if llm_result is not None:
            tool_results, trace_llm, analysis_text = llm_result
            tool_trace.extend(trace_llm)
            if not analysis_text.strip():
                analysis_text = chain.invoke(
                    {
                        "task": task,
                        "target_resources": ", ".join(target_resources)
                        if target_resources
                        else ("All resources" if is_multi_csv else "N/A"),
                        "input_context": input_context
                        + "\n\nTool Results:\n"
                        + str(tool_results),
                    }
                )
            return {
                "player": self.name,
                "task": task,
                "tool_results": tool_results,
                "tool_trace": tool_trace,
                "analysis": analysis_text,
                "success": True,
                "is_multi_csv": is_multi_csv,
            }

        tool_results = self._execute_eager_tools(
            context_key, resources_to_analyze, tool_trace
        )
        if self.role_key == "spatial_temporal_specialist":
            self._spatial_temporal_eager_extras(
                context_key, resources_to_analyze, tool_results, tool_trace
            )

        target_info = (
            ", ".join(target_resources)
            if target_resources
            else ("All resources" if is_multi_csv else "N/A")
        )
        llm_response = chain.invoke(
            {
                "task": task,
                "target_resources": target_info,
                "input_context": input_context + "\n\nTool Results:\n" + str(tool_results),
            }
        )

        out: Dict[str, Any] = {
            "player": self.name,
            "task": task,
            "tool_results": tool_results,
            "tool_trace": tool_trace,
            "analysis": llm_response,
            "success": True,
            "is_multi_csv": is_multi_csv,
        }
        if mode == "llm" and self.tools and llm_result is None:
            out["tool_execution_fallback"] = (
                "eager_after_bind_tools_failed_or_unsupported"
            )
        return out

    def generate_initial_work(
        self,
        task: str,
        context_info: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        """
        Generate initial work/analysis for a debate round.

        Args:
            task: The task to work on
            context_info: Info about the ExecutionContext
            context: Additional context (workspace, tool results, etc.)

        Returns:
            The player's initial analysis as a string
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are {self.name}. {self.role_prompt}

You are participating in a multi-agent analysis of a context (dataset, API, etc.).
Your goal is to provide your unique perspective and insights.""",
                ),
                (
                    "human",
                    """Task: {task}

Context: {context_name} ({context_type})
Resources: {resources}

Context and available information:
{context}

Provide your initial analysis. Be thorough and specific.
Focus on what you can contribute based on your role.""",
                ),
            ]
        )

        chain = prompt | self.llm | self._output_parser

        return chain.invoke(
            {
                "task": task,
                "context_name": context_info.get("name", "context"),
                "context_type": context_info.get("context_type", "unknown"),
                "resources": ", ".join(context_info.get("resources", [])),
                "context": str(context),
            }
        )

    def critique_work(self, task: str, other_players_work: Dict[str, str]) -> str:
        """
        Critique the work of other players.

        Args:
            task: The task being worked on
            other_players_work: Dictionary mapping player names to their work

        Returns:
            Critique as a string
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are {self.name}. {self.role_prompt}

You are reviewing the work of other analysts. Provide constructive criticism
that helps improve the overall analysis. Be specific about what could be
improved, what's missing, or what might be incorrect.""",
                ),
                (
                    "human",
                    """Task: {task}

Work from other players to critique:
{other_work}

Provide your critique. Focus on:
1. Accuracy and correctness
2. Completeness
3. Clarity and specificity
4. Suggestions for improvement""",
                ),
            ]
        )

        chain = prompt | self.llm | self._output_parser

        other_work_str = "\n\n".join(
            f"=== {name} ===\n{work}" for name, work in other_players_work.items()
        )

        return chain.invoke({"task": task, "other_work": other_work_str})

    def revise_work(
        self,
        task: str,
        my_original_work: str,
        critiques: List[str],
    ) -> str:
        """
        Revise work based on critiques received.

        Args:
            task: The task being worked on
            my_original_work: This player's original work
            critiques: List of critiques from other players

        Returns:
            Revised work as a string
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are {self.name}. {self.role_prompt}

You are revising your work based on feedback from other analysts.
Incorporate valid criticisms while maintaining your unique perspective.""",
                ),
                (
                    "human",
                    """Task: {task}

Your original work:
{original_work}

Critiques received:
{critiques}

Provide your revised analysis. Address the valid points raised in the critiques
while maintaining accuracy and your analytical perspective.""",
                ),
            ]
        )

        chain = prompt | self.llm | self._output_parser

        critiques_str = "\n\n".join(
            f"Critique {i + 1}:\n{c}" for i, c in enumerate(critiques)
        )

        return chain.invoke(
            {
                "task": task,
                "original_work": my_original_work,
                "critiques": critiques_str,
            }
        )

    def synthesize_results(
        self,
        task: str,
        all_results: List[Dict[str, Any]],
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        """
        Synthesize multiple results into a consolidated output.
        Uses this player's role/expertise to consolidate debate results.

        Args:
            task: The task that was worked on
            all_results: List of results from all players
            output_schema: Optional Pydantic model class for structured output.
                          If provided, returns a validated Pydantic model instance.
                          If None, returns a string (legacy behavior).

        Returns:
            Synthesized result as a string or Pydantic model instance
        """
        results_str = "\n\n".join(
            f"=== {r.get('player', 'Unknown')} ===\n{r.get('analysis', str(r))}"
            for r in all_results
        )

        if output_schema is not None:
            return self._synthesize_structured(task, results_str, output_schema)
        return self._synthesize_string(task, results_str)

    def _synthesize_string(self, task: str, results_str: str) -> str:
        """Synthesize results as a string (legacy behavior)."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are {self.name}. {self.role_prompt}

You are now synthesizing results from multiple analysts who worked on the same task.

**Your job:**
- Consolidate the findings into a single, authoritative result
- Resolve any conflicts by choosing the most accurate/complete information
- Preserve important details while removing redundancy
- Output a clear, concise result appropriate for the task

**Output requirements:**
- Output ONLY the consolidated result
- NO meta-commentary like "Based on the analyses..." or "The players found..."
- NO explanations of your synthesis process
- Keep the format appropriate for the task (e.g., numbers for counts, lists for columns)""",
                ),
                (
                    "human",
                    """Task: {task}

Results from all analysts:
{all_results}

Provide the consolidated result for this task. Output only the result, no commentary.""",
                ),
            ]
        )

        chain = prompt | self.llm | self._output_parser

        return chain.invoke({"task": task, "all_results": results_str})

    def _synthesize_structured(
        self,
        task: str,
        results_str: str,
        output_schema: Type[BaseModel],
    ) -> BaseModel:
        """
        Synthesize results into a structured Pydantic model.

        Uses LangChain's with_structured_output() for guaranteed schema compliance.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are {self.name}. {self.role_prompt}

You are synthesizing results from multiple analysts into a structured format.

**Your job:**
- Extract and consolidate all relevant information from the analyses
- Fill in ALL fields in the schema with concrete values from the gathered information
- Use null/None for fields where information is truly unavailable
- Resolve conflicts by choosing the most accurate/complete information

**CRITICAL:**
- Output MUST conform exactly to the provided schema
- Use actual values, not placeholders like "..."
- Be specific and concrete""",
                ),
                (
                    "human",
                    """Task: {task}

Results from all analysts:
{all_results}

Generate the final structured output.""",
                ),
            ]
        )

        structured_llm = self.llm.with_structured_output(output_schema)
        chain = prompt | structured_llm

        return chain.invoke({"task": task, "all_results": results_str})

    def __repr__(self):
        return f"Player(name={self.name}, tools={len(self.tools)})"


def create_player_from_config(
    config: Dict[str, Any],
    name: str,
    provider: str = None,
    role_key: Optional[str] = None,
) -> Player:
    """
    Factory function to create a Player from a configuration dictionary.

    Args:
        config: Dictionary with 'role_prompt', 'tools', and optional 'model_name', 'temperature'
        name: The name to assign to this player instance
        provider: LLM provider to use (default from config)

    Returns:
        Configured Player instance
    """
    rk = role_key if role_key is not None else config.get("_role_key")
    return Player(
        name=name,
        role_prompt=config.get("role_prompt", "You are a helpful analyst."),
        tools=config.get("tools", []),
        model_name=config.get("model_name"),
        temperature=config.get("temperature"),
        provider=provider,
        tool_execution_mode=config.get("tool_execution_mode"),
        max_tool_iterations=config.get("max_tool_iterations"),
        role_key=rk,
    )
