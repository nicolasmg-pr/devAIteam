import json
import httpx
from typing import List, Dict, Any, Type
from langchain_core.tools import StructuredTool
from pydantic import create_model, Field

def summarize_stitch_response(text: str) -> str:
    """Summarizes large Stitch JSON responses to prevent context window overflow."""
    text_stripped = text.strip()
    if not text_stripped:
        return ""
        
    # Check if it looks like JSON
    if not (text_stripped.startswith("{") or text_stripped.startswith("[")):
        # Truncate raw text if it's very long
        if len(text_stripped) > 1000:
            return text_stripped[:1000] + "\n... [TRUNCATED due to length] ..."
        return text_stripped
        
    try:
        data = json.loads(text_stripped)
        
        def clean_val(val: Any) -> Any:
            if isinstance(val, dict):
                cleaned = {}
                for k, v in val.items():
                    # Keep important identifiers, names, descriptions, and statuses
                    if k in ["name", "projectId", "sessionId", "title", "id", "visibility", "projectType", "description", "status", "type", "screenshot", "stitch_screen_url", "screen_url", "url", "uri", "html", "html_content", "design_system_notes"]:
                        cleaned[k] = clean_val(v)
                    elif k in ["screens", "outputComponents", "components"] and isinstance(v, list):
                        cleaned[k] = [clean_val(item) for item in v]
                    elif k == "design" and isinstance(v, dict):
                        cleaned[k] = clean_val(v)
                return cleaned
            elif isinstance(val, list):
                return [clean_val(item) for item in val]
            else:
                return val

        cleaned_data = clean_val(data)
        
        # If the cleaned object is empty or trivial, return a list of keys
        if not cleaned_data or (isinstance(cleaned_data, dict) and len(cleaned_data) == 0):
            if isinstance(data, dict):
                return f"JSON object with keys: {list(data.keys())}"
            else:
                return f"JSON list of length {len(data)}"
                
        return json.dumps(cleaned_data, indent=2, ensure_ascii=False)
    except Exception:
        if len(text_stripped) > 1000:
            return text_stripped[:1000] + "\n... [TRUNCATED due to length] ..."
        return text_stripped

def describe_schema(schema: dict, defs: dict, indent: str = "  ", visited: set = None) -> str:
    """Generates a human-readable text description of a JSON schema."""
    if visited is None:
        visited = set()
    if not isinstance(schema, dict):
        return str(schema)
        
    if "$ref" in schema:
        ref_path = schema["$ref"]
        ref_name = ref_path.split("/")[-1]
        if ref_name in visited:
            return f"Ref({ref_name}) [recursive]"
        visited.add(ref_name)
        if ref_name in defs:
            res = describe_schema(defs[ref_name], defs, indent, visited)
            visited.remove(ref_name)
            return res
        return f"Ref({ref_name})"
        
    type_str = schema.get("type", "any")
    desc = schema.get("description", "")
    
    if type_str == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        lines = []
        if desc:
            lines.append(f"{desc}")
        lines.append("{")
        for prop_name, prop_details in properties.items():
            req_marker = " (required)" if prop_name in required else ""
            prop_desc = prop_details.get("description", "").replace("\n", " ")
            prop_type = prop_details.get("type")
            if not prop_type and "$ref" in prop_details:
                ref_desc = describe_schema(prop_details, defs, indent + "  ", visited)
                lines.append(f"{indent}{prop_name}{req_marker}: {ref_desc}")
            else:
                enum_vals = prop_details.get("enum")
                enum_str = f" [Enum: {enum_vals}]" if enum_vals else ""
                lines.append(f"{indent}{prop_name}{req_marker}: {prop_type}{enum_str} - {prop_desc}")
        lines.append(indent[:-2] + "}")
        return "\n".join(lines)
        
    elif type_str == "array":
        items = schema.get("items", {})
        item_desc = describe_schema(items, defs, indent, visited)
        return f"List of [\n{indent}{item_desc}\n{indent[:-2]}]"
        
    else:
        enum_vals = schema.get("enum")
        enum_str = f" [Enum: {enum_vals}]" if enum_vals else ""
        return f"{type_str}{enum_str} - {desc}"

def resolve_schema(schema: dict, defs: dict) -> dict:
    while isinstance(schema, dict) and "$ref" in schema:
        ref_path = schema["$ref"]
        ref_name = ref_path.split("/")[-1]
        if ref_name in defs:
            schema = defs[ref_name]
        else:
            break
    return schema

def clean_value_by_schema(val: Any, schema: dict, defs: dict) -> Any:
    if not isinstance(schema, dict):
        return val
        
    schema = resolve_schema(schema, defs)
    
    # Check if there is an enum
    if "enum" in schema and isinstance(val, str):
        enum_vals = schema["enum"]
        for enum_val in enum_vals:
            if isinstance(enum_val, str) and val.strip().upper() == enum_val.strip().upper():
                return enum_val
        return val
        
    type_str = schema.get("type")
    
    if type_str == "object" and isinstance(val, dict):
        properties = schema.get("properties", {})
        cleaned_obj = {}
        for k, v in val.items():
            if k in properties:
                cleaned_obj[k] = clean_value_by_schema(v, properties[k], defs)
            else:
                cleaned_obj[k] = v # Keep unknown props as is
        return cleaned_obj
        
    elif type_str == "array" and isinstance(val, list):
        items_schema = schema.get("items", {})
        return [clean_value_by_schema(item, items_schema, defs) for item in val]
        
    return val

class StitchMCPClient:
    """A stateless HTTP JSON-RPC client for Google Stitch MCP."""

    def __init__(self, api_key: str, endpoint: str = "https://stitch.googleapis.com/mcp"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        self.client = httpx.Client(timeout=300.0)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        try:
            self.client.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _rpc_call(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request to the MCP endpoint.

        Never log the api_key, headers, full params, or full response bodies, as
        they may contain secrets or large/sensitive payloads.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        response = self.client.post(self.endpoint, headers=self.headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise Exception(f"MCP RPC Error ({method}): {data['error']}")
        return data.get("result", {})

    def get_tools(self) -> List[StructuredTool]:
        """Fetch available tools from Stitch MCP and wrap them as LangChain tools."""
        result = self._rpc_call("tools/list", {})
        tools = result.get("tools", [])
        
        langchain_tools = []
        for tool_def in tools:
            name = tool_def["name"]
            description = tool_def.get("description", "")
            input_schema = tool_def.get("inputSchema", {})
            defs = input_schema.get("$defs", input_schema.get("definitions", {}))
            
            # Dynamically create Pydantic model for tool arguments
            fields = {}
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])
            
            for prop_name, prop_details in properties.items():
                prop_type_str = prop_details.get("type")
                if not prop_type_str:
                    if "$ref" in prop_details:
                        prop_type_str = "object"
                    else:
                        prop_type_str = "string"
                
                prop_desc = prop_details.get("description", "")
                if "$ref" in prop_details or prop_type_str == "object":
                    schema_desc = describe_schema(prop_details, defs)
                    prop_desc = f"{prop_desc}\nSchema structure:\n{schema_desc}"
                
                # Simple type mapping
                python_type = str
                if prop_type_str == "integer":
                    python_type = int
                elif prop_type_str == "boolean":
                    python_type = bool
                elif prop_type_str == "number":
                    python_type = float
                elif prop_type_str == "array":
                    python_type = list
                elif prop_type_str == "object":
                    python_type = dict
                    
                is_req = prop_name in required
                default = ... if is_req else None
                fields[prop_name] = (python_type, Field(default=default, description=prop_desc))

            # Create dynamic Pydantic args schema
            args_schema = create_model(f"{name}_args", **fields)
            
            # Create the closure for the tool function
            def _create_func(tool_name: str, schema: dict, defs: dict):
                def _tool_func(**kwargs) -> str:
                    # Filter out None values to prevent API validation errors
                    filtered_args = {k: v for k, v in kwargs.items() if v is not None}
                    # Clean arguments using schema to align casing of enum values
                    cleaned_args = clean_value_by_schema(filtered_args, schema, defs)
                    res = self._rpc_call("tools/call", {
                        "name": tool_name,
                        "arguments": cleaned_args
                    })
                    # Format output text for the agent
                    content_blocks = res.get("content", [])
                    output_text = ""
                    for block in content_blocks:
                        if block.get("type") == "text":
                            output_text += block.get("text", "") + "\n"
                    raw_out = output_text.strip()
                    summarized = summarize_stitch_response(raw_out)
                    return summarized
                return _tool_func

            # Create the StructuredTool
            lc_tool = StructuredTool.from_function(
                func=_create_func(name, input_schema, defs),
                name=name,
                description=description,
                args_schema=args_schema,
            )
            langchain_tools.append(lc_tool)
            
        return langchain_tools
