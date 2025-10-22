#!/usr/bin/env python3

import os
import re
import json
import math
import time
import traceback
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from typing import Optional

import boto3
import pandas as pd
import matplotlib.pyplot as plt

# strands imports
from strands import tool
from strands.models import BedrockModel
from strands import Agent
from strands_tools import use_aws

# load env
load_dotenv()

# boto3 session (uses env or IAM role)
session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("REGION_NAME"),
)

HISTORY_FILE = "aws_agent_history.json"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Simple mapping from friendly names to regions
REGION_MAP = {
    "oregon": "us-west-2",
    "virginia": "us-east-1",
    "ohio": "us-east-2",
    "ireland": "eu-west-1",
    "london": "eu-west-2",
    "frankfurt": "eu-central-1",
    "paris": "eu-west-3",
    "tokyo": "ap-northeast-1",
    "seoul": "ap-northeast-2",
    "singapore": "ap-southeast-1",
    "sydney": "ap-southeast-2",
    "mumbai": "ap-south-1",
    "saopaulo": "sa-east-1",
    "canada": "ca-central-1",
}

def normalize_region(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = text.lower()
    # explicit code like us-west-2
    m = re.search(r"\b([a-z]{2}-[a-z]+-\d)\b", text)
    if m:
        return m.group(1)
    for k, v in REGION_MAP.items():
        if k in text:
            return v
    return None

def parse_intent(text: str) -> str:
    """Return 'list' | 'metrics' — choose metrics only when clear keywords present."""
    t = text.lower()
    # metrics keywords
    metrics_kw = ("cloudwatch", "metric", "metrics", "cpu", "utilization", "monitor", "monitoring", "load")
    for kw in metrics_kw:
        if kw in t:
            return "metrics"
    # list keywords
    list_kw = ("list", "describe", "show instances", "instance names", "instance name", "what are", "which instances")
    for kw in list_kw:
        if kw in t:
            return "list"
    # fallback: if user mentions "plot" or "graph", treat as metrics
    if any(k in t for k in ("plot", "graph", "chart")):
        return "metrics"
    # default to list (safer)
    return "list"

def extract_instance_id(text: str) -> Optional[str]:
    m = re.search(r"\b(i-[0-9a-fA-F]{8,17})\b", text)
    return m.group(1) if m else None

def extract_ordinal(text: str) -> Optional[int]:
    m = re.search(r"\b(?:instance|inst|#)\s*(?:no\.?|number)?\s*(\d{1,3})\b", text, re.I)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None

def read_history_instance_ids() -> list:
    """Read HISTORY_FILE and extract instance IDs in chronological order (oldest -> newest)."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            hist = json.load(f)
    except Exception:
        # fall back to empty
        return []
    text_blob = json.dumps(hist)
    ids = [m.group(1) for m in re.finditer(r"\b(i-[0-9a-fA-F]{8,17})\b", text_blob)]
    # deduplicate preserving order
    seen = set(); out=[]
    for i in ids:
        if i not in seen:
            seen.add(i); out.append(i)
    return out

def df_from_datapoints(datapoints, stat_key="Average"):
    rows=[]
    for dp in datapoints:
        ts = dp.get("Timestamp")
        try:
            ts_parsed = pd.to_datetime(ts)
        except:
            ts_parsed = pd.to_datetime(str(ts))
        rows.append({"timestamp": ts_parsed, stat_key: dp.get(stat_key)})
    if not rows:
        return pd.DataFrame(columns=["timestamp", stat_key]).set_index("timestamp")
    df = pd.DataFrame(rows).sort_values("timestamp").set_index("timestamp")
    return df

@tool
def aws_tool(user_text: str, prefer_plot: bool = False):
    """
    Single tool that inspects user_text and performs either:
      - listing instances (describe_instances), or
      - fetching CloudWatch CPUUtilization and saving a PNG plot.

    Returns a JSON-serializable dict with Status and result fields.
    The agent can call this tool with the raw user message.
    """
    try:
        intent = parse_intent(user_text)
        region = normalize_region(user_text) or getattr(session, "region_name", None) or os.getenv("REGION_NAME")
        # default to us-east-1 if nothing
        if not region:
            region = os.getenv("REGION_NAME") or "us-east-1"

        # ---- LIST INSTANCES ----
        if intent == "list":
            ec2 = session.client("ec2", region_name=region)
            paginator = ec2.get_paginator("describe_instances")
            instances=[]
            for page in paginator.paginate():
                for r in page.get("Reservations", []):
                    for inst in r.get("Instances", []):
                        instances.append({
                            "InstanceId": inst.get("InstanceId"),
                            "State": inst.get("State", {}).get("Name"),
                            "InstanceType": inst.get("InstanceType"),
                            "Name": next((t.get("Value") for t in inst.get("Tags", []) if t.get("Key")=="Name"), None),
                            "PrivateIpAddress": inst.get("PrivateIpAddress"),
                            "PublicIpAddress": inst.get("PublicIpAddress")
                        })
            # Save a short summary into history so ordinal mapping (instance 4) works later
            try:
                # append this tool output to HISTORY_FILE
                hist = []
                if os.path.exists(HISTORY_FILE):
                    with open(HISTORY_FILE, "r") as f:
                        hist = json.load(f)
                hist.append({"time": datetime.now(timezone.utc).isoformat(), "action":"list_instances", "region":region, "instances": instances})
                with open(HISTORY_FILE, "w") as f:
                    json.dump(hist, f, default=str, indent=2)
            except Exception:
                pass

            # concise return
            return {"Status":"OK", "Action":"list", "Region":region, "Count": len(instances), "Instances": instances}

        # ---- METRICS (plot) ----
        else:
            # try to get explicit instance id
            inst = extract_instance_id(user_text)
            if not inst:
                # maybe ordinal like "instance 4" — map from history
                idx = extract_ordinal(user_text)
                if idx:
                    ids = read_history_instance_ids()
                    if ids and 1 <= idx <= len(ids):
                        inst = ids[idx-1]
            if not inst:
                return {"Status":"Error", "Message":"No instance id found in the user's message or history. Provide an instance id (i-...) or run 'list instances' first."}

            # time window default 24 hours; try to parse explicit 'last 6 hours' etc.
            duration_value = 24
            duration_unit = "hours"
            m = re.search(r"(\d+)\s*(hours|hour|minutes|min|hrs|hr)", user_text, re.I)
            if m:
                duration_value = int(m.group(1))
                unit_raw = m.group(2).lower()
                duration_unit = "minutes" if "min" in unit_raw else "hours"

            # CloudWatch call
            now = datetime.now(timezone.utc)
            delta = timedelta(hours=duration_value) if duration_unit=="hours" else timedelta(minutes=duration_value)
            start_time = now - delta
            end_time = now
            total_seconds = int(delta.total_seconds())
            approx_period = max(60, math.ceil(total_seconds / 100))
            period = int(math.ceil(approx_period / 60.0) * 60)
            if period > total_seconds and total_seconds >= 60:
                period = int(math.floor(total_seconds / 60.0) * 60) or 60

            cw = session.client("cloudwatch", region_name=region)
            resp = cw.get_metric_statistics(
                Namespace="AWS/EC2",
                MetricName="CPUUtilization",
                Dimensions=[{"Name":"InstanceId","Value":inst}],
                StartTime=start_time,
                EndTime=end_time,
                Period=period,
                Statistics=["Average"]
            )
            dps = resp.get("Datapoints", [])
            dps.sort(key=lambda d: d.get("Timestamp"))

            if not dps:
                return {"Status":"Error", "Message":"No datapoints returned for this instance in the requested window.", "InstanceId": inst}

            # build dataframe and save plot (matplotlib)
            df = df_from_datapoints(dps, stat_key="Average")
            # filename
            ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            safe_fn = f"{inst}_{ts}.png"
            out_path = os.path.join(PLOT_DIR, safe_fn)

            # simple plot
            plt.figure(figsize=(10,4))
            plt.plot(df.index, df["Average"], marker="o", linestyle="-")
            plt.title(f"CPU Average — {inst} ({region})")
            plt.xlabel("Time (UTC)")
            plt.ylabel("CPU % (Average)")
            plt.grid(True)
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()

            # Save to history for mapping later
            try:
                hist = []
                if os.path.exists(HISTORY_FILE):
                    with open(HISTORY_FILE, "r") as f:
                        hist = json.load(f)
                hist.append({"time": datetime.now(timezone.utc).isoformat(), "action":"metrics", "region":region, "instance":inst, "plot":out_path})
                with open(HISTORY_FILE, "w") as f:
                    json.dump(hist, f, default=str, indent=2)
            except Exception:
                pass

            # summary stats
            values = [d.get("Average") for d in dps if d.get("Average") is not None]
            summary = {
                "count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "avg": (sum(values)/len(values)) if values else None
            }

            return {"Status":"OK", "Action":"metrics", "InstanceId": inst, "Region": region, "PlotPath": out_path, "Stats": summary}

    except Exception as e:
        return {"Status":"Error", "Message": f"{type(e).__name__}: {str(e)}", "Trace": traceback.format_exc()}

# --- Create Bedrock model & agent (same as before) ---
SYSTEM_PROMPT = """
You are a helpful AWS cloud assistant. Use the provided tool 'aws_tool' for AWS operations.
Pass the user's raw message to 'aws_tool' when appropriate.
"""

bedrock_model = BedrockModel(model_id="us.amazon.nova-pro-v1:0", temperature=0.7, top_p=0.9)

agent = Agent(
    model=bedrock_model,
    system_prompt=SYSTEM_PROMPT,
    tools=[use_aws, aws_tool],
    messages=[]
)

print("🧠 AWS Assistant ready. Type your command (or 'exit')\n")

# --- interactive loop ---
try:
    while True:
        txt = input("You: ").strip()
        if not txt:
            continue
        if txt.lower() in ("exit","quit"):
            print("Goodbye.")
            break

        # Call the agent with the user's message. The agent may call aws_tool internally.
        try:
            out = agent(txt)
        except Exception as e:
            # If agent fails, call the tool directly as last resort
            print("Agent error:", type(e).__name__, str(e))
            print("Calling aws_tool directly as fallback...")
            out = aws_tool(txt, prefer_plot=True)

        # print concise result
        if isinstance(out, (dict, list)):
            print("Assistant:", json.dumps(out, indent=2, default=str))
        else:
            print("Assistant:", out)

except KeyboardInterrupt:
    print("\nExiting...")

