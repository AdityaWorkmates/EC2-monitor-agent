#!/usr/bin/env python3

import os
import re
import json
import math
import traceback
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from typing import Optional

import boto3
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import logging

# strands imports (keep your original strands setup)
from strands import tool
from strands.models import BedrockModel
from strands import Agent
from strands_tools import use_aws

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("aws_agent_streamlit")

# load env
load_dotenv()

# For deploying in EC2 or ECS

# boto3 session (uses env or IAM role)
# session = boto3.Session(
#     aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
#     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#     region_name=os.getenv("REGION_NAME"),
# )

# For deploying in streamlit

# boto3 session (uses env or IAM role)
session = boto3.Session(
    aws_access_key_id=st.secrets.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=st.secrets.get("AWS_SECRET_ACCESS_KEY"),
    region_name=st.secrets.get("AWS_DEFAULT_REGION", "us-west-2"),
)


HISTORY_FILE = "aws_agent_history.json"
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

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
    m = re.search(r"\b([a-z]{2}-[a-z]+-\d)\b", text)
    if m:
        result = m.group(1)
        logger.debug(f"Normalized region from '{text}' to '{result}'")
        return result
    for k, v in REGION_MAP.items():
        if k in text:
            logger.debug(f"Normalized region from '{text}' to '{v}'")
            return v
    return None


def parse_intent(text: str) -> str:
    t = text.lower()
    metrics_kw = ("cloudwatch", "metric", "metrics", "cpu", "utilization", "monitor", "monitoring", "load")
    for kw in metrics_kw:
        if kw in t:
            intent = "metrics"
            logger.debug(f"Parsed intent '{intent}' from text '{text}'")
            return intent
    list_kw = ("list", "describe", "show instances", "instance names", "instance name", "what are", "which instances")
    for kw in list_kw:
        if kw in t:
            intent = "list"
            logger.debug(f"Parsed intent '{intent}' from text '{text}'")
            return intent
    if any(k in t for k in ("plot", "graph", "chart")):
        intent = "metrics"
        logger.debug(f"Parsed intent '{intent}' from text '{text}'")
        return intent
    return "list"


def extract_instance_id(text: str) -> Optional[str]:
    m = re.search(r"\b(i-[0-9a-fA-F]{8,17})\b", text)
    instance_id = m.group(1) if m else None
    logger.debug(f"Extracted instance id '{instance_id}' from text '{text}'")
    return instance_id


def extract_ordinal(text: str) -> Optional[int]:
    m = re.search(r"\b(?:instance|inst|#)\s*(?:no\.?|number)?\s*(\d{1,3})\b", text, re.I)
    if m:
        try:
            ordinal = int(m.group(1))
            logger.debug(f"Extracted ordinal '{ordinal}' from text '{text}'")
            return ordinal
        except:
            return None
    return None


def read_history_instance_ids() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            hist = json.load(f)
    except Exception:
        return []
    text_blob = json.dumps(hist)
    ids = [m.group(1) for m in re.finditer(r"\b(i-[0-9a-fA-F]{8,17})\b", text_blob)]
    seen = set(); out=[]
    for i in ids:
        if i not in seen:
            seen.add(i); out.append(i)
    logger.info("Reading history instance ids")
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
    logger.debug(f"Created dataframe from {len(datapoints)} datapoints")
    return df


@tool
def cloudwatch_metrics_tool(user_text: str, prefer_plot: bool = False):
    logger.info(f"cloudwatch_metrics_tool called with user_text: {user_text}")
    try:
        intent = parse_intent(user_text)
        region = normalize_region(user_text) or getattr(session, "region_name", None) or os.getenv("REGION_NAME")
        if not region:
            region = os.getenv("REGION_NAME") or "us-east-1"

        if intent == "list":
            logger.info(f"Listing instances in region {region}")
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
            logger.info(f"Found {len(instances)} instances")

            try:
                hist = []
                if os.path.exists(HISTORY_FILE):
                    with open(HISTORY_FILE, "r") as f:
                        hist = json.load(f)
                hist.append({"time": datetime.now(timezone.utc).isoformat(), "action":"list_instances", "region":region, "instances": instances})
                with open(HISTORY_FILE, "w") as f:
                    json.dump(hist, f, default=str, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write history: {e}")

            return {"Status":"OK", "Action":"list", "Region":region, "Count": len(instances), "Instances": instances}
        else:
            inst = extract_instance_id(user_text)
            if not inst:
                idx = extract_ordinal(user_text)
                if idx:
                    ids = read_history_instance_ids()
                    if ids and 1 <= idx <= len(ids):
                        inst = ids[idx-1]
            if not inst:
                logger.error("No instance id found in user message or history")
                return {"Status":"Error", "Message":"No instance id found in the user's message or history. Provide an instance id (i-...) or run 'list instances' first."}

            duration_value = 24
            duration_unit = "hours"
            m = re.search(r"(\d+)\s*(hours|hour|minutes|min|hrs|hr)", user_text, re.I)
            if m:
                duration_value = int(m.group(1))
                unit_raw = m.group(2).lower()
                duration_unit = "minutes" if "min" in unit_raw else "hours"

            now = datetime.now(timezone.utc)
            delta = timedelta(hours=duration_value) if duration_unit=="hours" else timedelta(minutes=duration_value)
            start_time = now - delta
            end_time = now
            total_seconds = int(delta.total_seconds())
            approx_period = max(60, math.ceil(total_seconds / 100))
            period = int(math.ceil(approx_period / 60.0) * 60)
            if period > total_seconds and total_seconds >= 60:
                period = int(math.floor(total_seconds / 60.0) * 60) or 60

            logger.info(f"Getting metrics for instance {inst} in region {region}")
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
                logger.warning(f"No datapoints returned for instance {inst}")
                return {"Status":"Error", "Message":"No datapoints returned for this instance in the requested window.", "InstanceId": inst}

            df = df_from_datapoints(dps, stat_key="Average")
            ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            safe_fn = f"{inst}_{ts}.png"
            out_path = os.path.join(PLOT_DIR, safe_fn)

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
            logger.info(f"Plot saved to {out_path}")

            try:
                hist = []
                if os.path.exists(HISTORY_FILE):
                    with open(HISTORY_FILE, "r") as f:
                        hist = json.load(f)
                hist.append({"time": datetime.now(timezone.utc).isoformat(), "action":"metrics", "region":region, "instance":inst, "plot":out_path})
                with open(HISTORY_FILE, "w") as f:
                    json.dump(hist, f, default=str, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write history: {e}")

            values = [d.get("Average") for d in dps if d.get("Average") is not None]
            summary = {
                "count": len(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "avg": (sum(values)/len(values)) if values else None
            }
            logger.info(f"Metrics summary: {summary}")

            return {"Status":"OK", "Action":"metrics", "InstanceId": inst, "Region": region, "PlotPath": out_path, "Stats": summary}
    except Exception as e:
        logger.error(f"Exception in cloudwatch_metrics_tool: {e}\n{traceback.format_exc()}")
        return {"Status":"Error", "Message": f"{type(e).__name__}: {str(e)}", "Trace": traceback.format_exc()}


# --- Agent setup ---
SYSTEM_PROMPT = """
You are a helpful AWS cloud assistant. Use the provided tool 'cloudwatch_metrics_tool' for AWS operations.
Pass the user's raw message to 'cloudwatch_metrics_tool' when appropriate.
when asked for cpu or memory utilization or how much disk space is free use the 'cloudwatch_metrics_tool' for getting the metrics and plot the data.
otherwise use the 'use_aws' tool to perform the operation.
always show detailed information.
"""

bedrock_model = BedrockModel(model_id="us.amazon.nova-pro-v1:0", temperature=0.7, top_p=0.9)

agent = Agent(
    model=bedrock_model,
    system_prompt=SYSTEM_PROMPT,
    tools=[use_aws, cloudwatch_metrics_tool],
    messages=[]
)


# --- Streamlit UI ---
st.set_page_config(page_title="AWS Assistant", layout="wide")
st.title("AWS Assistant")

if st.button("Clear History & Start New Chat"):
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        logger.info(f"Deleted history file: {HISTORY_FILE}")
    st.session_state.chat_history = []
    st.rerun()

st.write("Type commands like `list instances in mumbai` or `show cpu for i-0123456789abcdef in the last 6 hours`")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_assistant_response(user_text: str) -> dict:
    logger.info(f"Getting assistant response for: {user_text}")
    user_text = user_text.strip()
    if not user_text:
        return {"role": "assistant", "content": "Empty query."}

    try:
        out = agent(user_text)
    except Exception as e:
        logger.error(f"Agent failed, falling back to cloudwatch_metrics_tool: {e}")
        out = cloudwatch_metrics_tool(user_text, prefer_plot=True)

    assistant_message = {"role": "assistant", "content": ""}

    if isinstance(out, dict):
        if out.get("Action") == "metrics" and out.get("PlotPath"):
            assistant_message["content"] = f"Plot generated for {out.get('InstanceId')} — stats: {out.get('Stats')}"
            assistant_message["plot"] = out.get("PlotPath")
        elif out.get("Action") == "list":
            insts = out.get("Instances", [])
            rows = [f"• `{i.get('InstanceId')}`: {i.get('Name') or ''} ({i.get('State')})" for i in insts]
            assistant_message["content"] = f"Listed {out.get('Count', 0)} instances:\n" + "\n".join(rows[:50])
        elif out.get("Status") == "Error":
            assistant_message["content"] = f"Error: {out.get('Message', 'An unknown error occurred.')}"
        else:
            assistant_message["content"] = "Operation completed."
    else:
        assistant_message["content"] = str(out)

    logger.info(f"Assistant response: {assistant_message['content']}")
    return assistant_message


# --- Chat UI ---

# Render chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "plot" in message and message["plot"] and os.path.exists(message["plot"]):
            plot_path = message["plot"]
            st.image(plot_path, use_container_width=True)
            with open(plot_path, "rb") as f:
                st.download_button(
                    label="Download Plot",
                    data=f,
                    file_name=os.path.basename(plot_path),
                    mime="image/png"
                )

# Chat input and response
if user_input := st.chat_input("Type your command..."):
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            assistant_response = get_assistant_response(user_input)
            st.markdown(assistant_response["content"])
            if "plot" in assistant_response and assistant_response["plot"] and os.path.exists(assistant_response["plot"]):
                plot_path = assistant_response["plot"]
                st.image(plot_path, use_container_width=True)
                with open(plot_path, "rb") as f:
                    st.download_button(
                        label="Download Plot",
                        data=f,
                        file_name=os.path.basename(plot_path),
                        mime="image/png"
                    )
    
    st.session_state.chat_history.append(assistant_response)


st.markdown("---")
st.write("**Notes:** Ensure AWS creds are configured and the strands/Bedrock pieces are available in your environment.")


if __name__ == '__main__':
    logger.info("Streamlit AWS Assistant started")
    pass
