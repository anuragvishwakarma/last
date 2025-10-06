# agents.py
import boto3
import pandas as pd
from datetime import datetime, timedelta
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse

# === Embeddings & FAISS (unchanged) ===
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
embeddings = BedrockEmbeddings(client=bedrock_client, model_id="amazon.titan-embed-text-v2:0")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# === Nova Premier LLM ===
llm = ChatBedrockConverse(
    client=bedrock_client,
    model="amazon.nova-premier-v1:0",  # ✅ Use Nova Premier
    temperature=0.3,
    max_tokens=1000,
)

# === Maintenance Rules (same as before) ===
MAINTENANCE_RULES = {
    "A": {"interval_months": 6, "type": "T", "effort_hrs": 8},
    "B": {"interval_months": 0.25, "type": "T", "effort_hrs": 2},
    "C": {"interval_months": 1, "type": "T", "effort_hrs": 4},
    "D": {"interval_months": 3, "type": "T", "effort_hrs": 2},
    "E": {"interval_months": 3, "type": "T", "effort_hrs": 4},
    "F": {"interval_months": 3, "type": "T", "effort_hrs": 2},
    "G": {"interval_months": 3, "type": "T", "effort_hrs": 2},
    "H": {"interval_months": 6, "type": "T", "effort_hrs": 2},
    "I": {"interval_months": 6, "type": "T", "effort_hrs": 2},
    "J": {"interval_months": 3, "type": "T", "effort_hrs": 1},
    "K": {"interval_months": 6, "type": "T", "effort_hrs": 4},
    "L": {"interval_months": 12, "type": "O", "effort_hrs": 8},
    "M": {"interval_months": 6, "type": "T", "effort_hrs": 4},
    "N": {"interval_months": 3, "type": "T", "effort_hrs": 4},
    "O": {"interval_months": 1, "type": "T", "effort_hrs": 2},
    "P": {"interval_months": 3, "type": "O", "effort_hrs": 4},
}

def workflow_manager_agent(equipment_id: str, pos: str):
    df = pd.read_csv("data/synthetic_maintenance_records.csv", sep=";")
    df_eq_pos = df[(df["Equipment ID"] == equipment_id) & (df["Pos"] == pos)].copy()
    
    if df_eq_pos.empty:
        last_date = None
        assigned_worker = "Unassigned"
    else:
        df_eq_pos["Date of Inspection"] = pd.to_datetime(df_eq_pos["Date of Inspection"], format="%m/%d/%Y")
        last_row = df_eq_pos.loc[df_eq_pos["Date of Inspection"].idxmax()]
        last_date = last_row["Date of Inspection"]
        worker = df_eq_pos["Fieldworker Name"].mode()
        assigned_worker = worker.iloc[0] if not worker.empty else "Unassigned"

    rule = MAINTENANCE_RULES.get(pos, {"interval_months": 12, "type": "T", "effort_hrs": 4})
    now = datetime.today()
    
    if last_date is None:
        due_date = now
    else:
        due_date = last_date + timedelta(days=int(rule["interval_months"] * 30))
    
    overdue = now > due_date
    
    return {
        "equipment_id": equipment_id,
        "pos": pos,
        "component": df_eq_pos["Component/Function"].iloc[0] if not df_eq_pos.empty else "Unknown",
        "last_inspection": last_date.strftime("%Y-%m-%d") if last_date else "Never",
        "next_due": due_date.strftime("%Y-%m-%d"),
        "overdue": overdue,
        "assigned_to": assigned_worker,
        "effort_hours": rule["effort_hrs"],
        "inspection_type": rule["type"]
    }

def coordinator(query: str):
    # Retrieve context
    docs = vectorstore.similarity_search(query, k=8)
    context = "\n".join([d.page_content for d in docs])
    
    # Extract equipment & pos
    equipment_id = None
    pos = None
    for token in query.split():
        if token.startswith("HCK_EQ"):
            equipment_id = token
        if len(token) == 1 and token in "ABCDEFGHIJKLMNOP":
            pos = token

    # Get workflow recommendation
    workflow_info = {}
    if equipment_id and pos and pos in MAINTENANCE_RULES:
        try:
            workflow_info = workflow_manager_agent(equipment_id, pos)
        except Exception as e:
            workflow_info = {"error": str(e)}

    # Tech Agent
    tech_response = llm.invoke(
        "You are a Technical Spec Expert. Answer using ONLY the following technical manual excerpts:\n" + context + "\n\nQuestion: " + query
    )

    # Maint Agent
    maint_response = llm.invoke(
        "You are a Maintenance Log Analyst. Summarize maintenance history from logs:\n" + context + "\n\nQuestion: " + query
    )

    # Format Workflow Text
    workflow_text = ""
    if workflow_info and "error" not in workflow_info:
        status = "OVERDUE ⚠️" if workflow_info["overdue"] else "Scheduled"
        workflow_text = (
            f"\n\n🔧 **Workflow Recommendation**:\n"
            f"- Component: {workflow_info['component']} (Pos {workflow_info['pos']})\n"
            f"- Next due: {workflow_info['next_due']} ({status})\n"
            f"- Assign to: **{workflow_info['assigned_to']}**\n"
            f"- Effort: {workflow_info['effort_hours']} hrs\n"
        )
    elif "error" in workflow_info:
        workflow_text = f"\n\n⚠️ **Workflow Error**: {workflow_info['error']}"

    # Final Synthesis
    final_prompt = (
        "Technical Guidance:\n" + tech_response.content + "\n\n"
        "Maintenance History:\n" + maint_response.content + "\n\n"
        + workflow_text + "\n\n"
        "User Query: " + query
    )
    final_response = llm.invoke(final_prompt)
    return final_response.content