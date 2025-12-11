import requests
import json
import sys

def verify_chat_stream():
    url = "http://localhost:8000/chat"
    payload = {
        "message": "Verify streaming functionality.",
        "session_id": "auto-verify-session",
        "granularity_level": 0.5
    }
    
    print(f"Connecting to {url}...")
    try:
        with requests.post(url, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"Error: Status code {response.status_code}")
                print(response.text)
                return

            print("--- Stream Start ---")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        data = json.loads(decoded_line)
                        # We expect ContextBlock structure
                        content = data.get("content", "")
                        b_type = data.get("type", "unknown")
                        node_id = data.get("node_id")
                        state = data.get("state")
                        
                        if b_type == "graph_update":
                            graph_data = data.get("metadata", {}).get("graph", {})
                            nodes_len = len(graph_data.get("nodes", []))
                            print(f"\n[GRAPH UPDATE] Nodes: {nodes_len}")
                        elif b_type == "text":
                            # sys.stdout.write(content)
                            if state == "new":
                                print(f"\n[NODE START] {node_id}")
                            elif state == "finished":
                                print(f"\n[NODE FINISHED] {node_id}")
                            else:
                                sys.stdout.write(content)
                                sys.stdout.flush()
                        else:
                             print(f"\n[BLOCK] Type: {b_type} | Content: {content}")
                             
                    except json.JSONDecodeError:
                        print(f"\n[Raw]: {decoded_line}")
            print("\n--- Stream End ---")
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    verify_chat_stream()
