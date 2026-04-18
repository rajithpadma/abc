"""
AI Agent - OPTIMIZED: 700 tokens/msg (was 3200)
"""
import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import time

from config.config import (
    OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL,
    MAX_TOKENS, TEMPERATURE, MODEL_PRIORITY, DEFAULT_MODEL
)
from src.agent.tools import tool_executor
from src.database.database import db_manager

OPTIMIZED_PROMPT = """You are SupportBot — an expert, polite, policy-driven customer support assistant.

CORE BEHAVIOR:
- Be concise (2–4 lines), precise, and actionable.
- Always rely ONLY on the provided context blocks.
- NEVER hallucinate or assume missing data.
- If information is missing, clearly say: "This detail is not available in the system."
- Always suggest exactly ONE clear next step.
- Before finalizing a return/replacement for damage claims, ask required validation questions if missing:
  damage confirmation, packaging condition, and pickup/address confirmation.

TONE:
- Professional, calm, and respectful.
- If the user is frustrated, respond with empathy:
  "I understand this is frustrating — I’ll help you right away."
- Do NOT expose system internals, APIs, database names, or backend logic.
- Avoid complex/legal-heavy language unless explicitly required.

DECISION PRIORITY (highest → lowest):
1) ORDER data
2) DECISION LOGIC
3) POLICY
4) HIERARCHY
5) LEGAL
6) RISK

RESPONSE FORMAT (STRICT):
- Start with a short acknowledgement sentence.
- Provide 2–4 concise lines answering the query.
- End with ONE clear next step.

CONTEXT USAGE:
Use ONLY the structured blocks below.

ORDER:
{order}

DECISION LOGIC:
{decision}

POLICY:
{policy}

HIERARCHY:
{hierarchy}

LEGAL:
{legal}

RISK:
{risk}

RECENT CHAT:
{history}
"""

class SimpleCache:
    def __init__(self, ttl=300):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        self.cache[key] = (value, time.time())
    
    def clear(self):
        self.cache.clear()

class CustomerSupportAgent:
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.api_url = f"{OPENROUTER_API_URL}/chat/completions"
        self.model = DEFAULT_MODEL or OPENROUTER_MODEL
        self.model_priority = MODEL_PRIORITY or [self.model]
        if self.model not in self.model_priority:
            self.model_priority.insert(0, self.model)
        self.api_available = bool(self.api_key and "sk-or-" in self.api_key)
        
        if self.api_available:
            print(f"✓ API configured: {self.model}")
        else:
            print("⚠ API not configured")
        
        self.max_tokens = min(MAX_TOKENS, 800)
        self.temperature = TEMPERATURE
        self.conversation_history: Dict[str, List[Dict]] = {}
        self.session_context: Dict[str, Dict] = {}
        self.cache = SimpleCache(ttl=300)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.message_count = 0
    
    def initialize_session(self, session_id: str, user_id: str = None) -> Dict:
        self.conversation_history[session_id] = []
        self.session_context[session_id] = {
            "user_id": user_id,
            "session_id": session_id,
            "started_at": datetime.utcnow().isoformat()
        }
        return {"status": "initialized", "session_id": session_id}
    
    def set_selected_order(self, session_id: str, order_id: str):
        if session_id in self.session_context:
            self.session_context[session_id]["selected_order_id"] = order_id
    
    def chat(self, session_id: str, user_message: str, user_id: str = None, 
             image_analysis: Dict = None, selected_order_id: str = None) -> Dict:
        
        if session_id not in self.conversation_history:
            self.initialize_session(session_id, user_id)
        
        if user_id:
            self.session_context[session_id]["user_id"] = user_id
        if selected_order_id:
            self.session_context[session_id]["selected_order_id"] = selected_order_id
        
        context_message = user_message
        if image_analysis:
            context_message += f"\n[Image: {image_analysis.get('category', 'unknown')} {int(image_analysis.get('confidence', 0)*100)}%]"
        
        self.conversation_history[session_id].append({"role": "user", "content": context_message})
        messages = self._prepare_optimized_messages(session_id)
        
        if self.api_available:
            response = self._call_api(messages, session_id)
        else:
            response = self._fallback_response(user_message, session_id)
        
        self.conversation_history[session_id].append({"role": "assistant", "content": response["message"]})
        self._save_chat(session_id, user_message, response["message"])
        return response
    
    def _prepare_optimized_messages(self, session_id: str) -> List[Dict]:
        context = self.session_context.get(session_id, {})
        user_id = context.get("user_id")
        order_id = context.get("selected_order_id")
        order_block = "This detail is not available in the system."
        policy_block = "This detail is not available in the system."
        decision_block = "This detail is not available in the system."
        hierarchy_block = "This detail is not available in the system."
        legal_block = "This detail is not available in the system."
        risk_block = "This detail is not available in the system."
        
        if order_id:
            cache_key = f"order:{order_id}"
            order = self.cache.get(cache_key)
            if not order:
                order = db_manager.get_order_by_id(order_id)
                if order:
                    self.cache.set(cache_key, order)
            
            if order:
                order_block = (
                    f"Order ID: {order.get('order_id', 'N/A')} | "
                    f"Product: {order.get('product_name', 'N/A')} | "
                    f"Status: {order.get('status', 'N/A')} | "
                    f"Price: {order.get('price', 'N/A')} | "
                    f"Address: {order.get('address', 'N/A')}"
                )
                
                category = order.get('category')
                if category:
                    cache_key = f"policy:{category}"
                    policy = self.cache.get(cache_key)
                    if not policy:
                        policy = db_manager.get_policy_by_category(category)
                        if policy:
                            self.cache.set(cache_key, policy)
                    
                    if policy:
                        policy_block = (
                            f"Category: {category} | "
                            f"Return window: {policy.get('return_window_days', 30)} days | "
                            f"Replacement window: {policy.get('replacement_window_days', 7)} days | "
                            f"Risk: {policy.get('risk_level', 'N/A')}"
                        )
        elif user_id:
            order_block = f"User ID: {user_id} (order context available if selected)"

        decision_rules = db_manager.get_decision_rules()
        if decision_rules:
            decision_block = self._compact_records(decision_rules, max_items=5)

        hierarchy_rules = db_manager.get_policy_hierarchy_rules(limit=20)
        if hierarchy_rules:
            hierarchy_block = self._compact_records(hierarchy_rules, max_items=5)

        legal_rules = db_manager.get_legal_compliance_rules(limit=20)
        if legal_rules:
            legal_block = self._compact_records(legal_rules, max_items=5)

        risk_rules = db_manager.get_risk_assessment_rules(limit=20)
        if risk_rules:
            risk_block = self._compact_records(risk_rules, max_items=5)

        recent_chat = self._recent_chat_block(session_id)
        system_content = OPTIMIZED_PROMPT.format(
            order=order_block,
            decision=decision_block,
            policy=policy_block,
            hierarchy=hierarchy_block,
            legal=legal_block,
            risk=risk_block,
            history=recent_chat
        )
        
        messages = [{"role": "system", "content": system_content}]
        history = self.conversation_history.get(session_id, [])
        
        if len(history) <= 4:
            messages.extend(history)
        else:
            messages.append(history[0])
            middle = history[1:-3]
            if middle:
                summary = self._compress_history(middle)
                messages.append({"role": "system", "content": f"[Previous: {summary}]"})
            messages.extend(history[-3:])
        
        return messages

    def _compact_records(self, records: List[Dict], max_items: int = 5) -> str:
        compacted = []
        for record in records[:max_items]:
            parts = []
            for k, v in record.items():
                if k in {"_id"}:
                    continue
                if isinstance(v, (dict, list)):
                    continue
                text = str(v).strip()
                if text:
                    parts.append(f"{k}: {text}")
            if parts:
                compacted.append(" | ".join(parts[:5]))
        return "\n".join(compacted) if compacted else "This detail is not available in the system."

    def _recent_chat_block(self, session_id: str, limit: int = 6) -> str:
        history = self.conversation_history.get(session_id, [])
        if not history:
            return "No recent chat."
        lines = []
        for msg in history[-limit:]:
            role = msg.get("role", "user")
            content = (msg.get("content", "") or "").replace("\n", " ").strip()
            if content:
                lines.append(f"{role}: {content[:250]}")
        return "\n".join(lines) if lines else "No recent chat."
    
    def _compress_history(self, messages: List[Dict]) -> str:
        all_text = " ".join([m.get("content", "")[:100] for m in messages]).lower()
        topics = []
        if "return" in all_text or "refund" in all_text:
            topics.append("return")
        if "replace" in all_text or "damage" in all_text:
            topics.append("replace")
        if "track" in all_text:
            topics.append("track")
        if "image" in all_text:
            topics.append("image")
        return ", ".join(topics) if topics else "inquiry"
    
    def _call_api(self, messages: List[Dict], session_id: str) -> Dict:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://customer-support-app.local",
                "X-Title": "Customer Support"
            }
            
            estimated_input = sum(len(m.get("content", "")) // 4 for m in messages)
            self.total_input_tokens += estimated_input
            self.message_count += 1

            for model_name in self.model_priority:
                try:
                    payload = {
                        "model": model_name,
                        "messages": messages,
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature
                    }

                    response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)

                    if response.status_code == 200:
                        result = response.json()
                        assistant_message = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                        if assistant_message:
                            self.model = model_name
                            self.total_output_tokens += len(assistant_message) // 4
                            actions = self._process_actions(assistant_message, session_id)
                            return {"message": assistant_message, "actions": actions}

                    print(f"API error ({model_name}): {response.status_code}")
                except Exception as model_error:
                    print(f"API error ({model_name}): {model_error}")

            return self._fallback_response(self.conversation_history[session_id][-1]["content"], session_id)
            
        except Exception as e:
            print(f"API error: {e}")
            return self._fallback_response(self.conversation_history[session_id][-1]["content"], session_id)
    
    def _process_actions(self, response: str, session_id: str) -> List[Dict]:
        actions = []
        response_lower = response.lower()
        if "pickup" in response_lower and "scheduled" in response_lower:
            actions.append({"tool": "pickup_mentioned", "status": "detected"})
        if "delivery" in response_lower and "scheduled" in response_lower:
            actions.append({"tool": "delivery_mentioned", "status": "detected"})
        return actions
    
    def _fallback_response(self, user_message: str, session_id: str) -> Dict:
        msg_lower = user_message.lower()
        context = self.session_context.get(session_id, {})
        user_id = context.get("user_id")
        
        if any(w in msg_lower for w in ["hi", "hello", "hey"]):
            if user_id:
                user_data = db_manager.get_user_by_id(user_id)
                name = user_data.get("name", "Customer") if user_data else "Customer"
                return {"message": f"Hello {name}! I can help with orders, returns, replacements, and tracking.", "actions": []}
            return {"message": "Hello! How can I help?", "actions": []}
        
        if any(w in msg_lower for w in ["return", "refund"]):
            return {"message": "To return:\n1. Order ID\n2. Reason\n3. Upload image\n\nReturns within 30 days.", "actions": []}
        
        if any(w in msg_lower for w in ["replace", "damaged"]):
            return {"message": "For replacement:\n1. Order ID\n2. Describe issue\n3. Upload image\n\nReplacements within 7 days.", "actions": []}
        
        if any(w in msg_lower for w in ["track", "shipping"]):
            if user_id:
                shipments = db_manager.get_shipments_by_user(user_id)
                if shipments:
                    ship_list = ", ".join([s.get('shipment_id', '') for s in shipments[:3]])
                    return {"message": f"Your shipments: {ship_list}", "actions": []}
            return {"message": "Provide shipment ID or order ID to track.", "actions": []}
        
        return {"message": "I can help with orders, returns, replacements, and tracking.", "actions": []}
    
    def _save_chat(self, session_id: str, user_message: str, assistant_message: str):
        context = self.session_context.get(session_id, {})
        db_manager.save_chat_message({"session_id": session_id, "user_id": context.get("user_id"), "role": "user", "content": user_message})
        db_manager.save_chat_message({"session_id": session_id, "user_id": context.get("user_id"), "role": "assistant", "content": assistant_message})
    
    def generate_summary(self, session_id: str) -> Dict:
        history = self.conversation_history.get(session_id, [])
        context = self.session_context.get(session_id, {})
        
        if not history:
            return {"error": "No conversation"}
        
        if self.api_available:
            try:
                summary_prompt = "Summarize in JSON: issue_type, proposed_solution, resolution_status.\n\n"
                for msg in history[-5:]:
                    summary_prompt += f"{msg['role']}: {msg['content'][:200]}\n"
                
                response = requests.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": self.model, "messages": [{"role": "user", "content": summary_prompt}], "max_tokens": 200},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    summary_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    try:
                        summary = json.loads(summary_text)
                    except:
                        import re
                        match = re.search(r'\{.*\}', summary_text, re.DOTALL)
                        summary = json.loads(match.group()) if match else {}
                    
                    summary.update({"session_id": session_id, "user_id": context.get("user_id"), "timestamp": datetime.utcnow().isoformat(), "message_count": len(history)})
                    db_manager.save_chat_summary(summary)
                    return summary
            except Exception as e:
                print(f"AI summary error: {e}")
        
        return self._fallback_summary(session_id, history, context)
    
    def _fallback_summary(self, session_id: str, history: List, context: Dict) -> Dict:
        all_text = " ".join([m.get("content", "") for m in history]).lower()
        issue_type = "inquiry"
        if "return" in all_text or "refund" in all_text:
            issue_type = "return"
        elif "replace" in all_text or "damage" in all_text:
            issue_type = "replacement"
        elif "track" in all_text:
            issue_type = "tracking"
        
        summary = {
            "session_id": session_id,
            "user_id": context.get("user_id"),
            "issue_type": issue_type,
            "issue_description": f"Customer {issue_type}",
            "proposed_solution": "pending",
            "resolution_status": "pending",
            "customer_sentiment": "neutral",
            "action_items": ["Review conversation"],
            "timestamp": datetime.utcnow().isoformat(),
            "message_count": len(history)
        }
        db_manager.save_chat_summary(summary)
        return summary
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        return self.conversation_history.get(session_id, [])
    
    def get_token_stats(self) -> Dict:
        avg_input = self.total_input_tokens // max(self.message_count, 1)
        avg_output = self.total_output_tokens // max(self.message_count, 1)
        return {
            "total_messages": self.message_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "avg_input_per_message": avg_input,
            "avg_output_per_message": avg_output,
            "avg_total_per_message": avg_input + avg_output
        }
    
    def print_token_stats(self):
        stats = self.get_token_stats()
        print("\n" + "="*60)
        print("  TOKEN USAGE STATISTICS")
        print("="*60)
        print(f"Total Messages: {stats['total_messages']}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print(f"  - Input: {stats['total_input_tokens']:,}")
        print(f"  - Output: {stats['total_output_tokens']:,}")
        print(f"Average per message: {stats['avg_total_per_message']}")
        print("="*60 + "\n")
    
    def clear_session(self, session_id: str):
        self.conversation_history.pop(session_id, None)
        self.session_context.pop(session_id, None)

support_agent = CustomerSupportAgent()