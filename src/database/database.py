"""
Database Module for Agentic AI Customer Support
Uses MongoDB when available and falls back to bundled CSV data otherwise.
"""
import csv
import json
import os
import sys
import threading
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import COLLECTIONS, DATA_DIR, DATABASE_NAME, MONGODB_URI, RUNTIME_DATA_PATH


class DatabaseManager:
    """Manages data access across MongoDB and file-backed fallback storage."""

    _instance = None

    CSV_FILES = {
        "orders": "Order_Database_581_Records.csv",
        "products": "Product_Database.csv",
        "policies": "Policy_Summary_by_Category.csv",
        "decision_tree": "Decision_Tree_Logic.csv",
        "policy_hierarchy": "Policy_Hierarchy.csv",
        "legal_compliance": "Legal_Compliance.csv",
        "risk_assessment": "Risk_Assessment_Matrix.csv",
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.client = None
        self.db = None
        self.mode = "uninitialized"
        self._lock = threading.Lock()
        self.base_records: Dict[str, List[Dict[str, Any]]] = {}
        self.runtime_records: Dict[str, List[Dict[str, Any]]] = {
            "shipments": [],
            "chat_history": [],
            "returns": [],
            "replacements": [],
        }
        self._initialized = True

    def connect(self) -> bool:
        """Establish connection to MongoDB or load bundled CSV data."""
        if MONGODB_URI:
            try:
                self.client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
                self.client.admin.command("ping")
                self.db = self.client[DATABASE_NAME]
                self.mode = "mongodb"
                print(f"Connected to MongoDB: {DATABASE_NAME}")
                return True
            except ConnectionFailure as e:
                print(f"✗ Failed to connect to MongoDB: {e}")
            except PyMongoError as e:
                print(f"✗ MongoDB error: {e}")

        self.db = None
        self.client = None
        self._load_csv_fallback()
        return False

    def disconnect(self):
        if self.client:
            self.client.close()
            print("Disconnected from MongoDB")

    def _check_connection(self) -> bool:
        return self.db is not None

    def using_fallback(self) -> bool:
        return self.mode == "csv_fallback"

    def _load_csv_fallback(self):
        self.base_records = {}
        for key, filename in self.CSV_FILES.items():
            self.base_records[key] = self._read_csv_records(os.path.join(DATA_DIR, filename))
        self._hydrate_runtime_state()
        self.mode = "csv_fallback"
        print(f"Using bundled CSV fallback data from: {DATA_DIR}")

    def _read_csv_records(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            print(f"⚠ Missing CSV dataset: {path}")
            return []
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            return [self._normalize_loaded_record(row) for row in csv.DictReader(handle)]

    def _hydrate_runtime_state(self):
        if not os.path.exists(RUNTIME_DATA_PATH):
            return
        try:
            with open(RUNTIME_DATA_PATH, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            for key in self.runtime_records:
                values = raw.get(key, [])
                self.runtime_records[key] = [self._normalize_loaded_record(item) for item in values]
        except Exception as e:
            print(f"⚠ Unable to read runtime state: {e}")

    def _persist_runtime_state(self):
        os.makedirs(os.path.dirname(RUNTIME_DATA_PATH), exist_ok=True)
        payload = {
            key: [self._serialize_for_json(item) for item in values]
            for key, values in self.runtime_records.items()
        }
        with open(RUNTIME_DATA_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)

    def _normalize_loaded_record(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._normalize_loaded_record(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._normalize_loaded_record(item) for item in value]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return ""
            for parser in (datetime.fromisoformat,):
                try:
                    return parser(text)
                except ValueError:
                    pass
            lowered = text.lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
            return text
        return value

    def _serialize_for_json(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._serialize_for_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._serialize_for_json(item) for item in value]
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    def _clone(self, value: Any) -> Any:
        return deepcopy(value)

    def _records(self, key: str) -> List[Dict[str, Any]]:
        if self._check_connection():
            raise RuntimeError("MongoDB mode should query the database directly.")
        if key in self.base_records:
            return self.base_records[key]
        return self.runtime_records.get(key, [])

    def _next_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    def _normalize_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._clone(order)
        normalized["_id"] = str(normalized.get("_id", normalized.get("Order_ID", self._next_id("order"))))
        normalized["order_id"] = normalized.get("Order_ID", "")
        normalized["user_id"] = normalized.get("User_ID", "")
        normalized["customer_name"] = normalized.get("Customer_Name", "")
        normalized["email"] = normalized.get("Email", "")
        normalized["product_id"] = normalized.get("Product_ID", "")
        normalized["product_name"] = normalized.get("Product_Name", "")
        normalized["order_date"] = normalized.get("Date_Ordered", "")
        normalized["date"] = normalized.get("Date_Ordered", "")
        normalized["price"] = normalized.get("Price_Inr", "")
        normalized["payment_method"] = normalized.get("Payment_Mode", "")
        normalized["address"] = normalized.get("Billing_Address", "")
        normalized["status"] = normalized.get("Status", "Delivered")
        product = self.get_product_by_id(normalized["product_id"])
        normalized["category"] = normalized.get("Category") or (product.get("category") if product else "")
        return normalized

    def _normalize_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._clone(product)
        normalized["_id"] = str(normalized.get("_id", normalized.get("Product_ID", self._next_id("product"))))
        normalized["product_id"] = normalized.get("Product_ID", "")
        normalized["name"] = normalized.get("Product_Name", "")
        normalized["category"] = normalized.get("Category", "")
        normalized["price"] = normalized.get("Price_INR", "")
        normalized["description"] = normalized.get("Detailed_Description", "")
        normalized["risk_level"] = normalized.get("Risk_Level", "")
        normalized["return_eligible"] = normalized.get("Return_Eligible", False)
        normalized["return_days"] = normalized.get("Return_Days", 0)
        normalized["replacement_eligible"] = normalized.get("Replacement_Eligible", False)
        normalized["replacement_days"] = normalized.get("Replacement_Days", 0)
        normalized["warranty_months"] = normalized.get("Warranty_Months", 0)
        normalized["return_conditions"] = normalized.get("Return_Conditions", "")
        normalized["refund_policy"] = normalized.get("Refund_Policy", "")
        normalized["refund_method"] = normalized.get("Refund_Method", "")
        return normalized

    def _normalize_policy(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._clone(policy)
        normalized["_id"] = str(normalized.get("_id", normalized.get("Category", self._next_id("policy"))))
        normalized["category"] = normalized.get("Category", "")
        normalized["risk_level"] = normalized.get("Risk_Level", "")
        normalized["return_window_days"] = normalized.get("Return_Days_max", normalized.get("Return_Days", 7))
        normalized["replacement_window_days"] = normalized.get(
            "Replacement_Days_max",
            normalized.get("Replacement_Days", 14),
        )
        normalized["warranty_months"] = normalized.get("Warranty_Months_max", normalized.get("Warranty_Months", 0))
        return normalized

    def _normalize_runtime_doc(self, payload: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        record = self._clone(payload)
        record["_id"] = str(record.get("_id", self._next_id(prefix)))
        return record

    # =========================================================================
    # USER OPERATIONS
    # =========================================================================
    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        if self._check_connection():
            try:
                order = self.db[COLLECTIONS["orders"]].find_one({"User_ID": user_id})
                if not order:
                    return None
                return {
                    "_id": str(order.get("_id", "")),
                    "user_id": order.get("User_ID", user_id),
                    "name": order.get("Customer_Name", "Customer"),
                    "email": order.get("Email", ""),
                    "phone": order.get("Phone", ""),
                    "address": order.get("Billing_Address", ""),
                }
            except PyMongoError as e:
                print(f"Error fetching user: {e}")
                return None

        for order in self._records("orders"):
            if order.get("User_ID") == user_id:
                return {
                    "_id": str(order.get("Order_ID", self._next_id("user"))),
                    "user_id": order.get("User_ID", user_id),
                    "name": order.get("Customer_Name", "Customer"),
                    "email": order.get("Email", ""),
                    "phone": order.get("Phone", ""),
                    "address": order.get("Billing_Address", ""),
                }
        return None

    def authenticate_user(self, user_id: str) -> Optional[Dict]:
        return self.get_user_by_id(user_id)

    # =========================================================================
    # ORDER OPERATIONS
    # =========================================================================
    def get_orders_by_user(self, user_id: str) -> List[Dict]:
        if self._check_connection():
            try:
                orders = list(self.db[COLLECTIONS["orders"]].find({"User_ID": user_id}))
                return [self._normalize_order(order) for order in orders]
            except PyMongoError as e:
                print(f"Error fetching orders: {e}")
                return []

        return [self._normalize_order(order) for order in self._records("orders") if order.get("User_ID") == user_id]

    def get_order_by_id(self, order_id: str) -> Optional[Dict]:
        if self._check_connection():
            try:
                order = self.db[COLLECTIONS["orders"]].find_one({"Order_ID": order_id})
                return self._normalize_order(order) if order else None
            except PyMongoError as e:
                print(f"Error fetching order: {e}")
                return None

        for order in self._records("orders"):
            if order.get("Order_ID") == order_id:
                return self._normalize_order(order)
        return None

    def get_recent_orders(self, user_id: str, limit: int = 5) -> List[Dict]:
        orders = self.get_orders_by_user(user_id)
        return orders[:limit]

    # =========================================================================
    # PRODUCT OPERATIONS
    # =========================================================================
    def get_product_by_id(self, product_id: str) -> Optional[Dict]:
        if self._check_connection():
            try:
                product = self.db[COLLECTIONS["products"]].find_one({"Product_ID": product_id})
                return self._normalize_product(product) if product else None
            except PyMongoError as e:
                print(f"Error fetching product: {e}")
                return None

        for product in self._records("products"):
            if product.get("Product_ID") == product_id:
                return self._normalize_product(product)
        return None

    def get_products_by_ids(self, product_ids: List[str]) -> List[Dict]:
        wanted = set(product_ids)
        if not wanted:
            return []

        if self._check_connection():
            try:
                products = list(self.db[COLLECTIONS["products"]].find({"Product_ID": {"$in": list(wanted)}}))
                return [self._normalize_product(product) for product in products]
            except PyMongoError as e:
                print(f"Error fetching products: {e}")
                return []

        return [self._normalize_product(product) for product in self._records("products") if product.get("Product_ID") in wanted]

    # =========================================================================
    # POLICY OPERATIONS
    # =========================================================================
    def get_policy_by_category(self, category: str) -> Optional[Dict]:
        if self._check_connection():
            try:
                policy = self.db[COLLECTIONS["policies"]].find_one({"Category": category})
                return self._normalize_policy(policy) if policy else None
            except PyMongoError as e:
                print(f"Error fetching policy: {e}")
                return None

        for policy in self._records("policies"):
            if policy.get("Category") == category:
                return self._normalize_policy(policy)
        return None

    def get_all_policies(self) -> List[Dict]:
        if self._check_connection():
            try:
                policies = list(self.db[COLLECTIONS["policies"]].find())
                return [self._normalize_policy(policy) for policy in policies]
            except PyMongoError as e:
                print(f"Error fetching policies: {e}")
                return []

        return [self._normalize_policy(policy) for policy in self._records("policies")]

    def get_return_policy(self, product_id: str = None, category: str = None) -> Optional[Dict]:
        if product_id and not category:
            product = self.get_product_by_id(product_id)
            if product:
                category = product.get("category")
        if category:
            policy = self.get_policy_by_category(category)
            if policy:
                policy["return_window_days"] = policy.get("return_window_days", policy.get("return_days", 30))
                return policy
        return {
            "return_window_days": 30,
            "conditions": ["Product must be unused", "Original packaging required"],
            "refund_method": "Original payment method",
            "is_default": True,
        }

    def get_replacement_policy(self, product_id: str = None) -> Optional[Dict]:
        category = None
        if product_id:
            product = self.get_product_by_id(product_id)
            if product:
                category = product.get("category")
        if category:
            policy = self.get_policy_by_category(category)
            if policy:
                policy["replacement_window_days"] = policy.get(
                    "replacement_window_days",
                    policy.get("replacement_days", 7),
                )
                return policy
        return {
            "replacement_window_days": 7,
            "conditions": ["Defective product", "Wrong product delivered"],
            "is_default": True,
        }

    def get_decision_rules(self, issue_type: str = None) -> List[Dict]:
        if self._check_connection():
            try:
                query = {"issue_type": issue_type} if issue_type else {}
                rules = list(self.db[COLLECTIONS["decision_tree"]].find(query))
                return [{**rule, "_id": str(rule.get("_id", ""))} for rule in rules] or [{"rule": "Follow standard support guidelines"}]
            except PyMongoError as e:
                print(f"Error fetching decision rules: {e}")
                return []

        rules = self._clone(self._records("decision_tree"))
        return rules or [{"rule": "Follow standard support guidelines"}]

    def get_policy_hierarchy_rules(self, limit: int = 50) -> List[Dict]:
        if self._check_connection():
            try:
                docs = list(self.db[COLLECTIONS["policy_hierarchy"]].find().limit(limit))
                return [{**doc, "_id": str(doc.get("_id", ""))} for doc in docs]
            except PyMongoError as e:
                print(f"Error fetching policy hierarchy: {e}")
                return []
        return self._clone(self._records("policy_hierarchy")[:limit])

    def get_legal_compliance_rules(self, limit: int = 50) -> List[Dict]:
        if self._check_connection():
            try:
                docs = list(self.db[COLLECTIONS["legal_compliance"]].find().limit(limit))
                return [{**doc, "_id": str(doc.get("_id", ""))} for doc in docs]
            except PyMongoError as e:
                print(f"Error fetching legal compliance: {e}")
                return []
        return self._clone(self._records("legal_compliance")[:limit])

    def get_risk_assessment_rules(self, limit: int = 50) -> List[Dict]:
        if self._check_connection():
            try:
                docs = list(self.db[COLLECTIONS["risk_assessment"]].find().limit(limit))
                return [{**doc, "_id": str(doc.get("_id", ""))} for doc in docs]
            except PyMongoError as e:
                print(f"Error fetching risk assessment: {e}")
                return []
        return self._clone(self._records("risk_assessment")[:limit])

    # =========================================================================
    # SHIPMENT OPERATIONS
    # =========================================================================
    def create_shipment(self, shipment_data: Dict) -> Optional[str]:
        payload = self._clone(shipment_data)
        payload["created_at"] = payload.get("created_at", datetime.utcnow())
        payload["updated_at"] = datetime.utcnow()

        if self._check_connection():
            try:
                result = self.db[COLLECTIONS["shipments"]].insert_one(payload)
                return str(result.inserted_id)
            except PyMongoError as e:
                print(f"Error creating shipment: {e}")
                return None

        with self._lock:
            record = self._normalize_runtime_doc(payload, "shipment")
            self.runtime_records["shipments"].append(record)
            self._persist_runtime_state()
            return record["_id"]

    def get_shipment_by_id(self, shipment_id: str) -> Optional[Dict]:
        if self._check_connection():
            try:
                shipment = self.db[COLLECTIONS["shipments"]].find_one({"shipment_id": shipment_id})
                if shipment:
                    shipment["_id"] = str(shipment["_id"])
                return shipment
            except PyMongoError as e:
                print(f"Error fetching shipment: {e}")
                return None

        for shipment in self._records("shipments"):
            if shipment.get("shipment_id") == shipment_id:
                return self._clone(shipment)
        return None

    def get_shipments_by_user(self, user_id: str) -> List[Dict]:
        if self._check_connection():
            try:
                shipments = list(self.db[COLLECTIONS["shipments"]].find({"user_id": user_id}))
                for shipment in shipments:
                    shipment["_id"] = str(shipment["_id"])
                return shipments
            except PyMongoError as e:
                print(f"Error fetching shipments: {e}")
                return []

        return [self._clone(item) for item in self._records("shipments") if item.get("user_id") == user_id]

    def update_shipment_status(self, shipment_id: str, status: str, stage_data: Dict = None) -> bool:
        if self._check_connection():
            try:
                update_data = {"status": status, "updated_at": datetime.utcnow()}
                if stage_data:
                    update_data["current_stage"] = stage_data
                result = self.db[COLLECTIONS["shipments"]].update_one(
                    {"shipment_id": shipment_id},
                    {"$set": update_data, "$push": {"timeline": stage_data} if stage_data else {}},
                )
                return result.modified_count > 0
            except PyMongoError as e:
                print(f"Error updating shipment: {e}")
                return False

        with self._lock:
            for shipment in self.runtime_records["shipments"]:
                if shipment.get("shipment_id") == shipment_id:
                    shipment["status"] = status
                    shipment["updated_at"] = datetime.utcnow()
                    if stage_data:
                        shipment["current_stage"] = stage_data
                        timeline = shipment.get("timeline", [])
                        for existing_stage in timeline:
                            if existing_stage.get("name") == stage_data.get("name"):
                                existing_stage.update(stage_data)
                                break
                        else:
                            timeline.append(stage_data)
                    self._persist_runtime_state()
                    return True
        return False

    def get_all_shipments(self) -> List[Dict]:
        if self._check_connection():
            try:
                shipments = list(self.db[COLLECTIONS["shipments"]].find().sort("created_at", -1))
                for shipment in shipments:
                    shipment["_id"] = str(shipment["_id"])
                return shipments
            except PyMongoError as e:
                print(f"Error fetching all shipments: {e}")
                return []

        return sorted(
            [self._clone(item) for item in self._records("shipments")],
            key=lambda item: item.get("created_at") or datetime.min,
            reverse=True,
        )

    # =========================================================================
    # RETURN/REPLACEMENT OPERATIONS
    # =========================================================================
    def create_return_request(self, return_data: Dict) -> Optional[str]:
        payload = self._clone(return_data)
        payload["created_at"] = datetime.utcnow()
        payload["status"] = "pending"
        if self._check_connection():
            try:
                result = self.db[COLLECTIONS["returns"]].insert_one(payload)
                return str(result.inserted_id)
            except PyMongoError as e:
                print(f"Error creating return request: {e}")
                return None

        with self._lock:
            record = self._normalize_runtime_doc(payload, "return")
            self.runtime_records["returns"].append(record)
            self._persist_runtime_state()
            return record["_id"]

    def create_replacement_request(self, replacement_data: Dict) -> Optional[str]:
        payload = self._clone(replacement_data)
        payload["created_at"] = datetime.utcnow()
        payload["status"] = "pending"
        if self._check_connection():
            try:
                result = self.db[COLLECTIONS["replacements"]].insert_one(payload)
                return str(result.inserted_id)
            except PyMongoError as e:
                print(f"Error creating replacement request: {e}")
                return None

        with self._lock:
            record = self._normalize_runtime_doc(payload, "replacement")
            self.runtime_records["replacements"].append(record)
            self._persist_runtime_state()
            return record["_id"]

    # =========================================================================
    # CHAT HISTORY OPERATIONS
    # =========================================================================
    def save_chat_message(self, chat_data: Dict) -> Optional[str]:
        payload = self._clone(chat_data)
        payload["timestamp"] = payload.get("timestamp", datetime.utcnow())
        if self._check_connection():
            try:
                result = self.db[COLLECTIONS["chat_history"]].insert_one(payload)
                return str(result.inserted_id)
            except PyMongoError as e:
                print(f"Error saving chat: {e}")
                return None

        with self._lock:
            record = self._normalize_runtime_doc(payload, "chat")
            self.runtime_records["chat_history"].append(record)
            self._persist_runtime_state()
            return record["_id"]

    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        if self._check_connection():
            try:
                chats = list(
                    self.db[COLLECTIONS["chat_history"]]
                    .find({"session_id": session_id})
                    .sort("timestamp", 1)
                    .limit(limit)
                )
                for chat in chats:
                    chat["_id"] = str(chat["_id"])
                return chats
            except PyMongoError as e:
                print(f"Error fetching chat history: {e}")
                return []

        chats = [self._clone(item) for item in self._records("chat_history") if item.get("session_id") == session_id]
        chats.sort(key=lambda item: item.get("timestamp") or datetime.min)
        return chats[:limit]

    def save_chat_summary(self, summary_data: Dict) -> Optional[str]:
        payload = self._clone(summary_data)
        payload["type"] = "summary"
        payload["timestamp"] = payload.get("timestamp", datetime.utcnow())
        if self._check_connection():
            try:
                result = self.db[COLLECTIONS["chat_history"]].insert_one(payload)
                return str(result.inserted_id)
            except PyMongoError as e:
                print(f"Error saving summary: {e}")
                return None

        with self._lock:
            record = self._normalize_runtime_doc(payload, "summary")
            self.runtime_records["chat_history"].append(record)
            self._persist_runtime_state()
            return record["_id"]

    def get_all_chat_summaries(self) -> List[Dict]:
        if self._check_connection():
            try:
                summaries = list(self.db[COLLECTIONS["chat_history"]].find({"type": "summary"}).sort("timestamp", -1))
                for summary in summaries:
                    summary["_id"] = str(summary["_id"])
                return summaries
            except PyMongoError as e:
                print(f"Error fetching summaries: {e}")
                return []

        summaries = [self._clone(item) for item in self._records("chat_history") if item.get("type") == "summary"]
        summaries.sort(key=lambda item: item.get("timestamp") or datetime.min, reverse=True)
        return summaries


db_manager = DatabaseManager()
