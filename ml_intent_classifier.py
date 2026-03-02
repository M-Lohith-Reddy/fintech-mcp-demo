"""
Production-Ready Intent Classifier with ENHANCED Multi-Intent Support
Fixed: Multi-intent detection, entity extraction, and tool mapping
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import pickle
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ProductionIntentClassifier:

    def __init__(self, model_path: str = "models/", datasets_path: str = "datasets/"):
        self.model_path = model_path
        self.datasets_path = datasets_path
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[OneVsRestClassifier] = None
        self.mlb: Optional[MultiLabelBinarizer] = None
        self.intent_mappings = self._load_intent_mappings()
        self.entity_patterns = self._load_entity_patterns()

        model_file = os.path.join(model_path, "production_classifier.pkl")

        if os.path.exists(model_file):
            logger.info("Loading pre-trained model...")
            self.load_model()
        else:
            logger.info("No pre-trained model found. Please run train() first.")

    # ========================
    # INTENT CONFIG - ENHANCED
    # ========================

    def _load_intent_mappings(self) -> Dict[str, Dict[str, Any]]:
        return {
            # -----------------------------------
            # GST CALCULATION
            # -----------------------------------
            "calculate_gst": {
                "tool": "calculate_gst",
                "required_params": ["base_amount", "gst_rate"],
                "keywords": [
                    "calculate gst", "add gst", "gst on", "apply gst",
                    "gst for", "add tax", "gst amount", "how much gst",
                    "compute gst", "find gst", "what is gst", "show gst"
                ],
                "multi_triggers": ["calculate gst", "compute gst", "find gst", "add gst", "gst on"]
            },

            # -----------------------------------
            # REVERSE GST
            # -----------------------------------
            "reverse_gst": {
                "tool": "reverse_calculate_gst",
                "required_params": ["total_amount", "gst_rate"],
                "keywords": [
                    "reverse gst", "remove gst", "exclude gst", "before gst",
                    "without gst", "inclusive gst", "gst included",
                    "base price", "base amount", "excluding"
                ],
                "multi_triggers": ["reverse gst", "remove gst", "exclude gst", "without gst", "base price"]
            },

            # -----------------------------------
            # GST BREAKDOWN
            # -----------------------------------
            "gst_breakdown": {
                "tool": "gst_breakdown",
                "required_params": ["base_amount", "gst_rate"],
                "keywords": [
                    "breakdown", "gst breakdown", "split gst", "cgst",
                    "sgst", "igst", "tax split", "show breakdown",
                    "components", "show cgst", "show sgst"
                ],
                # FIX: Use more specific multi_triggers to avoid false positives.
                # "breakdown" alone is too broad — it was firing on queries that
                # simply mention amounts/rates without requesting a breakdown.
                "multi_triggers": ["gst breakdown", "show breakdown", "split gst", "cgst", "sgst", "igst"]
            },

            # -----------------------------------
            # RATE COMPARISON
            # -----------------------------------
            "compare_rates": {
                "tool": "compare_gst_rates",
                "required_params": ["base_amount", "rates"],
                "keywords": [
                    "compare gst", "compare rates", "which gst rate",
                    "difference between", "rate comparison", "compare",
                    "which rate", "better rate"
                ],
                "multi_triggers": ["compare gst", "compare rates", "rate comparison", "which rate", "difference between"]
            },

            # -----------------------------------
            # GSTIN VALIDATION
            # -----------------------------------
            "validate_gstin": {
                "tool": "validate_gstin",
                "required_params": ["gstin"],
                "keywords": [
                    "validate gstin", "check gstin", "gstin valid",
                    "verify gstin", "is this gstin valid", "validate",
                    "verify", "check"
                ],
                "multi_triggers": ["validate gstin", "verify gstin", "check gstin", "gstin valid"]
            },

            # -----------------------------------
            # COMPANY ONBOARDING
            # -----------------------------------
            "company_guide": {
                "tool": "get_company_onboarding_guide",
                "required_params": [],
                "keywords": [
                    "company onboarding", "register company", "company registration",
                    "how to onboard company", "start company", "onboard organization",
                    "company setup", "register my company", "register a company",
                    "company register", "setting up a company", "set up company"
                ],
                # FIX: Broadened to catch natural phrasings like "register my company"
                # and "how to register". The original triggers were too literal and
                # missed word-order variations (e.g. "register my company" vs "register company").
                "multi_triggers": [
                    "company onboarding", "register company", "company registration",
                    "register my company", "register a company", "start company",
                    "company setup", "company register", "set up company",
                    "registration process"
                ]
            },

            "company_documents": {
                "tool": "get_company_required_documents",
                "required_params": [],
                "keywords": [
                    "documents needed", "required documents", "company documents",
                    "documents for company", "what documents", "document checklist"
                ],
                "multi_triggers": ["required documents", "document checklist", "what documents", "documents needed"]
            },

            "company_field": {
                "tool": "get_validation_formats",
                "required_params": [],
                "keywords": [
                    "pan number", "gst number", "mandatory fields",
                    "field format", "validation format", "format"
                ],
                # FIX: Tightened triggers so "format" alone doesn't fire on
                # company_guide or gst_breakdown queries. Require compound phrases.
                "multi_triggers": ["pan number", "mandatory fields", "field format", "validation format"]
            },

            "company_process": {
                "tool": "get_onboarding_faq",
                "required_params": [],
                "keywords": [
                    "how long", "timeline", "processing time",
                    "approval time", "how many days", "duration"
                ],
                # FIX: "process" alone was matching "registration process" and
                # triggering company_process when company_guide was intended.
                "multi_triggers": ["processing time", "approval time", "how many days", "how long does", "timeline"]
            },

            # -----------------------------------
            # BANK ONBOARDING
            # -----------------------------------
            "bank_guide": {
                "tool": "get_bank_onboarding_guide",
                "required_params": [],
                "keywords": [
                    "bank onboarding", "register bank", "bank registration",
                    "add bank", "supported banks", "bank account", "connect bank"
                ],
                "multi_triggers": ["bank onboarding", "register bank", "add bank", "supported banks"]
            },

            # -----------------------------------
            # VENDOR
            # -----------------------------------
            "vendor_guide": {
                "tool": "get_vendor_onboarding_guide",
                "required_params": [],
                "keywords": [
                    "vendor", "supplier", "seller", "distributor", "vendor onboarding"
                ],
                "multi_triggers": ["vendor onboarding", "add vendor", "register vendor", "supplier onboarding"]
            },
            "redbus_search": {
    "tool": "redbus_search_redirect",
    "required_params": ["source_city", "destination_city"],
    "keywords": [
        "book bus", "search bus", "find bus", "bus from", "buses from",
        "bus tickets", "available buses", "bus timings", "bus schedule",
        "bus seat", "sleeper bus", "AC bus", "overnight bus",
        "plan my journey", "bus to", "travel by bus"
    ],
    "multi_triggers": ["book bus", "search bus", "bus from", "find bus", "bus tickets"]
},
"redbus_booking": {
    "tool": "redbus_booking_redirect",
    "required_params": ["tin"],
    "keywords": [
        "view booking", "check booking", "my booking", "my ticket",
        "booking details", "ticket details", "booking id", "TIN",
        "manage booking", "reservation", "retrieve booking"
    ],
    "multi_triggers": ["view booking", "check booking", "booking id", "my ticket", "TIN"]
},
"redbus_offers": {
    "tool": "redbus_offers_redirect",
    "required_params": [],
    "keywords": [
        "redbus offers", "bus offers", "bus deals", "redbus discount",
        "promo code", "coupon", "cashback", "cheap bus", "bus sale",
        "redbus coupon", "save money", "redbus deals"
    ],
    "multi_triggers": ["redbus offers", "redbus discount", "promo code", "bus deals", "cashback"]
},
"redbus_tracking": {
    "tool": "redbus_tracking_redirect",
    "required_params": ["tin"],
    "keywords": [
        "track bus", "track my bus", "live location", "bus location",
        "where is my bus", "bus tracking", "real time tracking",
        "bus ETA", "is bus on time", "bus arrived", "current position"
    ],
    "multi_triggers": ["track bus", "live location", "bus tracking", "where is my bus", "bus ETA"]
},
"redbus_routes": {
    "tool": "get_popular_routes",
    "required_params": [],
    "keywords": [
        "popular routes", "top routes", "common routes", "best routes",
        "most booked routes", "trending routes", "popular destinations",
        "route list", "where can I go by bus", "cities connected",
        "bus routes", "route suggestions"
    ],
    "multi_triggers": ["popular routes", "top routes", "bus routes", "route list", "cities connected"]
},
"redbus_open": {
    "tool": "open_redbus",
    "required_params": [],
    "keywords": [
        "open redbus", "launch redbus", "go to redbus", "redbus app",
        "redbus website", "visit redbus", "redbus.in", "start redbus",
        "navigate to redbus", "redbus homepage", "open bus booking"
    ],
    "multi_triggers": ["open redbus", "launch redbus", "go to redbus", "redbus app", "redbus website"]
},
        }

    def _load_entity_patterns(self) -> Dict[str, str]:
        return {
            "amount": r"(?:₹|rs\.?|inr|rupees?)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)",
            "percentage": r"(\d+(?:\.\d+)?)\s*(?:%|percent)",
            "gstin": r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}\b",
        }

    # ========================
    # DATA LOADING - ENHANCED FOR MULTI-INTENT
    # ========================

    def load_datasets(self) -> Tuple[List[str], List[List[str]]]:
        logger.info(f"Loading datasets from {self.datasets_path}")

        queries = []
        labels = []

        dataset_mapping = {
            "gst_variations.csv": ["calculate_gst"],
            "reverse_gst_variations.csv": ["reverse_gst"],
            "gst_breakdown_variations.csv": ["gst_breakdown"],
            "D_rate_comparison_400.csv": ["compare_rates"],
            "E_gstin_validation_300.csv": ["validate_gstin"],
            "F_multi_intent_400.csv": "MULTI",
            "_MConverter_eu_Multi_Intent_1500.csv": "MULTI",
            "Company_A_General_Onboarding_500.csv": ["company_guide"],
            "Company_B_Required_Documents_300.csv": ["company_documents"],
            "Company_C_Field_Questions_300.csv": ["company_field"],
            "Company_D_Process_Questions_300.csv": ["company_process"],
            # Vendor onboarding: remove/import/status/reject/register vendor
            "csv-export-2026-02-19.csv": ["vendor_guide"],      # 199 - vendor actions
            "csv-export-2026-02-19__1_.csv": ["vendor_guide"],  # 199 - vendor registration info/docs
            "csv-export-2026-02-19__2_.csv": ["vendor_guide"],  # 299 - how to onboard/create vendor
            # Bank onboarding: IFSC / supported banks / field formats / register bank
            "csv-export-2026-02-19__3_.csv": ["bank_guide"],    # 199 - IFSC validation
            "csv-export-2026-02-19__4_.csv": ["bank_guide"],    # 199 - supported banks
            "csv-export-2026-02-19__5_.csv": ["bank_guide"],    # 299 - bank field formats
            "csv-export-2026-02-19__6_.csv": ["bank_guide"],    # 299 - register bank account
            # Company field/validation errors
            "csv-export-2026-02-19__7_.csv": ["company_field"], # 199 - registration/validation errors
            "redbus_search_500.csv":   ["redbus_search"],
            "redbus_booking_500.csv":  ["redbus_booking"],
            "redbus_offers_500.csv":   ["redbus_offers"],
            "redbus_tracking_500.csv": ["redbus_tracking"],
            "redbus_routes_500.csv":   ["redbus_routes"],
            "redbus_open_500.csv":     ["redbus_open"],
        }

        for filename, intent_label in dataset_mapping.items():
            filepath = os.path.join(self.datasets_path, filename)

            if not os.path.exists(filepath):
                logger.warning(f"⚠ Missing file: {filename}")
                continue

            try:
                df = pd.read_csv(filepath, header=None, names=['query'], on_bad_lines='skip')
                df['query'] = df['query'].astype(str)
                df['query'] = df['query'].str.replace(r'^\d+\.\s*', '', regex=True)
                df['query'] = df['query'].str.strip()
                df = df[df['query'].str.len() > 5]

                for query in df['query'].tolist():
                    if intent_label == "MULTI":
                        detected = self._detect_multi_intents_from_query(query)
                        if detected:
                            queries.append(query)
                            labels.append(detected)
                    else:
                        queries.append(query)
                        labels.append(intent_label)

                logger.info(f"✓ Loaded {len(df)} examples from {filename}")

            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")

        logger.info(f"Total training examples: {len(queries)}")
        return queries, labels

    def _detect_multi_intents_from_query(self, query: str) -> List[str]:
        """
        Multi-intent detection for training data using multi_triggers.
        Uses the same tightened multi_triggers defined in _load_intent_mappings.
        """
        query_lower = query.lower()
        detected = []

        for intent_name, intent_data in self.intent_mappings.items():
            triggers = intent_data.get("multi_triggers", [])
            for trigger in triggers:
                if trigger in query_lower:
                    detected.append(intent_name)
                    break

        return list(set(detected))[:3]

    # ========================
    # TRAINING - ENHANCED
    # ========================

    def train(self):
        texts, labels = self.load_datasets()

        if len(texts) == 0:
            raise ValueError("No training data loaded!")

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.15, random_state=42, stratify=None
        )

        logger.info(f"Training: {len(X_train)} | Test: {len(X_test)}")

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=3000,
            stop_words='english',
            min_df=2,
            sublinear_tf=True
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.mlb = MultiLabelBinarizer()
        y_train_bin = self.mlb.fit_transform(y_train)
        y_test_bin = self.mlb.transform(y_test)

        logger.info(f"Intent classes: {list(self.mlb.classes_)}")

        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                solver='lbfgs',
                C=1.5,
                class_weight='balanced'
            )
        )

        self.classifier.fit(X_train_vec, y_train_bin)

        y_pred = self.classifier.predict(X_test_vec)
        y_pred_proba = self.classifier.predict_proba(X_test_vec)

        # ---- Metrics ----
        # 1. Exact match accuracy (strict — all labels must match, penalised by multi-label rows)
        exact_match = accuracy_score(y_test_bin, y_pred)

        # 2. Hamming score (per-label average) — best overall indicator for multi-label
        from sklearn.metrics import hamming_loss, f1_score
        hamming = 1 - hamming_loss(y_test_bin, y_pred)

        # 3. Macro F1 — balanced across all intents regardless of support
        macro_f1 = f1_score(y_test_bin, y_pred, average="macro", zero_division=0)

        # 4. Micro F1 — weighted by label frequency
        micro_f1 = f1_score(y_test_bin, y_pred, average="micro", zero_division=0)

        logger.info("=" * 70)
        logger.info(f"Exact Match Accuracy : {exact_match * 100:.2f}%  (strict, lower with multi-label data)")
        logger.info(f"Hamming Score        : {hamming * 100:.2f}%  (per-label avg — PRIMARY metric)")
        logger.info(f"Macro F1             : {macro_f1 * 100:.2f}%  (balanced across intents)")
        logger.info(f"Micro F1             : {micro_f1 * 100:.2f}%  (weighted by frequency)")
        logger.info("-" * 70)
        logger.info("Per-Intent Scores (F1 | Accuracy):")

        for idx, intent in enumerate(self.mlb.classes_):
            intent_acc = accuracy_score(y_test_bin[:, idx], y_pred[:, idx])
            intent_f1  = f1_score(y_test_bin[:, idx], y_pred[:, idx], zero_division=0)
            logger.info(f"  {intent:<22} F1: {intent_f1 * 100:.2f}%  |  Acc: {intent_acc * 100:.2f}%")

        logger.info("=" * 70)

        self.save_model()

    # ========================
    # PREDICTION - ENHANCED FOR MULTI-INTENT
    # ========================

    def predict_intents(self, query: str) -> List[str]:
        """
        Enhanced multi-intent prediction with hybrid approach.

        Fix summary:
        - multi_triggers are now compound phrases, not single words.
          This prevents spurious intent matches (e.g. "breakdown" firing
          on any query that happens to contain the word).
        - compare_rates no longer blindly inherits rates from calculate_gst
          context; the tool call builder handles rate scoping separately.
        - company_guide keywords include natural phrasings like
          "register my company" so the keyword fallback catches Test 2.
        """
        if not self.vectorizer or not self.classifier:
            raise ValueError("Model not loaded. Train or load model first.")

        query_lower = query.lower()

        # --- Step 1: ML prediction ---
        X = self.vectorizer.transform([query])
        probabilities = self.classifier.predict_proba(X)[0]

        # Adaptive threshold based on query word count
        query_words = len(query.split())
        if query_words > 15:
            threshold = 0.25
        elif query_words > 10:
            threshold = 0.30
        else:
            threshold = 0.35

        predicted = []
        for idx, prob in enumerate(probabilities):
            if prob > threshold:
                predicted.append(self.mlb.classes_[idx])

        if not predicted:
            predicted.append(self.mlb.classes_[np.argmax(probabilities)])

        # --- Step 2: Keyword-based enhancement ---
        keyword_matched = []

        multi_indicators = [" and ", " also ", " then ", " additionally ", " as well as ", " along with "]
        has_multi_indicator = any(ind in query_lower for ind in multi_indicators)

        if has_multi_indicator:
            # FIX: Only use multi_triggers (compound phrases) here — single-word
            # triggers like "breakdown" or "compare" caused false positives in
            # Tests 4 and 6 because they matched incidental words in the query.
            for intent, config in self.intent_mappings.items():
                for trigger in config.get("multi_triggers", []):
                    if trigger in query_lower:
                        keyword_matched.append(intent)
                        break

        # Regular full-keyword matching (runs regardless of multi-indicator)
        for intent, config in self.intent_mappings.items():
            for kw in config["keywords"]:
                if kw in query_lower and intent not in keyword_matched:
                    keyword_matched.append(intent)
                    break

        # Combine ML and keyword predictions
        combined = list(set(predicted + keyword_matched))

        if len(keyword_matched) >= 2:
            combined = keyword_matched + [p for p in predicted if p not in keyword_matched]

        # --- Step 3: Conflict resolution (post-processing) ---
        combined = self._resolve_intent_conflicts(query_lower, combined)

        return combined[:3]

    def _resolve_intent_conflicts(self, query_lower: str, intents: List[str]) -> List[str]:
        """
        Post-prediction conflict resolution to suppress spurious co-intents.

        Two rules handle the remaining test failures:

        Rule A — company sub-intent suppression (fixes Test 2):
          When company_guide is present WITHOUT an explicit document/process
          keyword in the query, drop company_documents and company_process.
          The ML model over-predicts these sub-intents because they appear
          together in training data, but a plain "register my company" query
          should map to only company_guide.

        Rule B — calculate_gst suppression inside gst_breakdown (fixes Test 6):
          When gst_breakdown is present and the query contains "breakdown" or
          "split" but does NOT contain an explicit "calculate gst" / "gst on"
          trigger, drop calculate_gst. The word "gst" + an amount is enough
          to push calculate_gst above threshold, but if the user said "show
          gst breakdown" they want breakdown, not a raw calculation.
        """
        resolved = list(intents)

        # Rule A: company sub-intent suppression
        if "company_guide" in resolved:
            # Explicit signals that sub-intents are genuinely needed
            doc_signals = ["document", "checklist", "what documents", "required documents"]
            process_signals = ["how long", "timeline", "processing time", "approval", "how many days", "duration"]

            if not any(sig in query_lower for sig in doc_signals):
                resolved = [i for i in resolved if i != "company_documents"]

            if not any(sig in query_lower for sig in process_signals):
                resolved = [i for i in resolved if i != "company_process"]

        # Rule B: suppress calculate_gst when query is purely a breakdown request
        if "gst_breakdown" in resolved and "calculate_gst" in resolved:
            # Explicit calculate signals — keep calculate_gst if present
            calc_signals = ["calculate gst", "compute gst", "find gst", "add gst", "gst amount",
                            "how much gst", "and calculate", "also calculate"]
            if not any(sig in query_lower for sig in calc_signals):
                resolved = [i for i in resolved if i != "calculate_gst"]

        return resolved

    # ========================
    # ENTITY EXTRACTION - ENHANCED
    # ========================

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """
        Enhanced entity extraction with better multi-rate handling.

        Fix: GSTIN contains digits (e.g. 29ABCDE1234F1Z5) that were being
        extracted as amounts before the actual base_amount in the query.
        Solution: extract and remove GSTIN first, then extract percentages,
        then extract amounts from the cleaned query so only real amounts remain.
        """
        entities = {}
        cleaned_query = query.lower()

        # Step 1: Extract GSTIN FIRST and remove it from cleaned_query
        # so its digits don't pollute amount extraction
        gstin_match = re.search(self.entity_patterns["gstin"], query)
        if gstin_match:
            entities["gstin"] = gstin_match.group(0)
            # Remove GSTIN from cleaned query (case-insensitive)
            cleaned_query = re.sub(self.entity_patterns["gstin"], "", cleaned_query, flags=re.IGNORECASE)

        # Step 2: Extract all percentages and remove them
        percent_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", cleaned_query)
        if percent_matches:
            entities["gst_rates"] = [float(p) for p in percent_matches]
            entities["gst_rate"] = float(percent_matches[0])

            for match in percent_matches:
                cleaned_query = cleaned_query.replace(f"{match}%", "")
                cleaned_query = cleaned_query.replace(f"{match} percent", "")

        # Step 3: Extract amounts from the cleaned query (GSTIN digits already removed)
        amount_matches = re.findall(r"(?:₹|rs\.?|inr|rupees?)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)", cleaned_query)
        if amount_matches:
            amounts = [float(a.replace(",", "")) for a in amount_matches]
            entities["amounts"] = amounts
            entities["base_amount"] = amounts[0] if amounts else None
            entities["amount"] = amounts[0] if amounts else None

            if len(amounts) > 1:
                entities["total_amount"] = amounts[1]

        # Default GST rate
        if "gst_rates" not in entities:
            entities["gst_rates"] = [18.0]
            entities["gst_rate"] = 18.0

        # Intra/inter state
        if "inter" in query.lower() or "interstate" in query.lower():
            entities["is_intra_state"] = False
        elif "intra" in query.lower() or "intrastate" in query.lower():
            entities["is_intra_state"] = True

        # Extract RedBus-related entities
        src_match = re.search(r"from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", query)
        dst_match = re.search(r"to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", query)
        if src_match:
            entities["source_city"] = src_match.group(1)
        if dst_match:
            entities["destination_city"] = dst_match.group(1)

        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
        if date_match:
            entities["travel_date"] = date_match.group(1)

        tin_match = re.search(r"\b(TIN\w+)\b", query, re.IGNORECASE)
        if tin_match:
            entities["tin"] = tin_match.group(1)

        return entities

    # ========================
    # FULL PIPELINE - ENHANCED
    # ========================

    def process_query(self, user_message: str):
        """
        Enhanced query processing with corrected multi-intent tool-call logic.
        """
        detected_intents = self.predict_intents(user_message)
        entities = self.extract_entities(user_message)

        logger.info(f"Detected intents: {detected_intents}")
        logger.info(f"Extracted entities: {entities}")

        tool_calls = []

        amount = entities.get("amount") or entities.get("base_amount")
        gst_rates = entities.get("gst_rates", [18.0])
        gstin = entities.get("gstin")
        total_amount = entities.get("total_amount")

        # -------------------------------
        # GST CALCULATION
        # -------------------------------
        if "calculate_gst" in detected_intents and amount:
            if "compare_rates" in detected_intents:
                # FIX (Test 4): When compare_rates is also detected, calculate_gst
                # should only fire for the FIRST/primary rate. The compare_rates tool
                # will handle the full multi-rate comparison — firing calculate_gst
                # for every rate would create duplicate/redundant tool calls.
                tool_calls.append({
                    "tool_name": "calculate_gst",
                    "parameters": {
                        "base_amount": amount,
                        "gst_rate": gst_rates[0]
                    }
                })
            else:
                # No compare_rates: fire once per detected rate (e.g. Test 7)
                for rate in gst_rates:
                    tool_calls.append({
                        "tool_name": "calculate_gst",
                        "parameters": {
                            "base_amount": amount,
                            "gst_rate": rate
                        }
                    })

        # -------------------------------
        # REVERSE GST
        # -------------------------------
        if "reverse_gst" in detected_intents:
            amt = total_amount if total_amount else amount
            if amt:
                tool_calls.append({
                    "tool_name": "reverse_calculate_gst",
                    "parameters": {
                        "total_amount": amt,
                        "gst_rate": gst_rates[0]
                    }
                })

        # -------------------------------
        # GST BREAKDOWN
        # -------------------------------
        if "gst_breakdown" in detected_intents and amount:
            params = {
                "base_amount": amount,
                "gst_rate": gst_rates[0]
            }
            if "is_intra_state" in entities:
                params["is_intra_state"] = entities["is_intra_state"]

            tool_calls.append({
                "tool_name": "gst_breakdown",
                "parameters": params
            })

        # -------------------------------
        # COMPARE GST RATES
        # -------------------------------
        if "compare_rates" in detected_intents and amount:
            # FIX (Test 4): Use all detected rates when >1 rate is present.
            # When only 1 rate is detected alongside "compare", fall back to
            # standard comparison rates so the tool still has meaningful input.
            rates_to_compare = gst_rates if len(gst_rates) > 1 else [5, 12, 18, 28]

            tool_calls.append({
                "tool_name": "compare_gst_rates",
                "parameters": {
                    "base_amount": amount,
                    "rates": rates_to_compare
                }
            })

        # -------------------------------
        # VALIDATE GSTIN
        # -------------------------------
        if "validate_gstin" in detected_intents and gstin:
            tool_calls.append({
                "tool_name": "validate_gstin",
                "parameters": {"gstin": gstin}
            })

        # -------------------------------
        # COMPANY ONBOARDING
        # -------------------------------
        if "company_guide" in detected_intents:
            tool_calls.append({
                "tool_name": "get_company_onboarding_guide",
                "parameters": {}
            })

        if "company_documents" in detected_intents:
            tool_calls.append({
                "tool_name": "get_company_required_documents",
                "parameters": {}
            })

        if "company_field" in detected_intents:
            tool_calls.append({
                "tool_name": "get_validation_formats",
                "parameters": {}
            })

        if "company_process" in detected_intents:
            tool_calls.append({
                "tool_name": "get_onboarding_faq",
                "parameters": {}
            })

        # -------------------------------
        # BANK ONBOARDING
        # -------------------------------
        if "bank_guide" in detected_intents:
            tool_calls.append({
                "tool_name": "get_bank_onboarding_guide",
                "parameters": {}
            })

        # -------------------------------
        # VENDOR
        # -------------------------------
        if "vendor_guide" in detected_intents:
            tool_calls.append({
                "tool_name": "get_vendor_onboarding_guide",
                "parameters": {}
            })

        # -------------------------------
        # REDBUS
        # -------------------------------
        if "redbus_search" in detected_intents:
            tool_calls.append({
                "tool_name": "redbus_search_redirect",
                "parameters": {
                    "source_city":      entities.get("source_city", ""),
                    "destination_city": entities.get("destination_city", ""),
                    "travel_date":      entities.get("travel_date"),
                    "redirect_to":      "web"
                }
            })

        if "redbus_booking" in detected_intents and entities.get("tin"):
            tool_calls.append({
                "tool_name": "redbus_booking_redirect",
                "parameters": {"tin": entities["tin"]}
            })

        if "redbus_offers" in detected_intents:
            tool_calls.append({
                "tool_name": "redbus_offers_redirect",
                "parameters": {"source_city": entities.get("source_city")}
            })

        if "redbus_tracking" in detected_intents and entities.get("tin"):
            tool_calls.append({
                "tool_name": "redbus_tracking_redirect",
                "parameters": {"tin": entities["tin"]}
            })

        if "redbus_routes" in detected_intents:
            tool_calls.append({
                "tool_name": "get_popular_routes",
                "parameters": {"source_city": entities.get("source_city")}
            })

        if "redbus_open" in detected_intents:
            tool_calls.append({
                "tool_name": "open_redbus",
                "parameters": {
                    "redirect_to": "app" if "app" in user_message.lower() else "web"
                }
            })

        return {
            "intents_detected": detected_intents,
            "tool_calls": tool_calls,
            "entities": entities,
            "is_multi_intent": len(detected_intents) > 1,
            "total_tools": len(tool_calls)
        }

    # ========================
    # SAVE / LOAD
    # ========================

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        model_data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "mlb": self.mlb,
            "version": "2.1.0"
        }
        filepath = os.path.join(self.model_path, "production_classifier.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        logger.info(f"✓ Model saved to {filepath}")

    def load_model(self):
        filepath = os.path.join(self.model_path, "production_classifier.pkl")
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        self.vectorizer = model_data["vectorizer"]
        self.classifier = model_data["classifier"]
        self.mlb = model_data["mlb"]
        logger.info(f"✓ Model loaded (v{model_data.get('version', '1.0.0')})")


# Global instance
intent_classifier = ProductionIntentClassifier()