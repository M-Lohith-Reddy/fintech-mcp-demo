"""
Production-Ready Intent Classifier — Bank AI Assistant
Covers: Payments, B2B, GST, EPF, ESIC, Payroll, Taxes,
        Insurance, Custom/SEZ, Bank Statement,
        Account Management, Transactions, Dues, Dashboard & Support
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
    # INTENT CONFIG
    # ========================

    def _load_intent_mappings(self) -> Dict[str, Dict[str, Any]]:
        return {

            # ── CORE PAYMENT ──────────────────────────────────────────
            "initiate_payment": {
                "tool": "initiate_payment",
                "required_params": ["beneficiary_id", "amount", "payment_mode"],
                "keywords": [
                    "send money", "transfer money", "initiate payment", "make payment",
                    "pay to", "transfer to", "send funds", "fund transfer",
                    "neft payment", "rtgs payment", "imps payment", "upi payment"
                ],
                "multi_triggers": ["initiate payment", "send money", "fund transfer", "transfer money", "pay to"]
            },
            "get_payment_status": {
                "tool": "get_payment_status",
                "required_params": ["transaction_id"],
                "keywords": [
                    "payment status", "track payment", "transaction status",
                    "check payment", "payment update", "utr status"
                ],
                "multi_triggers": ["payment status", "transaction status", "track payment", "check payment"]
            },
            "cancel_payment": {
                "tool": "cancel_payment",
                "required_params": ["transaction_id"],
                "keywords": [
                    "cancel payment", "stop payment", "abort payment", "revoke payment"
                ],
                "multi_triggers": ["cancel payment", "stop payment", "abort payment"]
            },
            "retry_payment": {
                "tool": "retry_payment",
                "required_params": ["transaction_id"],
                "keywords": [
                    "retry payment", "resend payment", "redo payment", "payment failed retry"
                ],
                "multi_triggers": ["retry payment", "resend payment", "redo payment"]
            },
            "get_payment_receipt": {
                "tool": "get_payment_receipt",
                "required_params": ["transaction_id"],
                "keywords": [
                    "payment receipt", "download receipt", "payment acknowledgment",
                    "receipt download", "transaction receipt"
                ],
                "multi_triggers": ["payment receipt", "download receipt", "transaction receipt"]
            },
            "validate_beneficiary": {
                "tool": "validate_beneficiary",
                "required_params": [],
                "keywords": [
                    "validate account", "verify account", "check account",
                    "validate upi", "verify beneficiary", "validate beneficiary"
                ],
                "multi_triggers": ["validate beneficiary", "verify account", "validate account"]
            },

            # ── UPLOAD PAYMENT ────────────────────────────────────────
            "upload_bulk_payment": {
                "tool": "upload_bulk_payment",
                "required_params": ["file_name", "file_base64"],
                "keywords": [
                    "bulk payment", "upload payment", "batch payment",
                    "bulk transfer", "upload file payment", "multiple payments"
                ],
                "multi_triggers": ["bulk payment", "upload payment", "batch payment", "bulk transfer"]
            },
            "validate_payment_file": {
                "tool": "validate_payment_file",
                "required_params": ["upload_id"],
                "keywords": [
                    "validate payment file", "check payment file", "verify upload"
                ],
                "multi_triggers": ["validate payment file", "verify upload", "check payment file"]
            },

            # ── B2B ───────────────────────────────────────────────────
            "onboard_business_partner": {
                "tool": "onboard_business_partner",
                "required_params": ["company_name", "gstin", "pan", "contact_email", "contact_phone"],
                "keywords": [
                    "onboard partner", "add partner", "register partner",
                    "new business partner", "b2b onboarding", "partner registration"
                ],
                "multi_triggers": ["onboard partner", "add partner", "b2b onboarding", "partner registration"]
            },
            "send_invoice": {
                "tool": "send_invoice",
                "required_params": ["partner_id", "invoice_number", "invoice_date", "due_date", "amount"],
                "keywords": [
                    "send invoice", "create invoice", "raise invoice",
                    "generate invoice", "invoice to partner"
                ],
                "multi_triggers": ["send invoice", "raise invoice", "generate invoice", "create invoice"]
            },
            "get_received_invoices": {
                "tool": "get_received_invoices",
                "required_params": [],
                "keywords": [
                    "received invoices", "incoming invoices", "bills received",
                    "pending invoices", "view invoices", "all invoices"
                ],
                "multi_triggers": ["received invoices", "incoming invoices", "pending invoices"]
            },
            "acknowledge_payment": {
                "tool": "acknowledge_payment",
                "required_params": ["invoice_id", "transaction_id"],
                "keywords": [
                    "acknowledge payment", "payment acknowledgment",
                    "confirm payment", "payment confirmation"
                ],
                "multi_triggers": ["acknowledge payment", "payment acknowledgment", "confirm payment"]
            },
            "create_proforma_invoice": {
                "tool": "create_proforma_invoice",
                "required_params": ["partner_id", "validity_date", "amount", "description"],
                "keywords": [
                    "proforma invoice", "pre-sale invoice", "create proforma",
                    "proforma document", "quotation invoice"
                ],
                "multi_triggers": ["proforma invoice", "create proforma", "pre-sale invoice"]
            },
            "create_cd_note": {
                "tool": "create_cd_note",
                "required_params": ["partner_id", "note_type", "original_invoice_id", "amount", "reason"],
                "keywords": [
                    "credit note", "debit note", "cd note", "adjustment note",
                    "create credit note", "create debit note"
                ],
                "multi_triggers": ["credit note", "debit note", "cd note", "adjustment note"]
            },
            "create_purchase_order": {
                "tool": "create_purchase_order",
                "required_params": ["partner_id", "po_date", "delivery_date", "amount", "description"],
                "keywords": [
                    "purchase order", "create po", "raise po",
                    "new purchase order", "vendor order"
                ],
                "multi_triggers": ["purchase order", "create po", "raise po", "vendor order"]
            },

            # ── INSURANCE ─────────────────────────────────────────────
            "fetch_insurance_dues": {
                "tool": "fetch_insurance_dues",
                "required_params": [],
                "keywords": [
                    "insurance dues", "premium due", "insurance premium",
                    "policy due", "insurance payment due"
                ],
                "multi_triggers": ["insurance dues", "premium due", "policy due", "insurance premium due"]
            },
            "pay_insurance_premium": {
                "tool": "pay_insurance_premium",
                "required_params": ["policy_number", "amount"],
                "keywords": [
                    "pay insurance", "pay premium", "insurance payment",
                    "premium payment", "policy payment"
                ],
                "multi_triggers": ["pay insurance", "pay premium", "insurance payment", "premium payment"]
            },
            "get_insurance_payment_history": {
                "tool": "get_insurance_payment_history",
                "required_params": [],
                "keywords": [
                    "insurance history", "premium history", "insurance payments",
                    "past insurance", "policy payment history"
                ],
                "multi_triggers": ["insurance history", "premium history", "insurance payment history"]
            },

            # ── BANK STATEMENT ────────────────────────────────────────
            "fetch_bank_statement": {
                "tool": "fetch_bank_statement",
                "required_params": ["account_number", "from_date", "to_date"],
                "keywords": [
                    "bank statement", "account statement", "fetch statement",
                    "view statement", "statement for"
                ],
                "multi_triggers": ["bank statement", "account statement", "fetch statement"]
            },
            "download_bank_statement": {
                "tool": "download_bank_statement",
                "required_params": ["account_number", "from_date", "to_date"],
                "keywords": [
                    "download statement", "export statement", "statement pdf",
                    "statement excel", "statement download"
                ],
                "multi_triggers": ["download statement", "export statement", "statement pdf", "statement download"]
            },
            "get_account_balance": {
                "tool": "get_account_balance",
                "required_params": ["account_number"],
                "keywords": [
                    "account balance", "check balance", "available balance",
                    "current balance", "balance inquiry"
                ],
                "multi_triggers": ["account balance", "check balance", "available balance"]
            },
            "get_transaction_history": {
                "tool": "get_transaction_history",
                "required_params": ["account_number"],
                "keywords": [
                    "transaction history", "recent transactions", "past transactions",
                    "view transactions", "transaction list"
                ],
                "multi_triggers": ["transaction history", "recent transactions", "past transactions"]
            },

            # ── CUSTOM / SEZ ──────────────────────────────────────────
            "pay_custom_duty": {
                "tool": "pay_custom_duty",
                "required_params": ["bill_of_entry_number", "amount", "port_code", "importer_code"],
                "keywords": [
                    "custom duty", "pay custom", "customs payment",
                    "import duty", "sez payment", "customs duty"
                ],
                "multi_triggers": ["custom duty", "pay custom", "customs payment", "import duty"]
            },
            "track_custom_duty_payment": {
                "tool": "track_custom_duty_payment",
                "required_params": ["transaction_id"],
                "keywords": [
                    "track customs", "custom payment status", "duty payment status",
                    "customs tracking"
                ],
                "multi_triggers": ["track customs", "custom payment status", "customs tracking"]
            },
            "get_custom_duty_history": {
                "tool": "get_custom_duty_history",
                "required_params": [],
                "keywords": [
                    "customs history", "custom duty history", "past customs payments",
                    "import duty history"
                ],
                "multi_triggers": ["customs history", "custom duty history", "import duty history"]
            },

            # ── GST ───────────────────────────────────────────────────
            "fetch_gst_dues": {
                "tool": "fetch_gst_dues",
                "required_params": ["gstin"],
                "keywords": [
                    "gst dues", "gst pending", "gst liability",
                    "gst return due", "pending gst"
                ],
                "multi_triggers": ["gst dues", "pending gst", "gst return due", "gst liability"]
            },
            "pay_gst": {
                "tool": "pay_gst",
                "required_params": ["gstin", "challan_number", "amount", "tax_type"],
                "keywords": [
                    "pay gst", "gst payment", "pay igst",
                    "pay cgst", "pay sgst", "pay cess"
                ],
                "multi_triggers": ["pay gst", "gst payment", "pay igst", "pay cgst"]
            },
            "create_gst_challan": {
                "tool": "create_gst_challan",
                "required_params": ["gstin", "return_period"],
                "keywords": [
                    "gst challan", "create challan", "pmt-06", "generate challan",
                    "gst challan creation"
                ],
                "multi_triggers": ["gst challan", "create challan", "pmt-06", "generate challan"]
            },
            "get_gst_payment_history": {
                "tool": "get_gst_payment_history",
                "required_params": ["gstin"],
                "keywords": [
                    "gst payment history", "past gst payments",
                    "gst history", "previous gst payments"
                ],
                "multi_triggers": ["gst payment history", "gst history", "past gst payments"]
            },

            # ── ESIC ──────────────────────────────────────────────────
            "fetch_esic_dues": {
                "tool": "fetch_esic_dues",
                "required_params": ["establishment_code", "month"],
                "keywords": [
                    "esic dues", "esic contribution", "esic pending",
                    "esic payment due", "employee state insurance"
                ],
                "multi_triggers": ["esic dues", "esic contribution", "esic pending", "esic payment due"]
            },
            "pay_esic": {
                "tool": "pay_esic",
                "required_params": ["establishment_code", "month", "amount"],
                "keywords": [
                    "pay esic", "esic payment", "esic challan",
                    "employee insurance payment"
                ],
                "multi_triggers": ["pay esic", "esic payment", "esic challan"]
            },
            "get_esic_payment_history": {
                "tool": "get_esic_payment_history",
                "required_params": ["establishment_code"],
                "keywords": [
                    "esic history", "esic payment history",
                    "past esic", "esic records"
                ],
                "multi_triggers": ["esic history", "esic payment history", "past esic"]
            },

            # ── EPF ───────────────────────────────────────────────────
            "fetch_epf_dues": {
                "tool": "fetch_epf_dues",
                "required_params": ["establishment_id", "month"],
                "keywords": [
                    "epf dues", "pf dues", "epf contribution",
                    "provident fund due", "pf pending"
                ],
                "multi_triggers": ["epf dues", "pf dues", "epf contribution", "provident fund due"]
            },
            "pay_epf": {
                "tool": "pay_epf",
                "required_params": ["establishment_id", "month", "amount"],
                "keywords": [
                    "pay epf", "pf payment", "epf challan",
                    "provident fund payment", "pay pf"
                ],
                "multi_triggers": ["pay epf", "pf payment", "epf challan", "pay pf"]
            },
            "get_epf_payment_history": {
                "tool": "get_epf_payment_history",
                "required_params": ["establishment_id"],
                "keywords": [
                    "epf history", "pf history", "epf payment history",
                    "past pf payments", "epf records"
                ],
                "multi_triggers": ["epf history", "pf history", "epf payment history"]
            },

            # ── PAYROLL ───────────────────────────────────────────────
            "fetch_payroll_summary": {
                "tool": "fetch_payroll_summary",
                "required_params": ["month"],
                "keywords": [
                    "payroll summary", "salary summary", "payroll report",
                    "employee salary", "payroll details"
                ],
                "multi_triggers": ["payroll summary", "salary summary", "payroll report"]
            },
            "process_payroll": {
                "tool": "process_payroll",
                "required_params": ["month", "account_number", "approved_by"],
                "keywords": [
                    "process payroll", "run payroll", "salary disbursement",
                    "disburse salary", "pay salaries", "payroll processing"
                ],
                "multi_triggers": ["process payroll", "run payroll", "salary disbursement", "disburse salary"]
            },
            "get_payroll_history": {
                "tool": "get_payroll_history",
                "required_params": [],
                "keywords": [
                    "payroll history", "salary history", "past payroll",
                    "payroll records", "previous salaries"
                ],
                "multi_triggers": ["payroll history", "salary history", "past payroll"]
            },

            # ── TAXES ─────────────────────────────────────────────────
            "fetch_tax_dues": {
                "tool": "fetch_tax_dues",
                "required_params": ["pan"],
                "keywords": [
                    "tax dues", "pending tax", "tax liability",
                    "tds dues", "advance tax due", "tax outstanding"
                ],
                "multi_triggers": ["tax dues", "pending tax", "tds dues", "advance tax due"]
            },
            "pay_direct_tax": {
                "tool": "pay_direct_tax",
                "required_params": ["pan", "tax_type", "assessment_year", "amount", "challan_type"],
                "keywords": [
                    "pay tds", "direct tax", "pay advance tax",
                    "income tax payment", "self assessment tax"
                ],
                "multi_triggers": ["pay tds", "direct tax", "pay advance tax", "income tax payment"]
            },
            "pay_state_tax": {
                "tool": "pay_state_tax",
                "required_params": ["state", "tax_category", "amount", "assessment_period"],
                "keywords": [
                    "state tax", "professional tax", "pay state tax",
                    "vat payment", "state tax payment"
                ],
                "multi_triggers": ["state tax", "professional tax", "pay state tax", "vat payment"]
            },
            "pay_bulk_tax": {
                "tool": "pay_bulk_tax",
                "required_params": ["file_name", "file_base64", "tax_type"],
                "keywords": [
                    "bulk tax", "bulk tds", "tax bulk payment",
                    "multiple tax payments", "bulk tax payment"
                ],
                "multi_triggers": ["bulk tax", "bulk tds", "tax bulk payment", "multiple tax payments"]
            },
            "get_tax_payment_history": {
                "tool": "get_tax_payment_history",
                "required_params": ["pan"],
                "keywords": [
                    "tax history", "tax payment history", "past tax payments",
                    "tds history", "tax records"
                ],
                "multi_triggers": ["tax history", "tax payment history", "tds history", "past tax payments"]
            },

            # ── ACCOUNT MANAGEMENT ────────────────────────────────────
            "get_account_summary": {
                "tool": "get_account_summary",
                "required_params": [],
                "keywords": [
                    "account summary", "all accounts", "my accounts",
                    "linked accounts summary", "accounts overview"
                ],
                "multi_triggers": ["account summary", "all accounts", "accounts overview"]
            },
            "get_account_details": {
                "tool": "get_account_details",
                "required_params": ["account_number"],
                "keywords": [
                    "account details", "account info", "bank account details",
                    "ifsc details", "account information"
                ],
                "multi_triggers": ["account details", "account info", "bank account details"]
            },
            "get_linked_accounts": {
                "tool": "get_linked_accounts",
                "required_params": [],
                "keywords": [
                    "linked accounts", "all linked", "connected accounts",
                    "my bank accounts", "list accounts"
                ],
                "multi_triggers": ["linked accounts", "connected accounts", "list accounts"]
            },
            "set_default_account": {
                "tool": "set_default_account",
                "required_params": ["account_number"],
                "keywords": [
                    "set default account", "primary account", "default bank account",
                    "make default", "set primary"
                ],
                "multi_triggers": ["set default account", "primary account", "make default"]
            },

            # ── TRANSACTION & HISTORY ─────────────────────────────────
            "search_transactions": {
                "tool": "search_transactions",
                "required_params": [],
                "keywords": [
                    "search transactions", "find transaction", "filter transactions",
                    "transaction search", "look up transaction"
                ],
                "multi_triggers": ["search transactions", "find transaction", "filter transactions"]
            },
            "get_transaction_details": {
                "tool": "get_transaction_details",
                "required_params": ["transaction_id"],
                "keywords": [
                    "transaction details", "transaction info",
                    "detail of transaction", "transaction breakdown"
                ],
                "multi_triggers": ["transaction details", "transaction info", "transaction breakdown"]
            },
            "download_transaction_report": {
                "tool": "download_transaction_report",
                "required_params": ["from_date", "to_date"],
                "keywords": [
                    "transaction report", "download transactions", "export transactions",
                    "transaction export", "transactions excel"
                ],
                "multi_triggers": ["transaction report", "download transactions", "export transactions"]
            },
            "get_pending_transactions": {
                "tool": "get_pending_transactions",
                "required_params": [],
                "keywords": [
                    "pending transactions", "in-process payments",
                    "outstanding transactions", "pending payments"
                ],
                "multi_triggers": ["pending transactions", "in-process payments", "outstanding transactions"]
            },

            # ── DUES & REMINDERS ──────────────────────────────────────
            "get_upcoming_dues": {
                "tool": "get_upcoming_dues",
                "required_params": [],
                "keywords": [
                    "upcoming dues", "all dues", "what is due",
                    "payment dues", "scheduled dues", "due payments"
                ],
                "multi_triggers": ["upcoming dues", "all dues", "payment dues", "due payments"]
            },
            "get_overdue_payments": {
                "tool": "get_overdue_payments",
                "required_params": [],
                "keywords": [
                    "overdue payments", "missed payments", "overdue dues",
                    "late payments", "payment overdue"
                ],
                "multi_triggers": ["overdue payments", "missed payments", "late payments"]
            },
            "set_payment_reminder": {
                "tool": "set_payment_reminder",
                "required_params": ["title", "due_date"],
                "keywords": [
                    "set reminder", "payment reminder", "remind me",
                    "add reminder", "due date reminder"
                ],
                "multi_triggers": ["set reminder", "payment reminder", "due date reminder"]
            },
            "get_reminder_list": {
                "tool": "get_reminder_list",
                "required_params": [],
                "keywords": [
                    "reminder list", "my reminders", "all reminders",
                    "view reminders", "active reminders"
                ],
                "multi_triggers": ["reminder list", "my reminders", "view reminders"]
            },
            "delete_reminder": {
                "tool": "delete_reminder",
                "required_params": ["reminder_id"],
                "keywords": [
                    "delete reminder", "remove reminder",
                    "cancel reminder", "clear reminder"
                ],
                "multi_triggers": ["delete reminder", "remove reminder", "cancel reminder"]
            },

            # ── DASHBOARD & ANALYTICS ─────────────────────────────────
            "get_dashboard_summary": {
                "tool": "get_dashboard_summary",
                "required_params": [],
                "keywords": [
                    "dashboard", "overview", "account health",
                    "financial summary", "dashboard summary"
                ],
                "multi_triggers": ["dashboard summary", "account health", "financial summary", "overview"]
            },
            "get_spending_analytics": {
                "tool": "get_spending_analytics",
                "required_params": [],
                "keywords": [
                    "spending analytics", "expense breakdown", "category wise spending",
                    "spending report", "where am i spending"
                ],
                "multi_triggers": ["spending analytics", "expense breakdown", "spending report"]
            },
            "get_cashflow_summary": {
                "tool": "get_cashflow_summary",
                "required_params": [],
                "keywords": [
                    "cashflow", "cash flow", "inflow outflow",
                    "net cashflow", "cash summary"
                ],
                "multi_triggers": ["cashflow", "cash flow", "inflow outflow", "net cashflow"]
            },
            "get_monthly_report": {
                "tool": "get_monthly_report",
                "required_params": ["month"],
                "keywords": [
                    "monthly report", "month report", "financial report",
                    "monthly summary", "report for month"
                ],
                "multi_triggers": ["monthly report", "monthly summary", "financial report"]
            },
            "get_vendor_payment_summary": {
                "tool": "get_vendor_payment_summary",
                "required_params": [],
                "keywords": [
                    "vendor payment summary", "vendor wise payment",
                    "top vendors", "vendor payments"
                ],
                "multi_triggers": ["vendor payment summary", "vendor wise payment", "top vendors"]
            },

            # ── COMPANY MANAGEMENT ────────────────────────────────────
            "get_company_profile": {
                "tool": "get_company_profile",
                "required_params": [],
                "keywords": [
                    "company profile", "company details", "business profile",
                    "company info", "organization details"
                ],
                "multi_triggers": ["company profile", "company details", "business profile"]
            },
            "update_company_details": {
                "tool": "update_company_details",
                "required_params": ["field", "value"],
                "keywords": [
                    "update company", "change company details", "edit company",
                    "modify company info", "update business details"
                ],
                "multi_triggers": ["update company", "change company details", "edit company"]
            },
            "get_gst_profile": {
                "tool": "get_gst_profile",
                "required_params": [],
                "keywords": [
                    "gst profile", "linked gst", "gst numbers",
                    "company gstin", "registered gst"
                ],
                "multi_triggers": ["gst profile", "linked gst", "company gstin"]
            },
            "get_authorized_signatories": {
                "tool": "get_authorized_signatories",
                "required_params": [],
                "keywords": [
                    "authorized signatories", "authorized persons",
                    "company signatories", "who can sign"
                ],
                "multi_triggers": ["authorized signatories", "authorized persons", "company signatories"]
            },
            "manage_user_roles": {
                "tool": "manage_user_roles",
                "required_params": ["user_id", "role", "action"],
                "keywords": [
                    "user role", "assign role", "change role",
                    "user permission", "maker checker"
                ],
                "multi_triggers": ["user role", "assign role", "change role", "user permission"]
            },

            # ── GST CALCULATOR (→ gst_client_manager / server.py) ─────
            "calculate_gst": {
                "tool": "calculate_gst",
                "required_params": ["base_amount", "gst_rate"],
                "keywords": [
                    "calculate gst", "add gst", "gst on", "apply gst",
                    "gst for", "add tax", "gst amount", "how much gst",
                    "compute gst", "find gst", "what is gst on"
                ],
                "multi_triggers": ["calculate gst", "compute gst", "find gst", "add gst", "gst on"]
            },
            "reverse_gst": {
                "tool": "reverse_calculate_gst",
                "required_params": ["total_amount", "gst_rate"],
                "keywords": [
                    "reverse gst", "remove gst", "exclude gst", "before gst",
                    "without gst", "inclusive gst", "gst included",
                    "base price from total", "base amount from total", "excluding gst"
                ],
                "multi_triggers": ["reverse gst", "remove gst", "exclude gst", "without gst", "base price"]
            },
            "gst_breakdown": {
                "tool": "gst_breakdown",
                "required_params": ["base_amount", "gst_rate"],
                "keywords": [
                    "gst breakdown", "split gst", "cgst sgst", "igst breakdown",
                    "tax split", "show breakdown", "cgst and sgst", "show cgst",
                    "show sgst", "intra state gst", "inter state gst"
                ],
                "multi_triggers": ["gst breakdown", "show breakdown", "split gst", "cgst sgst", "igst breakdown"]
            },
            "compare_rates": {
                "tool": "compare_gst_rates",
                "required_params": ["base_amount", "rates"],
                "keywords": [
                    "compare gst", "compare rates", "compare gst rates",
                    "which gst rate", "rate comparison", "different gst rates",
                    "better rate", "gst rate difference"
                ],
                "multi_triggers": ["compare gst", "compare rates", "rate comparison", "gst rate difference"]
            },
            "validate_gstin": {
                "tool": "validate_gstin",
                "required_params": ["gstin"],
                "keywords": [
                    "validate gstin", "check gstin", "gstin valid",
                    "verify gstin", "is gstin valid", "gstin check",
                    "gstin verification", "validate gst number"
                ],
                "multi_triggers": ["validate gstin", "verify gstin", "check gstin", "gstin valid"]
            },

            # ── ONBOARDING INFO (→ info_client_manager / info_server.py) ──
            "company_guide": {
                "tool": "get_company_onboarding_guide",
                "required_params": [],
                "keywords": [
                    "company onboarding", "register company", "company registration",
                    "how to onboard company", "start company", "onboard organization",
                    "company setup", "register my company", "register a company",
                    "setting up a company", "set up company", "company register"
                ],
                "multi_triggers": [
                    "company onboarding", "register company", "company registration",
                    "register my company", "register a company", "company setup",
                    "set up company", "registration process"
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
                    "pan number format", "gst number format", "mandatory fields",
                    "field format", "validation format", "field validation"
                ],
                "multi_triggers": ["pan number format", "mandatory fields", "field format", "validation format"]
            },
            "company_process": {
                "tool": "get_onboarding_faq",
                "required_params": [],
                "keywords": [
                    "how long onboarding", "onboarding timeline", "processing time",
                    "approval time", "how many days to register", "onboarding duration"
                ],
                "multi_triggers": ["processing time", "approval time", "how many days", "onboarding timeline"]
            },
            "bank_guide": {
                "tool": "get_bank_onboarding_guide",
                "required_params": [],
                "keywords": [
                    "bank onboarding", "register bank", "bank registration",
                    "add bank account", "supported banks", "connect bank account",
                    "how to add bank", "bank account onboarding"
                ],
                "multi_triggers": ["bank onboarding", "register bank", "add bank account", "supported banks"]
            },
            "vendor_guide": {
                "tool": "get_vendor_onboarding_guide",
                "required_params": [],
                "keywords": [
                    "vendor onboarding", "add vendor", "register vendor",
                    "supplier onboarding", "how to add vendor", "vendor registration",
                    "onboard supplier", "create vendor"
                ],
                "multi_triggers": ["vendor onboarding", "add vendor", "register vendor", "supplier onboarding"]
            },

            # ── SUPPORT ───────────────────────────────────────────────
            "raise_support_ticket": {
                "tool": "raise_support_ticket",
                "required_params": ["category", "subject", "description"],
                "keywords": [
                    "support ticket", "raise ticket", "create ticket",
                    "report issue", "raise complaint", "log issue"
                ],
                "multi_triggers": ["support ticket", "raise ticket", "create ticket", "report issue"]
            },
            "get_ticket_history": {
                "tool": "get_ticket_history",
                "required_params": [],
                "keywords": [
                    "ticket history", "my tickets", "all tickets",
                    "past tickets", "support history"
                ],
                "multi_triggers": ["ticket history", "my tickets", "past tickets"]
            },
            "chat_with_support": {
                "tool": "chat_with_support",
                "required_params": ["issue_summary"],
                "keywords": [
                    "chat support", "live support", "talk to agent",
                    "chat with agent", "live chat"
                ],
                "multi_triggers": ["chat support", "live support", "talk to agent", "live chat"]
            },
            "get_contact_details": {
                "tool": "get_contact_details",
                "required_params": [],
                "keywords": [
                    "contact details", "support contact", "helpline",
                    "customer care", "contact number"
                ],
                "multi_triggers": ["contact details", "support contact", "helpline", "customer care"]
            },
        }

    def _load_entity_patterns(self) -> Dict[str, str]:
        return {
            "amount":     r"(?:₹|rs\.?|inr|rupees?)?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)",
            "percentage": r"(\d+(?:\.\d+)?)\s*(?:%|percent)",
            "gstin":      r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}\b",
            "pan":        r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
            "account":    r"\b\d{9,18}\b",
            "ifsc":       r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
            "date":       r"\b(\d{4}-\d{2}-\d{2})\b",
            "month":      r"\b(0[1-9]|1[0-2])-(\d{4})\b",
        }

    # ========================
    # DATA LOADING
    # ========================

    def load_datasets(self) -> Tuple[List[str], List[List[str]]]:
        logger.info(f"Loading datasets from {self.datasets_path}")

        queries = []
        labels  = []

        dataset_mapping = dataset_mapping = {
    # ── Payment 
    "payment_initiate_500.csv":                  ["initiate_payment"],
    "payment_status_300.csv":                    ["get_payment_status"],
    "payment_bulk_upload_300.csv":               ["upload_bulk_payment"],

    # ── B2B 
    "b2b_partner_onboard_400.csv":               ["onboard_business_partner"],
    "b2b_invoice_send_300.csv":                  ["send_invoice"],
    "b2b_invoice_receive_300.csv":               ["get_received_invoices"],
    "b2b_purchase_order_300.csv":                ["create_purchase_order"],

    # ── Compliance 
    "gst_pay_400.csv":                           ["pay_gst"],
    "gst_challan_300.csv":                       ["create_gst_challan"],
    "epf_pay_400.csv":                           ["pay_epf"],
    "esic_pay_400.csv":                          ["pay_esic"],
    "payroll_process_400.csv":                   ["process_payroll"],
    "tax_direct_400.csv":                        ["pay_direct_tax"],
    "tax_state_300.csv":                         ["pay_state_tax"],
    "pay_insurance_premium_200.csv":             ["pay_insurance_premium"],       # NEW ✓

    # ── Account & Transactions 
    "account_balance_300.csv":                   ["get_account_balance"],
    "account_statement_300.csv":                 ["fetch_bank_statement"],
    "transaction_history_300.csv":               ["get_transaction_history"],
    "get_account_details_200.csv":               ["get_account_details"],         # NEW ✓

    # ── Dashboard & Dues 
    "dashboard_400.csv":                         ["get_dashboard_summary"],
    "dues_upcoming_300.csv":                     ["get_upcoming_dues"],
    "dues_upcoming_boost_400.csv":               ["get_upcoming_dues"],           # NEW ✓

    # ── Fetch Dues 
    "fetch_epf_dues_200.csv":                    ["fetch_epf_dues"],              # NEW ✓
    "fetch_gst_dues_200.csv":                    ["fetch_gst_dues"],              # NEW ✓
    "fetch_esic_dues_200.csv":                   ["fetch_esic_dues"],             # NEW ✓
    "fetch_tax_dues_200.csv":                    ["fetch_tax_dues"],              # NEW ✓

    # ── Reminders
    "set_payment_reminder_200.csv":              ["set_payment_reminder"],        # NEW ✓

    # ── Support 
    "support_ticket_300.csv":                    ["raise_support_ticket"],

    # ── GST Calculator 
    "gst_variations.csv":                        ["calculate_gst"],
    "reverse_gst_variations.csv":                ["reverse_gst"],
    "gst_breakdown_variations.csv":              ["gst_breakdown"],
    "D_rate_comparison_400.csv":                 ["compare_rates"],
    "E_gstin_validation_300.csv":                ["validate_gstin"],
    "get_gst_profile_200.csv":                   ["get_gst_profile"],             # NEW ✓
    "calc_compare_boost_600.csv":                ["calculate_gst", "compare_rates"],  # NEW ✓ Test 4 fix

    # ── Onboarding — Company 
    "Company_A_General_Onboarding_500.csv":      ["company_guide"],
    "Company_B_Required_Documents_300.csv":      ["company_documents"],
    "Company_C_Field_Questions_300.csv":         ["company_field"],
    "Company_D_Process_Questions_300.csv":       ["company_process"],
    "company_process_boost_300.csv":             ["company_process"],             # NEW ✓
    "company_profile_300.csv":                   ["get_company_profile"],         # NEW ✓
    "company_update_300.csv":                    ["update_company_details"],      # NEW ✓

    # ── Onboarding — Bank 
    "csv-export-2026-02-19__3_.csv":             ["bank_guide"],
    "csv-export-2026-02-19__4_.csv":             ["bank_guide"],
    "csv-export-2026-02-19__5_.csv":             ["bank_guide"],
    "csv-export-2026-02-19__6_.csv":             ["bank_guide"],
    "csv-export-2026-02-19__7_.csv":             ["bank_guide"],

    # ── Onboarding — Vendor 
    "csv-export-2026-02-19.csv":                 ["vendor_guide"],
    "csv-export-2026-02-19__1_.csv":             ["vendor_guide"],
    "csv-export-2026-02-19__2_.csv":             ["vendor_guide"],

    # ── Multi-Intent 
    "multi_intent_bank_600.csv":                 "MULTI",
    "F_multi_intent_400.csv":                    "MULTI",
    "_MConverter_eu_Multi_Intent_1500.csv":      "MULTI",
}

        for filename, intent_label in dataset_mapping.items():
            filepath = os.path.join(self.datasets_path, filename)

            if not os.path.exists(filepath):
                logger.warning(f"⚠ Missing file: {filename}")
                continue

            try:
                df = pd.read_csv(filepath, header=None, names=["query"], on_bad_lines="skip")
                df["query"] = df["query"].astype(str)
                df["query"] = df["query"].str.replace(r"^\d+\.\s*", "", regex=True).str.strip()
                df = df[df["query"].str.len() > 5]

                for query in df["query"].tolist():
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
        """Multi-intent detection for training data using multi_triggers."""
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
    # TRAINING
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
            max_features=5000,
            stop_words="english",
            min_df=2,
            sublinear_tf=True
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec  = self.vectorizer.transform(X_test)

        self.mlb = MultiLabelBinarizer()
        y_train_bin = self.mlb.fit_transform(y_train)
        y_test_bin  = self.mlb.transform(y_test)

        logger.info(f"Intent classes: {list(self.mlb.classes_)}")

        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                C=1.5,
                class_weight="balanced"
            )
        )

        self.classifier.fit(X_train_vec, y_train_bin)

        # capture probabilities for threshold experiments
        probabilities_test = self.classifier.predict_proba(X_test_vec)
        # perform default binary prediction for reporting
        y_pred = self.classifier.predict(X_test_vec)

        # analyze results
        self._log_label_distribution(y_train)
        self._report_misclassifications(X_test, y_test, y_pred)

        # threshold tuning (uniform)
        self._search_thresholds(X_test, probabilities_test, y_test)

        from sklearn.metrics import hamming_loss, f1_score

        exact_match = accuracy_score(y_test_bin, y_pred)
        hamming     = 1 - hamming_loss(y_test_bin, y_pred)
        macro_f1    = f1_score(y_test_bin, y_pred, average="macro",  zero_division=0)
        micro_f1    = f1_score(y_test_bin, y_pred, average="micro",  zero_division=0)

        logger.info("=" * 70)
        logger.info(f"Exact Match Accuracy : {exact_match * 100:.2f}%")
        logger.info(f"Hamming Score        : {hamming     * 100:.2f}%  ← PRIMARY metric")
        logger.info(f"Macro F1             : {macro_f1    * 100:.2f}%")
        logger.info(f"Micro F1             : {micro_f1    * 100:.2f}%")
        logger.info("-" * 70)
        logger.info("Per-Intent Scores (F1 | Accuracy):")

        for idx, intent in enumerate(self.mlb.classes_):
            intent_acc = accuracy_score(y_test_bin[:, idx], y_pred[:, idx])
            intent_f1  = f1_score(y_test_bin[:, idx], y_pred[:, idx], zero_division=0)
            logger.info(f"  {intent:<35} F1: {intent_f1 * 100:.2f}%  |  Acc: {intent_acc * 100:.2f}%")

        logger.info("=" * 70)

        self.save_model()

    # ------------------------
    # Diagnostics helpers
    # ------------------------

    def _log_label_distribution(self, labels: List[List[str]]) -> None:
        """Print count of examples per intent in the supplied label set."""
        counts: Dict[str, int] = {}
        for lab in labels:
            for intent in lab:
                counts[intent] = counts.get(intent, 0) + 1
        sorted_counts = sorted(counts.items(), key=lambda x: x[1])
        logger.info("Label distribution (fewest to most examples):")
        for intent, cnt in sorted_counts:
            logger.info(f"  {intent:<30} {cnt}")

    def _report_misclassifications(self, X_test: List[str], y_test: List[List[str]], y_pred_bin: Any) -> None:
        """Log a few test examples where predictions did not exactly match labels."""
        if hasattr(y_pred_bin, 'tolist'):
            y_pred = y_pred_bin.tolist()
        else:
            y_pred = y_pred_bin
        mismatches = []
        for i, (query, true_vec, pred_vec) in enumerate(zip(X_test, y_test, y_pred)):
            true_intents = set(true_vec)
            pred_intents = {self.mlb.classes_[idx] for idx, val in enumerate(pred_vec) if val}
            if true_intents != pred_intents:
                mismatches.append((query, true_intents, pred_intents))
        logger.info(f"Found {len(mismatches)} mismatched test samples (exact match). Showing up to 10 examples:")
        for query, true_intents, pred_intents in mismatches[:10]:
            logger.info(f"  Query: {query}")
            logger.info(f"    True : {true_intents}")
            logger.info(f"    Pred : {pred_intents}")

    def _search_thresholds(self, X_test: List[str], probs: Any, y_test: List[List[str]]) -> None:
        """Evaluate uniform thresholds to see impact on exact-match accuracy."""
        best_thresh = None
        best_score = -1.0
        from sklearn.metrics import accuracy_score
        # convert y_test to binary array using same mlb
        y_test_bin = self.mlb.transform(y_test)
        for thresh in [i/100 for i in range(10, 91, 5)]:
            preds_bin = []
            for vec in probs:
                flags = [1 if p > thresh else 0 for p in vec]
                if sum(flags) == 0:
                    # choose highest probability if none pass threshold
                    flags[np.argmax(vec)] = 1
                preds_bin.append(flags)
            score = accuracy_score(y_test_bin, preds_bin)
            if score > best_score:
                best_score = score
                best_thresh = thresh
        logger.info(f"Threshold search: best uniform threshold {best_thresh:.2f} => exact-match {best_score*100:.2f}%")

    # ========================
    # PREDICTION
    # ========================

    def predict_intents(self, query: str) -> List[str]:
        """Hybrid ML + keyword multi-intent prediction."""
        if not self.vectorizer or not self.classifier:
            raise ValueError("Model not loaded. Run train() first.")

        query_lower = query.lower()

        # Step 1: ML prediction
        X = self.vectorizer.transform([query])
        probabilities = self.classifier.predict_proba(X)[0]

        query_words = len(query.split())
        if query_words > 15:
            threshold = 0.25
        elif query_words > 10:
            threshold = 0.30
        else:
            threshold = 0.35

        predicted = [
            self.mlb.classes_[idx]
            for idx, prob in enumerate(probabilities)
            if prob > threshold
        ]

        if not predicted:
            predicted.append(self.mlb.classes_[np.argmax(probabilities)])

        # Step 2: Keyword enhancement
        keyword_matched = []

        multi_indicators = [" and ", " also ", " then ", " additionally ", " as well as ", " along with "]
        has_multi_indicator = any(ind in query_lower for ind in multi_indicators)

        if has_multi_indicator:
            for intent, config in self.intent_mappings.items():
                for trigger in config.get("multi_triggers", []):
                    if trigger in query_lower:
                        keyword_matched.append(intent)
                        break

        for intent, config in self.intent_mappings.items():
            for kw in config["keywords"]:
                if kw in query_lower and intent not in keyword_matched:
                    keyword_matched.append(intent)
                    break

        combined = list(set(predicted + keyword_matched))

        if len(keyword_matched) >= 2:
            combined = keyword_matched + [p for p in predicted if p not in keyword_matched]

        combined = self._resolve_intent_conflicts(query_lower, combined)
        return combined[:3]

    def _resolve_intent_conflicts(self, query_lower: str, intents: List[str]) -> List[str]:
        """Suppress spurious co-intents."""
        resolved = list(intents)

        # Suppress get_account_balance when get_account_summary is present
        if "get_account_summary" in resolved and "get_account_balance" in resolved:
            balance_signals = ["balance", "available", "current balance"]
            if not any(sig in query_lower for sig in balance_signals):
                resolved = [i for i in resolved if i != "get_account_balance"]

        # Suppress fetch_gst_dues when pay_gst is explicitly requested
        if "pay_gst" in resolved and "fetch_gst_dues" in resolved:
            if "dues" not in query_lower and "pending" not in query_lower:
                resolved = [i for i in resolved if i != "fetch_gst_dues"]

        # Suppress fetch_epf_dues when pay_epf is explicitly requested
        if "pay_epf" in resolved and "fetch_epf_dues" in resolved:
            if "dues" not in query_lower and "pending" not in query_lower:
                resolved = [i for i in resolved if i != "fetch_epf_dues"]

        # Suppress fetch_esic_dues when pay_esic is explicitly requested
        if "pay_esic" in resolved and "fetch_esic_dues" in resolved:
            if "dues" not in query_lower and "pending" not in query_lower:
                resolved = [i for i in resolved if i != "fetch_esic_dues"]

        # Suppress calculate_gst when query is purely a breakdown request
        if "gst_breakdown" in resolved and "calculate_gst" in resolved:
            calc_signals = ["calculate gst", "compute gst", "find gst", "add gst",
                            "how much gst", "and calculate", "also calculate"]
            if not any(sig in query_lower for sig in calc_signals):
                resolved = [i for i in resolved if i != "calculate_gst"]

        # Suppress compare_rates firing calculate_gst for every rate
        # Keep `calculate_gst` when user explicitly asked to calculate (e.g. "calculate gst"),
        # otherwise remove it to avoid redundant per-rate calculations.
        if "compare_rates" in resolved and "calculate_gst" in resolved:
            calc_signals = [
                "calculate gst", "calculate", "compute gst", "find gst", "add gst",
                "how much gst", "and calculate", "also calculate", "calculate on", "gst on"
            ]
            if not any(sig in query_lower for sig in calc_signals):
                resolved = [i for i in resolved if i != "calculate_gst"]

        # Suppress company sub-intents when only company_guide is needed
        if "company_guide" in resolved:
            doc_signals     = ["document", "checklist", "what documents", "required documents"]
            process_signals = ["how long", "timeline", "processing time", "approval", "how many days"]
            if not any(sig in query_lower for sig in doc_signals):
                resolved = [i for i in resolved if i != "company_documents"]
            if not any(sig in query_lower for sig in process_signals):
                resolved = [i for i in resolved if i != "company_process"]

        return resolved

    # ========================
    # ENTITY EXTRACTION
    # ========================

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract banking entities from query."""
        entities = {}
        cleaned_query = query

        # GSTIN
        gstin_match = re.search(self.entity_patterns["gstin"], query)
        if gstin_match:
            entities["gstin"] = gstin_match.group(0)
            cleaned_query = re.sub(self.entity_patterns["gstin"], "", cleaned_query, flags=re.IGNORECASE)

        # PAN
        pan_match = re.search(self.entity_patterns["pan"], query)
        if pan_match:
            entities["pan"] = pan_match.group(0)
            cleaned_query = re.sub(self.entity_patterns["pan"], "", cleaned_query, flags=re.IGNORECASE)

        # IFSC
        ifsc_match = re.search(self.entity_patterns["ifsc"], query)
        if ifsc_match:
            entities["ifsc_code"] = ifsc_match.group(0)

        # Dates
        date_matches = re.findall(self.entity_patterns["date"], query)
        if date_matches:
            entities["from_date"] = date_matches[0]
            if len(date_matches) > 1:
                entities["to_date"] = date_matches[1]

        # Month
        month_match = re.search(self.entity_patterns["month"], query)
        if month_match:
            entities["month"] = month_match.group(0)

        # Percentages
        percent_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", cleaned_query)
        if percent_matches:
            entities["gst_rates"] = [float(p) for p in percent_matches]
            entities["gst_rate"]  = float(percent_matches[0])
            for m in percent_matches:
                cleaned_query = cleaned_query.replace(f"{m}%", "").replace(f"{m} percent", "")

        # Amounts
        amount_matches = re.findall(r"(?:₹|rs\.?|inr|rupees?)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)", cleaned_query)
        if amount_matches:
            amounts = [float(a.replace(",", "")) for a in amount_matches if float(a.replace(",", "")) > 0]
            if amounts:
                entities["amounts"]      = amounts
                entities["amount"]       = amounts[0]
                entities["base_amount"]  = amounts[0]
            if len(amounts) > 1:
                entities["total_amount"] = amounts[1]

        # Account number (long digit string not already matched)
        acct_match = re.search(r"\b(\d{9,18})\b", cleaned_query)
        if acct_match and "account_number" not in entities:
            entities["account_number"] = acct_match.group(1)

        # Payment mode
        for mode in ["NEFT", "RTGS", "IMPS", "UPI"]:
            if mode.lower() in query.lower():
                entities["payment_mode"] = mode
                break

        # Transaction ID
        txn_match = re.search(r"\b(TXN\w+)\b", query, re.IGNORECASE)
        if txn_match:
            entities["transaction_id"] = txn_match.group(1)

        # Intra / Inter state
        if "inter" in query.lower() or "interstate" in query.lower():
            entities["is_intra_state"] = False
        elif "intra" in query.lower() or "intrastate" in query.lower():
            entities["is_intra_state"] = True

        return entities

    # ========================
    # FULL PIPELINE
    # ========================

    def process_query(self, user_message: str) -> Dict[str, Any]:
        """Detect intents, extract entities, and build tool calls."""
        detected_intents = self.predict_intents(user_message)
        entities         = self.extract_entities(user_message)

        logger.info(f"Detected intents : {detected_intents}")
        logger.info(f"Extracted entities: {entities}")

        tool_calls = []

        amount          = entities.get("amount") or entities.get("base_amount")
        gst_rates       = entities.get("gst_rates", [18.0])
        gstin           = entities.get("gstin")
        pan             = entities.get("pan")
        account_number  = entities.get("account_number", "")
        transaction_id  = entities.get("transaction_id", "")
        month           = entities.get("month", "")
        from_date       = entities.get("from_date", "")
        to_date         = entities.get("to_date", "")
        payment_mode    = entities.get("payment_mode", "NEFT")

        # ── CORE PAYMENT ──────────────────────────────────────────────
        if "initiate_payment" in detected_intents:
            tool_calls.append({"tool_name": "initiate_payment", "parameters": {
                "beneficiary_id": entities.get("beneficiary_id", ""),
                "amount":         amount or 0,
                "payment_mode":   payment_mode,
            }})

        if "get_payment_status" in detected_intents and transaction_id:
            tool_calls.append({"tool_name": "get_payment_status", "parameters": {"transaction_id": transaction_id}})

        if "cancel_payment" in detected_intents and transaction_id:
            tool_calls.append({"tool_name": "cancel_payment", "parameters": {"transaction_id": transaction_id}})

        if "retry_payment" in detected_intents and transaction_id:
            tool_calls.append({"tool_name": "retry_payment", "parameters": {"transaction_id": transaction_id}})

        if "get_payment_receipt" in detected_intents and transaction_id:
            tool_calls.append({"tool_name": "get_payment_receipt", "parameters": {"transaction_id": transaction_id}})

        if "validate_beneficiary" in detected_intents:
            tool_calls.append({"tool_name": "validate_beneficiary", "parameters": {
                "account_number": account_number,
                "ifsc_code":      entities.get("ifsc_code", ""),
            }})

        # ── UPLOAD PAYMENT ────────────────────────────────────────────
        if "upload_bulk_payment" in detected_intents:
            tool_calls.append({"tool_name": "upload_bulk_payment", "parameters": {
                "file_name":   entities.get("file_name", ""),
                "file_base64": "",
                "file_format": "CSV",
            }})

        if "validate_payment_file" in detected_intents:
            tool_calls.append({"tool_name": "validate_payment_file", "parameters": {
                "upload_id": entities.get("upload_id", ""),
            }})

        # ── B2B ───────────────────────────────────────────────────────
        if "onboard_business_partner" in detected_intents:
            tool_calls.append({"tool_name": "onboard_business_partner", "parameters": {
                "company_name":  entities.get("company_name", ""),
                "gstin":         gstin or "",
                "pan":           pan or "",
                "contact_email": entities.get("contact_email", ""),
                "contact_phone": entities.get("contact_phone", ""),
            }})

        if "send_invoice" in detected_intents:
            tool_calls.append({"tool_name": "send_invoice", "parameters": {
                "partner_id":     entities.get("partner_id", ""),
                "invoice_number": entities.get("invoice_number", ""),
                "invoice_date":   from_date,
                "due_date":       to_date,
                "amount":         amount or 0,
            }})

        if "get_received_invoices" in detected_intents:
            tool_calls.append({"tool_name": "get_received_invoices", "parameters": {"status": "ALL"}})

        if "acknowledge_payment" in detected_intents:
            tool_calls.append({"tool_name": "acknowledge_payment", "parameters": {
                "invoice_id":     entities.get("invoice_id", ""),
                "transaction_id": transaction_id,
            }})

        if "create_proforma_invoice" in detected_intents:
            tool_calls.append({"tool_name": "create_proforma_invoice", "parameters": {
                "partner_id":    entities.get("partner_id", ""),
                "validity_date": to_date,
                "amount":        amount or 0,
                "description":   "",
            }})

        if "create_cd_note" in detected_intents:
            tool_calls.append({"tool_name": "create_cd_note", "parameters": {
                "partner_id":          entities.get("partner_id", ""),
                "note_type":           "CREDIT",
                "original_invoice_id": entities.get("invoice_id", ""),
                "amount":              amount or 0,
                "reason":              "",
            }})

        if "create_purchase_order" in detected_intents:
            tool_calls.append({"tool_name": "create_purchase_order", "parameters": {
                "partner_id":    entities.get("partner_id", ""),
                "po_date":       from_date,
                "delivery_date": to_date,
                "amount":        amount or 0,
                "description":   "",
            }})

        # ── INSURANCE ─────────────────────────────────────────────────
        if "fetch_insurance_dues" in detected_intents:
            tool_calls.append({"tool_name": "fetch_insurance_dues", "parameters": {}})

        if "pay_insurance_premium" in detected_intents:
            tool_calls.append({"tool_name": "pay_insurance_premium", "parameters": {
                "policy_number": entities.get("policy_number", ""),
                "amount":        amount or 0,
            }})

        if "get_insurance_payment_history" in detected_intents:
            tool_calls.append({"tool_name": "get_insurance_payment_history", "parameters": {}})

        # ── BANK STATEMENT ────────────────────────────────────────────
        if "fetch_bank_statement" in detected_intents:
            tool_calls.append({"tool_name": "fetch_bank_statement", "parameters": {
                "account_number": account_number,
                "from_date":      from_date,
                "to_date":        to_date,
            }})

        if "download_bank_statement" in detected_intents:
            tool_calls.append({"tool_name": "download_bank_statement", "parameters": {
                "account_number": account_number,
                "from_date":      from_date,
                "to_date":        to_date,
                "format":         "PDF",
            }})

        if "get_account_balance" in detected_intents:
            tool_calls.append({"tool_name": "get_account_balance", "parameters": {
                "account_number": account_number,
            }})

        if "get_transaction_history" in detected_intents:
            tool_calls.append({"tool_name": "get_transaction_history", "parameters": {
                "account_number": account_number,
                "from_date":      from_date,
                "to_date":        to_date,
            }})

        # ── CUSTOM / SEZ ──────────────────────────────────────────────
        if "pay_custom_duty" in detected_intents:
            tool_calls.append({"tool_name": "pay_custom_duty", "parameters": {
                "bill_of_entry_number": entities.get("bill_of_entry_number", ""),
                "amount":               amount or 0,
                "port_code":            entities.get("port_code", ""),
                "importer_code":        entities.get("importer_code", ""),
            }})

        if "track_custom_duty_payment" in detected_intents and transaction_id:
            tool_calls.append({"tool_name": "track_custom_duty_payment", "parameters": {"transaction_id": transaction_id}})

        if "get_custom_duty_history" in detected_intents:
            tool_calls.append({"tool_name": "get_custom_duty_history", "parameters": {}})

        # ── GST ───────────────────────────────────────────────────────
        if "fetch_gst_dues" in detected_intents and gstin:
            tool_calls.append({"tool_name": "fetch_gst_dues", "parameters": {"gstin": gstin}})

        if "pay_gst" in detected_intents and gstin:
            tool_calls.append({"tool_name": "pay_gst", "parameters": {
                "gstin":          gstin,
                "challan_number": entities.get("challan_number", ""),
                "amount":         amount or 0,
                "tax_type":       entities.get("tax_type", "CGST"),
            }})

        if "create_gst_challan" in detected_intents and gstin:
            tool_calls.append({"tool_name": "create_gst_challan", "parameters": {
                "gstin":         gstin,
                "return_period": entities.get("return_period", month.replace("-", "") if month else ""),
            }})

        if "get_gst_payment_history" in detected_intents and gstin:
            tool_calls.append({"tool_name": "get_gst_payment_history", "parameters": {"gstin": gstin}})

        # ── ESIC ──────────────────────────────────────────────────────
        if "fetch_esic_dues" in detected_intents:
            tool_calls.append({"tool_name": "fetch_esic_dues", "parameters": {
                "establishment_code": entities.get("establishment_code", ""),
                "month":              month,
            }})

        if "pay_esic" in detected_intents:
            tool_calls.append({"tool_name": "pay_esic", "parameters": {
                "establishment_code": entities.get("establishment_code", ""),
                "month":              month,
                "amount":             amount or 0,
            }})

        if "get_esic_payment_history" in detected_intents:
            tool_calls.append({"tool_name": "get_esic_payment_history", "parameters": {
                "establishment_code": entities.get("establishment_code", ""),
            }})

        # ── EPF ───────────────────────────────────────────────────────
        if "fetch_epf_dues" in detected_intents:
            tool_calls.append({"tool_name": "fetch_epf_dues", "parameters": {
                "establishment_id": entities.get("establishment_id", ""),
                "month":            month,
            }})

        if "pay_epf" in detected_intents:
            tool_calls.append({"tool_name": "pay_epf", "parameters": {
                "establishment_id": entities.get("establishment_id", ""),
                "month":            month,
                "amount":           amount or 0,
            }})

        if "get_epf_payment_history" in detected_intents:
            tool_calls.append({"tool_name": "get_epf_payment_history", "parameters": {
                "establishment_id": entities.get("establishment_id", ""),
            }})

        # ── PAYROLL ───────────────────────────────────────────────────
        if "fetch_payroll_summary" in detected_intents:
            tool_calls.append({"tool_name": "fetch_payroll_summary", "parameters": {"month": month}})

        if "process_payroll" in detected_intents:
            tool_calls.append({"tool_name": "process_payroll", "parameters": {
                "month":          month,
                "account_number": account_number,
                "approved_by":    entities.get("approved_by", ""),
            }})

        if "get_payroll_history" in detected_intents:
            tool_calls.append({"tool_name": "get_payroll_history", "parameters": {}})

        # ── TAXES ─────────────────────────────────────────────────────
        if "fetch_tax_dues" in detected_intents and pan:
            tool_calls.append({"tool_name": "fetch_tax_dues", "parameters": {"pan": pan}})

        if "pay_direct_tax" in detected_intents and pan:
            tool_calls.append({"tool_name": "pay_direct_tax", "parameters": {
                "pan":             pan,
                "tax_type":        entities.get("tax_type", "TDS"),
                "assessment_year": entities.get("assessment_year", "2026-27"),
                "amount":          amount or 0,
                "challan_type":    entities.get("challan_type", "281"),
            }})

        if "pay_state_tax" in detected_intents:
            tool_calls.append({"tool_name": "pay_state_tax", "parameters": {
                "state":             entities.get("state", ""),
                "tax_category":      entities.get("tax_category", "Professional Tax"),
                "amount":            amount or 0,
                "assessment_period": entities.get("assessment_period", ""),
            }})

        if "pay_bulk_tax" in detected_intents:
            tool_calls.append({"tool_name": "pay_bulk_tax", "parameters": {
                "file_name":   entities.get("file_name", ""),
                "file_base64": "",
                "tax_type":    entities.get("tax_type", "TDS"),
            }})

        if "get_tax_payment_history" in detected_intents and pan:
            tool_calls.append({"tool_name": "get_tax_payment_history", "parameters": {"pan": pan}})

        # ── ACCOUNT MANAGEMENT ────────────────────────────────────────
        if "get_account_summary" in detected_intents:
            tool_calls.append({"tool_name": "get_account_summary", "parameters": {}})

        if "get_account_details" in detected_intents:
            tool_calls.append({"tool_name": "get_account_details", "parameters": {"account_number": account_number}})

        if "get_linked_accounts" in detected_intents:
            tool_calls.append({"tool_name": "get_linked_accounts", "parameters": {}})

        if "set_default_account" in detected_intents:
            tool_calls.append({"tool_name": "set_default_account", "parameters": {"account_number": account_number}})

        # ── TRANSACTION & HISTORY ─────────────────────────────────────
        if "search_transactions" in detected_intents:
            tool_calls.append({"tool_name": "search_transactions", "parameters": {
                "from_date": from_date, "to_date": to_date,
            }})

        if "get_transaction_details" in detected_intents and transaction_id:
            tool_calls.append({"tool_name": "get_transaction_details", "parameters": {"transaction_id": transaction_id}})

        if "download_transaction_report" in detected_intents:
            tool_calls.append({"tool_name": "download_transaction_report", "parameters": {
                "from_date": from_date, "to_date": to_date, "format": "XLSX",
            }})

        if "get_pending_transactions" in detected_intents:
            tool_calls.append({"tool_name": "get_pending_transactions", "parameters": {}})

        # ── DUES & REMINDERS ──────────────────────────────────────────
        if "get_upcoming_dues" in detected_intents:
            tool_calls.append({"tool_name": "get_upcoming_dues", "parameters": {"days_ahead": 30}})

        if "get_overdue_payments" in detected_intents:
            tool_calls.append({"tool_name": "get_overdue_payments", "parameters": {}})

        if "set_payment_reminder" in detected_intents:
            tool_calls.append({"tool_name": "set_payment_reminder", "parameters": {
                "title":    entities.get("reminder_title", ""),
                "due_date": to_date or from_date,
            }})

        if "get_reminder_list" in detected_intents:
            tool_calls.append({"tool_name": "get_reminder_list", "parameters": {}})

        if "delete_reminder" in detected_intents:
            tool_calls.append({"tool_name": "delete_reminder", "parameters": {
                "reminder_id": entities.get("reminder_id", ""),
            }})

        # ── DASHBOARD & ANALYTICS ─────────────────────────────────────
        if "get_dashboard_summary" in detected_intents:
            tool_calls.append({"tool_name": "get_dashboard_summary", "parameters": {}})

        if "get_spending_analytics" in detected_intents:
            tool_calls.append({"tool_name": "get_spending_analytics", "parameters": {
                "from_date": from_date, "to_date": to_date,
            }})

        if "get_cashflow_summary" in detected_intents:
            tool_calls.append({"tool_name": "get_cashflow_summary", "parameters": {"month": month}})

        if "get_monthly_report" in detected_intents:
            tool_calls.append({"tool_name": "get_monthly_report", "parameters": {"month": month}})

        if "get_vendor_payment_summary" in detected_intents:
            tool_calls.append({"tool_name": "get_vendor_payment_summary", "parameters": {}})

        # ── COMPANY MANAGEMENT ────────────────────────────────────────
        if "get_company_profile" in detected_intents:
            tool_calls.append({"tool_name": "get_company_profile", "parameters": {}})

        if "update_company_details" in detected_intents:
            tool_calls.append({"tool_name": "update_company_details", "parameters": {
                "field": entities.get("field", ""),
                "value": entities.get("value", ""),
            }})

        if "get_gst_profile" in detected_intents:
            tool_calls.append({"tool_name": "get_gst_profile", "parameters": {}})

        if "get_authorized_signatories" in detected_intents:
            tool_calls.append({"tool_name": "get_authorized_signatories", "parameters": {}})

        if "manage_user_roles" in detected_intents:
            tool_calls.append({"tool_name": "manage_user_roles", "parameters": {
                "user_id": entities.get("user_id", ""),
                "role":    entities.get("role", "VIEWER"),
                "action":  entities.get("action", "ASSIGN"),
            }})

        # ── SUPPORT ───────────────────────────────────────────────────
        if "raise_support_ticket" in detected_intents:
            tool_calls.append({"tool_name": "raise_support_ticket", "parameters": {
                "category":    entities.get("ticket_category", "OTHER"),
                "subject":     entities.get("subject", ""),
                "description": user_message,
            }})

        if "get_ticket_history" in detected_intents:
            tool_calls.append({"tool_name": "get_ticket_history", "parameters": {"status": "ALL"}})

        if "chat_with_support" in detected_intents:
            tool_calls.append({"tool_name": "chat_with_support", "parameters": {
                "issue_summary": user_message[:200],
            }})

        if "get_contact_details" in detected_intents:
            tool_calls.append({"tool_name": "get_contact_details", "parameters": {"category": "GENERAL"}})

        # ── GST CALCULATOR (→ gst_client_manager) ─────────────────────
        if "calculate_gst" in detected_intents and amount:
            # If compare_rates is also requested, only calculate for the primary rate
            # and let `compare_rates` handle multi-rate comparisons to avoid duplicate
            # per-rate calculate calls.
            rates_for_calc = gst_rates if "compare_rates" not in detected_intents else [gst_rates[0]]
            for rate in rates_for_calc:
                tool_calls.append({"tool_name": "calculate_gst", "parameters": {
                    "base_amount": amount,
                    "gst_rate":    rate,
                }})

        if "reverse_gst" in detected_intents:
            amt = entities.get("total_amount") or amount
            if amt:
                tool_calls.append({"tool_name": "reverse_calculate_gst", "parameters": {
                    "total_amount": amt,
                    "gst_rate":     gst_rates[0],
                }})

        if "gst_breakdown" in detected_intents and amount:
            params = {"base_amount": amount, "gst_rate": gst_rates[0]}
            if "is_intra_state" in entities:
                params["is_intra_state"] = entities["is_intra_state"]
            tool_calls.append({"tool_name": "gst_breakdown", "parameters": params})

        if "compare_rates" in detected_intents and amount:
            rates_to_compare = gst_rates if len(gst_rates) > 1 else [5, 12, 18, 28]
            tool_calls.append({"tool_name": "compare_gst_rates", "parameters": {
                "base_amount": amount,
                "rates":       rates_to_compare,
            }})

        if "validate_gstin" in detected_intents and gstin:
            tool_calls.append({"tool_name": "validate_gstin", "parameters": {"gstin": gstin}})

        # ── ONBOARDING INFO (→ info_client_manager) ───────────────────
        if "company_guide" in detected_intents:
            tool_calls.append({"tool_name": "get_company_onboarding_guide", "parameters": {}})

        if "company_documents" in detected_intents:
            tool_calls.append({"tool_name": "get_company_required_documents", "parameters": {}})

        if "company_field" in detected_intents:
            tool_calls.append({"tool_name": "get_validation_formats", "parameters": {}})

        if "company_process" in detected_intents:
            tool_calls.append({"tool_name": "get_onboarding_faq", "parameters": {}})

        if "bank_guide" in detected_intents:
            tool_calls.append({"tool_name": "get_bank_onboarding_guide", "parameters": {}})

        if "vendor_guide" in detected_intents:
            tool_calls.append({"tool_name": "get_vendor_onboarding_guide", "parameters": {}})

        return {
            "intents_detected": detected_intents,
            "tool_calls":       tool_calls,
            "entities":         entities,
            "is_multi_intent":  len(detected_intents) > 1,
            "total_tools":      len(tool_calls),
        }

    # ========================
    # SAVE / LOAD
    # ========================

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        model_data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "mlb":        self.mlb,
            "version":    "3.0.0"
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
        self.mlb        = model_data["mlb"]
        logger.info(f"✓ Model loaded (v{model_data.get('version', '1.0.0')})")


# Global instance
intent_classifier = ProductionIntentClassifier()