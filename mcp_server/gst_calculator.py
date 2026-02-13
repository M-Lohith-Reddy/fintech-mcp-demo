"""
GST Calculator - Business Logic
Can integrate with your external GST API
"""
from typing import Dict, List, Any, Optional
import httpx
import logging

logger = logging.getLogger(__name__)
  

class GSTCalculator:
    """GST calculation business logic"""
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        self.api_url = api_url
        self.api_key = api_key
        self.use_external_api = bool(api_url)
    
    async def calculate_gst(self, base_amount: float, gst_rate: float) -> Dict[str, Any]:
        """
        Calculate GST amount and total
        
        If GST API is configured, uses it. Otherwise uses local calculation.
        """
        if self.use_external_api:
            try:
                return await self._call_external_api(base_amount, gst_rate)
            except Exception as e:
                logger.warning(f"External API failed, using local calculation: {e}")
        
        # Local calculation (fallback or default)
        return self._calculate_locally(base_amount, gst_rate)
    
    async def _call_external_api(self, base_amount: float, gst_rate: float) -> Dict[str, Any]:
        """Call your external GST API"""
        async with httpx.AsyncClient() as client:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = await client.post(
                self.api_url,
                json={
                    "amount": base_amount,
                    "rate": gst_rate
                },
                headers=headers,
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
    
    def _calculate_locally(self, base_amount: float, gst_rate: float) -> Dict[str, Any]:
        """Local GST calculation"""
        if base_amount < 0 or gst_rate < 0:
            raise ValueError("Amount and rate must be positive")
        
        gst_amount = (base_amount * gst_rate) / 100
        total_amount = base_amount + gst_amount
        
        return {
            "base_amount": round(base_amount, 2),
            "gst_rate": gst_rate,
            "gst_amount": round(gst_amount, 2),
            "total_amount": round(total_amount, 2),
            "source": "external_api" if self.use_external_api else "local"
        }
    
    def reverse_calculate_gst(self, total_amount: float, gst_rate: float) -> Dict[str, Any]:
        """Calculate base amount from total (reverse calculation)"""
        if total_amount < 0 or gst_rate < 0:
            raise ValueError("Amount and rate must be positive")
        
        base_amount = total_amount / (1 + gst_rate / 100)
        gst_amount = total_amount - base_amount
        
        return {
            "total_amount": round(total_amount, 2),
            "gst_rate": gst_rate,
            "base_amount": round(base_amount, 2),
            "gst_amount": round(gst_amount, 2)
        }
    
    def get_gst_breakdown(
        self, 
        base_amount: float, 
        gst_rate: float, 
        is_intra_state: bool = True
    ) -> Dict[str, Any]:
        """Get detailed GST breakdown (CGST, SGST, IGST)"""
        calculation = self._calculate_locally(base_amount, gst_rate)
        
        if is_intra_state:
            # Intra-state: CGST + SGST
            cgst = calculation["gst_amount"] / 2
            sgst = calculation["gst_amount"] / 2
            
            breakdown = {
                "type": "Intra-State",
                "cgst": round(cgst, 2),
                "sgst": round(sgst, 2),
                "igst": 0,
                "cgst_rate": gst_rate / 2,
                "sgst_rate": gst_rate / 2,
                "igst_rate": 0
            }
        else:
            # Inter-state: IGST
            breakdown = {
                "type": "Inter-State",
                "cgst": 0,
                "sgst": 0,
                "igst": round(calculation["gst_amount"], 2),
                "cgst_rate": 0,
                "sgst_rate": 0,
                "igst_rate": gst_rate
            }
        
        return {
            **calculation,
            "breakdown": breakdown
        }
    
    def compare_gst_rates(self, base_amount: float, rates: List[float]) -> Dict[str, Any]:
        """Compare same amount with different GST rates"""
        if not rates:
            raise ValueError("Rates list cannot be empty")
        
        comparisons = []
        for rate in rates:
            calc = self._calculate_locally(base_amount, rate)
            comparisons.append({
                "rate": rate,
                **calc
            })
        
        # Sort by rate
        comparisons.sort(key=lambda x: x["rate"])
        
        # Add differences
        lowest_total = comparisons[0]["total_amount"]
        for comp in comparisons:
            comp["difference_from_lowest"] = round(
                comp["total_amount"] - lowest_total, 2
            )
        
        return {
            "base_amount": base_amount,
            "comparisons": comparisons,
            "lowest_rate": comparisons[0]["rate"],
            "highest_rate": comparisons[-1]["rate"],
            "max_difference": round(
                comparisons[-1]["total_amount"] - comparisons[0]["total_amount"], 2
            )
        }
    
    def validate_gstin(self, gstin: str) -> Dict[str, Any]:
        """Validate GSTIN format"""
        import re
        
        if not gstin or not isinstance(gstin, str):
            return {
                "valid": False,
                "error": "GSTIN must be a string"
            }
        
        # GSTIN format: 2-digit state + 10-char PAN + entity + Z + checksum
        gstin_pattern = re.compile(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}$')
        
        if not gstin_pattern.match(gstin):
            return {
                "valid": False,
                "error": "Invalid GSTIN format",
                "expected_format": "2-digit state code + 5 letters + 4 digits + 1 letter + 1 alphanumeric + Z + 1 alphanumeric"
            }
        
        # Extract components
        return {
            "valid": True,
            "gstin": gstin,
            "components": {
                "state_code": gstin[:2],
                "pan_number": gstin[2:12],
                "entity_number": gstin[12],
                "default_letter": gstin[13],
                "checksum": gstin[14]
            }
        }