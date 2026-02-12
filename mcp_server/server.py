"""
MCP Server using FastMCP
Implements GST calculation tools
"""
from fastmcp import FastMCP
from mcp_server.gst_calculator import GSTCalculator
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("GST Calculator")

# Initialize GST calculator
calculator = GSTCalculator()


@mcp.tool()
async def calculate_gst(base_amount: float, gst_rate: float) -> dict:
    """
    Calculate GST amount and total from base amount.
    
    Args:
        base_amount: The base amount before GST (e.g., 10000)
        gst_rate: The GST rate percentage (e.g., 18 for 18%)
    
    Returns:
        Dictionary with base_amount, gst_amount, total_amount, and gst_rate
    
    Example:
        calculate_gst(10000, 18)
        Returns: {"base_amount": 10000, "gst_amount": 1800, "total_amount": 11800, "gst_rate": 18}
    """
    logger.info(f"Calculating GST: base={base_amount}, rate={gst_rate}")
    try:
        result = await calculator.calculate_gst(base_amount, gst_rate)
        logger.info(f"GST calculated successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Error calculating GST: {e}")
        raise


@mcp.tool()
def reverse_calculate_gst(total_amount: float, gst_rate: float) -> dict:
    """
    Calculate base amount from total amount (reverse calculation).
    
    Args:
        total_amount: The total amount including GST (e.g., 11800)
        gst_rate: The GST rate percentage (e.g., 18 for 18%)
    
    Returns:
        Dictionary with total_amount, base_amount, gst_amount, and gst_rate
    
    Example:
        reverse_calculate_gst(11800, 18)
        Returns: {"total_amount": 11800, "base_amount": 10000, "gst_amount": 1800, "gst_rate": 18}
    """
    logger.info(f"Reverse calculating GST: total={total_amount}, rate={gst_rate}")
    try:
        result = calculator.reverse_calculate_gst(total_amount, gst_rate)
        logger.info(f"Reverse GST calculated successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in reverse calculation: {e}")
        raise


@mcp.tool()
def gst_breakdown(base_amount: float, gst_rate: float, is_intra_state: bool = True) -> dict:
    """
    Get detailed GST breakdown showing CGST, SGST, or IGST.
    
    Args:
        base_amount: The base amount before GST
        gst_rate: The GST rate percentage
        is_intra_state: True for intra-state (CGST+SGST), False for inter-state (IGST)
    
    Returns:
        Dictionary with base amounts and breakdown of CGST/SGST/IGST
    
    Example:
        gst_breakdown(10000, 18, True)
        Returns: {
            "base_amount": 10000,
            "gst_amount": 1800,
            "total_amount": 11800,
            "breakdown": {
                "type": "Intra-State",
                "cgst": 900,
                "sgst": 900,
                "igst": 0,
                "cgst_rate": 9,
                "sgst_rate": 9
            }
        }
    """
    logger.info(f"GST breakdown: base={base_amount}, rate={gst_rate}, intra_state={is_intra_state}")
    try:
        result = calculator.get_gst_breakdown(base_amount, gst_rate, is_intra_state)
        logger.info(f"Breakdown calculated successfully")
        return result
    except Exception as e:
        logger.error(f"Error in breakdown calculation: {e}")
        raise


@mcp.tool()
def compare_gst_rates(base_amount: float, rates: List[float]) -> dict:
    """
    Compare the same base amount with different GST rates.
    
    Args:
        base_amount: The base amount to compare
        rates: List of GST rates to compare (e.g., [5, 12, 18, 28])
    
    Returns:
        Dictionary with comparisons for each rate
    
    Example:
        compare_gst_rates(10000, [5, 12, 18])
        Returns: {
            "base_amount": 10000,
            "comparisons": [
                {"rate": 5, "gst_amount": 500, "total_amount": 10500, "difference_from_lowest": 0},
                {"rate": 12, "gst_amount": 1200, "total_amount": 11200, "difference_from_lowest": 700},
                {"rate": 18, "gst_amount": 1800, "total_amount": 11800, "difference_from_lowest": 1300}
            ]
        }
    """
    logger.info(f"Comparing GST rates: base={base_amount}, rates={rates}")
    try:
        result = calculator.compare_gst_rates(base_amount, rates)
        logger.info(f"Comparison completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in rate comparison: {e}")
        raise


@mcp.tool()
def validate_gstin(gstin: str) -> dict:
    """
    Validate GST Identification Number (GSTIN) format.
    
    Args:
        gstin: 15-character GSTIN to validate (e.g., "29ABCDE1234F1Z5")
    
    Returns:
        Dictionary with validation result and components if valid
    
    Example:
        validate_gstin("29ABCDE1234F1Z5")
        Returns: {
            "valid": True,
            "gstin": "29ABCDE1234F1Z5",
            "components": {
                "state_code": "29",
                "pan_number": "ABCDE1234F",
                "entity_number": "1",
                "default_letter": "Z",
                "checksum": "5"
            }
        }
    """
    logger.info(f"Validating GSTIN: {gstin}")
    try:
        result = calculator.validate_gstin(gstin)
        logger.info(f"GSTIN validation result: {result['valid']}")
        return result
    except Exception as e:
        logger.error(f"Error validating GSTIN: {e}")
        raise


# Main entry point for MCP server
if __name__ == "__main__":
    logger.info("Starting GST Calculator MCP Server...")
    mcp.run()