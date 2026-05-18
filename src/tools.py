from langchain_core.tools import tool


@tool
def get_delivery_status(order_id: str) -> str:
    """Get the delivery status of an order by order ID."""
    mock_orders = {
        "1234": "Your order #1234 is out for delivery and will arrive by 5pm today.",
        "5678": "Your order #5678 has been delivered successfully on May 15th.",
        "9999": "Your order #9999 is being processed and will ship within 24 hours.",
    }
    return mock_orders.get(order_id, f"Order #{order_id} not found in our system.")


@tool
def get_plan_pricing() -> str:
    """Get current Nexus pricing plans."""
    return """
    Nexus Plans:
    - Free: up to 3 users, 5 projects, 1GB storage
    - Pro: $12/user/month, unlimited projects, 50GB storage, priority support
    - Enterprise: custom pricing, SSO, advanced security, dedicated account manager
    """


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the conversation to a human support agent."""
    return f"I've escalated your case to our support team. Reason: {reason}. You will receive an email within 2 hours."