"""
pricing.py
----------
Per-token pricing for AWS Bedrock models.

Langfuse uses these to compute the cost columns on its dashboard.
Update the rates when AWS changes Bedrock pricing.

Source: https://aws.amazon.com/bedrock/pricing/
"""

# model_id → {"input": $/token, "output": $/token}
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Amazon Nova Micro — $0.035 / 1M input, $0.14 / 1M output
    "amazon.nova-micro-v1:0": {
        "input":  0.035 / 1_000_000,
        "output": 0.14  / 1_000_000,
    },
    # Amazon Nova Lite
    "amazon.nova-lite-v1:0": {
        "input":  0.06 / 1_000_000,
        "output": 0.24 / 1_000_000,
    },
    # Amazon Nova Pro
    "amazon.nova-pro-v1:0": {
        "input":  0.80 / 1_000_000,
        "output": 3.20 / 1_000_000,
    },
    # Claude 3.5 Sonnet via Bedrock
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "input":  3.0  / 1_000_000,
        "output": 15.0 / 1_000_000,
    },
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> dict:
    """Return {"input_cost", "output_cost", "total_cost"} in USD."""
    rates = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    ic = input_tokens  * rates["input"]
    oc = output_tokens * rates["output"]
    return {
        "input_cost":  round(ic, 10),
        "output_cost": round(oc, 10),
        "total_cost":  round(ic + oc, 10),
    }